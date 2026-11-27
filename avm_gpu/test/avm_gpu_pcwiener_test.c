/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at aomedia.org/license/software-license/bsd-3-c-c/.  If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * aomedia.org/license/patent-license/.
 */

/*
 * End-to-end parity gate for PC-Wiener loop restoration on the GPU:
 * classification (kernels/pcwcls.comp) followed by filtering
 * (kernels/pcwiener.comp), compared with AVM's own
 * apply_pc_wiener_highbd() (av2/common/restoration.c).
 *
 * The reference is driven directly: the test owns a PcwienerBuffers, fills the
 * quantiser-dependent qval/tskip LUT itself, and calls apply_pc_wiener_highbd()
 * once with classify_only = true (reference class ids) and once with
 * classify_only = false (reference filtered plane). The GPU then has to
 * reproduce both: class ids from the kernel, then the plane from the kernel's
 * own class ids. A second comparison feeds the reference class ids to the
 * filter kernel to separate the two stages.
 *
 * Exit 0 = every case exact, 77 = no Vulkan device, 1 = a mismatch.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "av2/common/enums.h"
#include "av2/common/convolve.h"
#include "av2/common/av2_common_int.h"
#include "av2/common/restoration.h"

#include "avm_gpu/vulkan_gpu.h"
#include "avm_gpu/pcwiener.h"

/* av2/common/restoration.c; no public prototype. */
void apply_pc_wiener_highbd(
    const uint16_t *dgd, int width, int height, int stride, uint16_t *dst,
    int dst_stride, const uint8_t *tskip, int tskip_stride,
    uint8_t *wiener_class_id, int wiener_class_id_stride, bool is_uv,
    int bit_depth, bool classify_only,
    const int16_t (*pcwiener_filters_luma)[NUM_PC_WIENER_TAPS_LUMA],
    const uint8_t *filter_selector, PcwienerBuffers *buffers,
    bool tskip_zero_flag, const struct AV2Common *cm,
    MB_MODE_INFO **mbmi_ptr_procunit, int mi_stride, int ss_x, int ss_y,
    const bool *lossless_segment);

/* The classifier's accumulators are sized for one restoration procunit
 * (PC_WIENER_FEATURE_ACC_SIZE = (RESTORATION_PROC_UNIT_SIZE + MAX_FEATURE_LENGTH
 * + 1) / PC_WIENER_BLOCK_SIZE), i.e. 64 pixels wide. */
#define W 64
#define H 64
#define PAD 8
#define STRIDE (W + 2 * PAD)
#define ROWS (H + 2 * PAD)
#define CELL_STRIDE (W / 4)
#define CELL_ROWS (H / 4)

static unsigned int rng_state = 0x71c0ffeeu;
static unsigned int rnd(void) {
  rng_state = rng_state * 1103515245u + 12345u;
  return rng_state >> 7;
}

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(void) {
  char err[AVM_GPU_ERR_LEN];
  avm_gpu_ctx *ctx = NULL;
  if (avm_gpu_ctx_create(&ctx, err) != 0) {
    fprintf(stderr, "avm_gpu: %s\n", err);
    return 77; /* skip: no Vulkan device */
  }
  avm_gpu_pcwiener *pc = NULL;
  if (avm_gpu_pcwiener_create(ctx, &pc, err) != 0) {
    fprintf(stderr, "avm_gpu pcwiener: %s\n", err);
    avm_gpu_ctx_destroy(ctx);
    return 1;
  }
  printf("device: %s\n", avm_gpu_device_name(ctx));

  /* Buffers the classifier streams through (allocated here because the
   * allocation helpers in restoration.c are static). */
  PcwienerBuffers buffers;
  memset(&buffers, 0, sizeof(buffers));
  buffers.buffer_width = W + 2 * MAX_FEATURE_LENGTH;
  for (int i = 0; i < NUM_FEATURE_LINE_BUFFERS; i++) {
    buffers.feature_line_buffers[i] = calloc(buffers.buffer_width, sizeof(int16_t));
  }
  for (int i = 0; i < NUM_PC_WIENER_FEATURES; i++) {
    buffers.feature_sum_buffers[i] = calloc(buffers.buffer_width, sizeof(int));
  }
  buffers.tskip_sum_buffer = calloc(buffers.buffer_width, sizeof(int8_t));
  /* A plausible qval table: fill_qval_given_tskip_lut() stores 255 * qval with
   * qval in PC_WIENER_PREC_FEATURE units, i.e. +-30000 * 255 here, which is
   * the range that produces a wide spread of classes on this content. It is an
   * input to both sides, so only its range matters. */
  for (int t = 0; t < 256; t++) {
    for (int f = 0; f < NUM_PC_WIENER_FEATURES; f++) {
      buffers.qval_given_tskip_lut[t][f] =
          255 * ((int32_t)(rnd() % 60001) - 30000);
    }
  }
  buffers.prev_qindex = -1;
  buffers.prev_bit_depth = -1;

  struct AV2Common *cm = calloc(1, sizeof(struct AV2Common));
  if (!cm) return 1;
  cm->features.has_lossless_segment = 0;

  const NonsepFilterConfig cfg = { PC_WIENER_PREC_FILTER,
                                   (int)(sizeof(wienerns_simd_config_y) /
                                         sizeof(wienerns_simd_config_y[0])),
                                   0,
                                   wienerns_simd_config_y,
                                   NULL,
                                   0,
                                   0,
                                   1,
                                   0 };
  const int16_t(*filters)[NUM_PC_WIENER_TAPS_LUMA] = get_filter_set(0);
  const uint8_t *selector = get_filter_selector(0);

  uint16_t *dgd_buf = malloc(sizeof(uint16_t) * STRIDE * ROWS);
  uint16_t *ref_buf = malloc(sizeof(uint16_t) * STRIDE * ROWS);
  uint16_t *gpu_buf = malloc(sizeof(uint16_t) * STRIDE * ROWS);
  uint16_t *filt_ref_buf = malloc(sizeof(uint16_t) * STRIDE * ROWS);
  uint8_t *cid_ref = malloc((size_t)CELL_STRIDE * CELL_ROWS);
  uint8_t *cid_gpu = malloc((size_t)CELL_STRIDE * CELL_ROWS);
  uint8_t *tskip = malloc((size_t)CELL_STRIDE * CELL_ROWS);
  if (!dgd_buf || !ref_buf || !gpu_buf || !filt_ref_buf || !cid_ref ||
      !cid_gpu || !tskip) {
    return 1;
  }

  printf("%-22s %5s %9s %9s %9s %s\n", "case", "bd", "cpu ms", "cls ms",
         "filt ms", "parity");
  int failures = 0;

  for (int bd = 10; bd <= 12; bd += 2) {
    const int maxv = (1 << bd) - 1;
    for (int zero_tskip = 1; zero_tskip >= 0; zero_tskip--) {
      /* Smooth content (a bounded random walk), not white noise: the
       * classifier saturates on white noise and every block lands in the same
       * class, which would make this gate vacuous. Border samples are real
       * data so the classifier and the filter can read outside the visible
       * area as the C does. */
      {
        /* Activity varies in 8x8 tiles (flat / busy), so the features cover a
         * wide range and the classes spread out. */
        int v = maxv / 2;
        for (int y = 0; y < ROWS; y++) {
          for (int x = 0; x < STRIDE; x++) {
            const int busy = ((x >> 3) + (y >> 3)) & 1;
            const int step = busy ? 9 : 1;
            v += (int)(rnd() % (unsigned)step) - (step / 2);
            if (v < 0) v = 0;
            if (v > maxv) v = maxv;
            dgd_buf[y * STRIDE + x] = (uint16_t)v;
          }
        }
      }
      for (int i = 0; i < CELL_STRIDE * CELL_ROWS; i++) tskip[i] = (uint8_t)(rnd() & 1);
      const uint16_t *dgd = dgd_buf + PAD * STRIDE + PAD;
      uint16_t *ref = ref_buf + PAD * STRIDE + PAD;
      uint16_t *filt_ref = filt_ref_buf + PAD * STRIDE + PAD;
      uint16_t *gpu = gpu_buf + PAD * STRIDE + PAD;

      memset(cid_ref, 0, (size_t)CELL_STRIDE * CELL_ROWS);
      memset(cid_gpu, 0, (size_t)CELL_STRIDE * CELL_ROWS);
      memcpy(ref_buf, dgd_buf, sizeof(uint16_t) * STRIDE * ROWS);
      memcpy(filt_ref_buf, dgd_buf, sizeof(uint16_t) * STRIDE * ROWS);
      memcpy(gpu_buf, dgd_buf, sizeof(uint16_t) * STRIDE * ROWS);

      /* --- reference: classification, then filtering --- */
      const double t0 = now_ms();
      apply_pc_wiener_highbd(dgd, W, H, STRIDE, ref, STRIDE, tskip, CELL_STRIDE,
                             cid_ref, CELL_STRIDE, false, bd, true, filters,
                             selector, &buffers, zero_tskip != 0, cm, NULL, 0,
                             0, 0, NULL);
      apply_pc_wiener_highbd(dgd, W, H, STRIDE, filt_ref, STRIDE, tskip,
                             CELL_STRIDE, cid_ref, CELL_STRIDE, false, bd,
                             false, filters, selector, &buffers,
                             zero_tskip != 0, cm, NULL, 0, 0, 0, NULL);
      const double cpu_ms = now_ms() - t0;

      /* --- GPU: classification --- */
      avm_gpu_pcwiener_classify_params cp;
      memset(&cp, 0, sizeof(cp));
      cp.width = W;
      cp.height = H;
      cp.stride = STRIDE;
      cp.rows = ROWS;
      cp.lead_rows = PAD;
      cp.lead_cols = PAD;
      cp.bd = bd;
      cp.cell_stride = CELL_STRIDE;
      cp.cell_rows = CELL_ROWS;
      cp.tskip_stride = zero_tskip ? 0 : CELL_STRIDE;
      cp.qval_lut = &buffers.qval_given_tskip_lut[0][0];
      cp.normalizers = buffers.feature_normalizers;
      cp.thresholds = pc_wiener_thresholds;
      cp.lut_to_class = pc_wiener_lut_to_class_index;
      cp.prec_feature = PC_WIENER_PREC_FEATURE;
      cp.thr_shift = pc_wiener_threshold_shift;

      const double t1 = now_ms();
      if (avm_gpu_pcwiener_classify(ctx, pc, &cp, dgd, tskip, cid_gpu, err) !=
          0) {
        fprintf(stderr, "avm_gpu pcwiener classify: %s\n", err);
        return 1;
      }
      const double cls_ms = now_ms() - t1;

      /* The gate must not be vacuous: the reference has to spread the blocks
       * over many classes, otherwise any classifier would match it. */
      int seen[256] = { 0 };
      int distinct = 0;
      for (int i = 0; i < CELL_STRIDE * CELL_ROWS; i++) {
        if (!seen[cid_ref[i]]++) distinct++;
      }
      /* With tx-skip the LUT row varies per block through the tskip feature,
       * so the classes spread widely; without it every block reads LUT row 0,
       * and the class is dominated by that row's offsets (the C behaves the
       * same), so only require that the zero-tskip case is not a single
       * constant. */
      const int min_distinct = zero_tskip ? 1 : 16;
      if (distinct < min_distinct) {
        printf("  reference produced only %d distinct classes, gate would be "
               "vacuous\n",
               distinct);
        failures++;
        continue;
      }

      int cls_diff = 0;
      for (int i = 0; i < CELL_STRIDE * CELL_ROWS; i++) {
        if (cid_ref[i] != cid_gpu[i]) {
          if (cls_diff < 3) {
            printf("  (%d,%d): class gpu %u != cpu %u\n", i % CELL_STRIDE,
                   i / CELL_STRIDE, cid_gpu[i], cid_ref[i]);
          }
          cls_diff++;
        }
      }

      /* --- GPU: filtering, from the kernel's own class ids --- */
      avm_gpu_pcwiener_params fp;
      memset(&fp, 0, sizeof(fp));
      fp.width = W;
      fp.height = H;
      fp.stride = STRIDE;
      fp.dst_stride = STRIDE;
      fp.lead_rows = PAD;
      fp.lead_cols = PAD;
      fp.rows = ROWS;
      fp.bd = bd;
      fp.class_id_stride = CELL_STRIDE;
      fp.num_sym = cfg.num_pixels / 2;
      fp.singleton_index = cfg.config[cfg.num_pixels - 1][NONSEP_BUF_POS];
      fp.prec_bits = cfg.prec_bits;
      fp.num_taps = NUM_PC_WIENER_TAPS_LUMA;
      fp.config = cfg.config;

      const double t2 = now_ms();
      if (avm_gpu_pcwiener_filter(ctx, pc, &fp, dgd, cid_gpu, selector,
                                  &filters[0][0], gpu, err) != 0) {
        fprintf(stderr, "avm_gpu pcwiener filter: %s\n", err);
        return 1;
      }
      const double filt_ms = now_ms() - t2;

      int diff = 0;
      for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
          if (filt_ref[y * STRIDE + x] != gpu[y * STRIDE + x]) {
            if (diff < 3) {
              printf("  (%d,%d): gpu %u != cpu %u\n", x, y,
                     gpu[y * STRIDE + x], filt_ref[y * STRIDE + x]);
            }
            diff++;
          }
        }
      }

      /* The filter stage alone, fed the reference class ids. */
      memcpy(gpu_buf, filt_ref_buf, sizeof(uint16_t) * STRIDE * ROWS);
      int filt_only_diff = 0;
      if (avm_gpu_pcwiener_filter(ctx, pc, &fp, dgd, cid_ref, selector,
                                  &filters[0][0], gpu, err) != 0) {
        return 1;
      }
      for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
          if (filt_ref[y * STRIDE + x] != gpu[y * STRIDE + x]) filt_only_diff++;
        }
      }

      char name[32];
      snprintf(name, sizeof(name), "bd %d tskip %s", bd,
               zero_tskip ? "zero" : "real");
      const int bad = cls_diff != 0 || diff != 0 || filt_only_diff != 0;
      if (cls_diff || filt_only_diff) {
        printf("  class mismatches %d, filter-only mismatches %d\n", cls_diff,
               filt_only_diff);
      }
      failures += bad;
      printf("%-22s %5d %9.3f %9.3f %9.3f %s (%d classes)\n", name, bd,
             cpu_ms, cls_ms, filt_ms, bad ? "MISMATCH" : "ok", distinct);
    }
  }

  free(dgd_buf);
  free(ref_buf);
  free(gpu_buf);
  free(filt_ref_buf);
  free(cid_ref);
  free(cid_gpu);
  free(tskip);
  for (int i = 0; i < NUM_FEATURE_LINE_BUFFERS; i++) {
    free(buffers.feature_line_buffers[i]);
  }
  for (int i = 0; i < NUM_PC_WIENER_FEATURES; i++) {
    free(buffers.feature_sum_buffers[i]);
  }
  free(buffers.tskip_sum_buffer);
  free(cm);
  avm_gpu_pcwiener_destroy(ctx, pc);
  avm_gpu_ctx_destroy(ctx);
  printf("\n%s\n", failures ? "FAILED" : "ALL PARITY CASES PASSED");
  return failures ? 1 : 0;
}
