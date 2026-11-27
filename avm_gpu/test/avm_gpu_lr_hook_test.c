/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at
 * aomedia.org/license/patent-license/.
 */

/*
 * Gate for the loop-restoration seam (avm_gpu/vulkan_hook_lr.c), the function
 * av2/common/restoration.c::pc_wiener_stripe_highbd() calls with one procunit.
 *
 * It exercises the seam exactly as the call site does - procunit-sized (64/32/
 * partial) stripes with the three-sample border a restoration stripe carries,
 * tx-skip planes, classify-only and full filtering, 10- and 12-bit - and
 * compares against apply_pc_wiener_highbd() on identical inputs.
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
#include "avm_gpu/vulkan_hook_lr.h"

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

#define PAD 3
#define MAXW 64
#define MAXH 96
#define STRIDE (MAXW + 2 * PAD)
#define ROWS (MAXH + 2 * PAD)

static unsigned int rng_state = 0x2f1b3d55u;
static unsigned int rnd(void) {
  rng_state = rng_state * 1103515245u + 12345u;
  return rng_state >> 7;
}

int main(void) {
  /* The seam is inert without the flag; the hook reads it on first use. */
  setenv("AV2_GPU_LR", "1", 1);
  char err[AVM_GPU_ERR_LEN];
  avm_gpu_ctx *ctx = NULL;
  if (avm_gpu_ctx_create(&ctx, err) != 0) {
    fprintf(stderr, "avm_gpu: %s\n", err);
    return 77; /* skip: no Vulkan device */
  }
  printf("device: %s\n", avm_gpu_device_name(ctx));

  /* The seam reads the qval/tskip LUT and the normalisers from the buffers the
   * call site passes, so build a PcwienerBuffers the same way. */
  PcwienerBuffers buffers;
  memset(&buffers, 0, sizeof(buffers));
  buffers.buffer_width = MAXW + 2 * MAX_FEATURE_LENGTH;
  for (int i = 0; i < NUM_FEATURE_LINE_BUFFERS; i++) {
    buffers.feature_line_buffers[i] = calloc(buffers.buffer_width, sizeof(int16_t));
  }
  for (int i = 0; i < NUM_PC_WIENER_FEATURES; i++) {
    buffers.feature_sum_buffers[i] = calloc(buffers.buffer_width, sizeof(int));
  }
  buffers.tskip_sum_buffer = calloc(buffers.buffer_width, sizeof(int8_t));
  for (int t = 0; t < 256; t++) {
    for (int f = 0; f < NUM_PC_WIENER_FEATURES; f++) {
      buffers.qval_given_tskip_lut[t][f] =
          255 * ((int32_t)(rnd() % 60001) - 30000);
    }
  }
  for (int i = 0; i < NUM_PC_WIENER_FEATURES; i++) {
    buffers.feature_normalizers[i] = feature_normalizers_luma[i];
  }
  buffers.feature_normalizers[NUM_PC_WIENER_FEATURES] = tskip_normalizer;

  struct AV2Common *cm = calloc(1, sizeof(struct AV2Common));
  if (!cm) return 1;
  cm->features.has_lossless_segment = 0;

  const int16_t(*filters)[NUM_PC_WIENER_TAPS_LUMA] = get_filter_set(0);
  const uint8_t *selector = get_filter_selector(0);
  /* Tap *positions* (the config) and taps per filter are different counts:
   * the bank stores NUM_PC_WIENER_TAPS_LUMA taps, the config lists the 25
   * signed positions (12 pairs + the centre). */
  const int num_config = (int)(sizeof(wienerns_simd_config_y) /
                               sizeof(wienerns_simd_config_y[0]));

  uint16_t *src_buf = malloc(sizeof(uint16_t) * STRIDE * ROWS);
  uint16_t *ref_buf = malloc(sizeof(uint16_t) * STRIDE * ROWS);
  uint16_t *gpu_buf = malloc(sizeof(uint16_t) * STRIDE * ROWS);
  uint8_t *cid_ref = malloc(MAXW * MAXH);
  uint8_t *cid_gpu = malloc(MAXW * MAXH);
  uint8_t *tskip = malloc(MAXW * MAXH);
  uint8_t *mask = malloc(MAXW * MAXH);
  if (!src_buf || !ref_buf || !gpu_buf || !cid_ref || !cid_gpu || !tskip ||
      !mask) {
    return 1;
  }

  const char *tz = getenv("LRH_TSKIP_ZERO");
  const int tskip_zero = tz && *tz && *tz != '0';
  printf("%-20s %5s %8s %s\n", "case", "bd", "cpu ms", "parity");
  int failures = 0;

  static const struct {
    int w, h;
  } kCases[] = {
    { 64, 96 }, { 64, 64 }, { 32, 64 }, { 17, 40 }, { 64, 32 }, { 9, 13 },
  };

  for (int bd = 10; bd <= 12; bd += 2) {
    const int maxv = (1 << bd) - 1;
    for (size_t c = 0; c < sizeof(kCases) / sizeof(kCases[0]); c++) {
      const int w = kCases[c].w, h = kCases[c].h;
      const int cell_stride = (w + 3) / 4;
      /* The caller's buffers are MI-indexed, so their strides are wider than
       * the 4x4 cell grid the classifier works on (the encoder passes the
       * frame's MI stride). Both are different here on purpose. */
      const int cid_stride = cell_stride + 5;
      const int tskip_stride = cell_stride + 3;
      const int cell_rows = (h + 3) / 4;

      /* A restoration stripe: the visible area plus PAD samples of real data on
       * every side, which is what the C reads too. */
      {
        int v = maxv / 2;
        for (int y = 0; y < ROWS; y++) {
          for (int x = 0; x < STRIDE; x++) {
            const int busy = ((x >> 3) + (y >> 3)) & 1;
            const int step = busy ? 9 : 1;
            v += (int)(rnd() % (unsigned)step) - (step / 2);
            if (v < 0) v = 0;
            if (v > maxv) v = maxv;
            src_buf[y * STRIDE + x] = (uint16_t)v;
          }
        }
      }
      const uint16_t *src = src_buf + PAD * STRIDE + PAD;
      uint16_t *ref = ref_buf + PAD * STRIDE + PAD;
      uint16_t *gpu = gpu_buf + PAD * STRIDE + PAD;
      memcpy(ref_buf, src_buf, sizeof(uint16_t) * STRIDE * ROWS);
      memcpy(gpu_buf, src_buf, sizeof(uint16_t) * STRIDE * ROWS);
      for (int i = 0; i < tskip_stride * cell_rows; i++) {
        tskip[i] = (uint8_t)(rnd() & 1);
      }
      for (int i = 0; i < cell_stride * cell_rows; i++) mask[i] = 0;
      memset(cid_ref, 0, MAXW * MAXH);
      memset(cid_gpu, 0, MAXW * MAXH);

      const double t0 = (double)clock() / (double)(CLOCKS_PER_SEC / 1000);
      apply_pc_wiener_highbd(src, w, h, STRIDE, ref, STRIDE, tskip, tskip_stride,
                             cid_ref, cid_stride, false, bd, false, filters,
                             selector, &buffers, tskip_zero != 0, cm, NULL, 0,
                             0, 0, NULL);
      const double cpu_ms =
          (double)clock() / (double)(CLOCKS_PER_SEC / 1000) - t0;

      const int done = avm_gpu_hook_pcwiener(
          src, w, h, STRIDE, gpu, STRIDE, tskip, tskip_stride, cid_gpu,
          cid_stride, bd, 0, filters, selector,
          &buffers.qval_given_tskip_lut[0][0], buffers.feature_normalizers,
          pc_wiener_thresholds, pc_wiener_lut_to_class_index,
          PC_WIENER_PREC_FEATURE, pc_wiener_threshold_shift, tskip_zero, mask,
          cell_stride,
          NUM_PC_WIENER_TAPS_LUMA, wienerns_simd_config_y, num_config,
          PC_WIENER_PREC_FILTER);
      if (!done) {
        printf("  the seam declined %dx%d\n", w, h);
        failures++;
        continue;
      }

      int diff = 0;
      int cls_diff = 0;
      for (int by = 0; by < cell_rows; by++) {
        for (int bx = 0; bx < cell_stride; bx++) {
          const int i = by * cid_stride + bx;
          if (cid_ref[i] != cid_gpu[i]) {
            if (cls_diff < 3) {
              printf("  cls (%d,%d): gpu %u != cpu %u\n", bx, by, cid_gpu[i],
                     cid_ref[i]);
            }
            cls_diff++;
          }
        }
      }
      if (cls_diff) printf("  class mismatches %d\n", cls_diff);
      diff += cls_diff;
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          if (ref[y * STRIDE + x] != gpu[y * STRIDE + x]) {
            if (diff < 3) {
              printf("  (%d,%d): gpu %u != cpu %u\n", x, y,
                     gpu[y * STRIDE + x], ref[y * STRIDE + x]);
            }
            diff++;
          }
        }
      }
      failures += diff != 0;
      char name[32];
      snprintf(name, sizeof(name), "%dx%d stripe", w, h);
      printf("%-20s %5d %8.3f %s\n", name, bd, cpu_ms,
             diff ? "MISMATCH" : "ok");
    }
  }

  free(src_buf);
  free(ref_buf);
  free(gpu_buf);
  free(cid_ref);
  free(cid_gpu);
  free(tskip);
  free(mask);
  for (int i = 0; i < NUM_FEATURE_LINE_BUFFERS; i++) {
    free(buffers.feature_line_buffers[i]);
  }
  for (int i = 0; i < NUM_PC_WIENER_FEATURES; i++) {
    free(buffers.feature_sum_buffers[i]);
  }
  free(buffers.tskip_sum_buffer);
  free(cm);
  avm_gpu_ctx_destroy(ctx);
  printf("\n%s\n", failures ? "FAILED" : "ALL PARITY CASES PASSED");
  return failures ? 1 : 0;
}
