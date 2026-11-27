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
 * Parity gate for the PC-Wiener loop-restoration filter stage against
 * av2_convolve_symmetric_highbd_c() (av2/common/convolve.c), which is what
 * apply_pc_wiener_highbd() calls per 4x4 class block in its USE_CONVOLVE_SYM
 * build. The filter banks, the class -> filter selector and the tap geometry
 * all come from AVM itself (get_filter_set(), get_filter_selector(),
 * wienerns_simd_config_y == pcwiener_tap_config_luma), so the gate compares the
 * kernel against the reference with the reference's own tables.
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
#include "av2/common/restoration.h"

#include "avm_gpu/vulkan_gpu.h"
#include "avm_gpu/pcwiener.h"

/* Defined in av2/common/convolve.c; av2/common/restoration.c is the only
 * caller, so it has no public prototype. */
void av2_convolve_symmetric_highbd_c(const uint16_t *dgd, int stride,
                                     const NonsepFilterConfig *filter_config,
                                     const int16_t *filter, uint16_t *dst,
                                     int dst_stride, int bit_depth,
                                     int block_row_begin, int block_row_end,
                                     int block_col_begin, int block_col_end);

#define MAXW 96
#define MAXH 96
#define STRIDE 112   /* row stride of the padded plane */
#define PAD 8        /* border samples around the visible area */
#define ROWS (MAXH + 2 * PAD)
#define FILTER_SETS 4

static unsigned int rng_state = 0x51ed270bu;
static unsigned int rnd(void) {
  rng_state = rng_state * 1103515245u + 12345u;
  return rng_state >> 7;
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
  printf("%-14s %5s %8s %8s %s\n", "case", "bd", "cpu ms", "gpu ms",
         "parity");

  /* The NonsepFilterConfig apply_pc_wiener_highbd() builds for luma. */
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
  const int singleton_index =
      cfg.config[cfg.num_pixels - 1][NONSEP_BUF_POS];
  const int num_sym = cfg.num_pixels / 2;

  uint16_t *dgd_buf = malloc(sizeof(uint16_t) * STRIDE * ROWS);
  uint16_t *ref_buf = malloc(sizeof(uint16_t) * STRIDE * ROWS);
  uint16_t *gpu_buf = malloc(sizeof(uint16_t) * STRIDE * ROWS);
  uint8_t *class_id = malloc(((MAXH + 3) / 4) * ((MAXW + 3) / 4));
  if (!dgd_buf || !ref_buf || !gpu_buf || !class_id) return 1;

  /* The last geometry is not a multiple of 4, so the 4x4 class blocks at the
   * right/bottom edge are partial, exactly as in the C block loop. */
  static const int kGeom[][2] = { { MAXW, MAXH }, { 90, 86 } };
  int failures = 0;
  for (size_t g = 0; g < sizeof(kGeom) / sizeof(kGeom[0]); g++) {
   const int w = kGeom[g][0], h = kGeom[g][1];
   const int cid_stride = (w + 3) / 4;
   for (int bd = 10; bd <= 12; bd += 2) {
    const int maxv = (1 << bd) - 1;
    for (int set = 0; set < FILTER_SETS; set++) {
      const int16_t(*filters)[NUM_PC_WIENER_TAPS_LUMA] = get_filter_set(set);
      const uint8_t *selector = get_filter_selector(set);

      /* A plane with a 3-sample border of real data on every side, so the
       * filter can read outside the visible area exactly as the C does. */
      for (int y = 0; y < ROWS; y++) {
        for (int x = 0; x < STRIDE; x++) {
          dgd_buf[y * STRIDE + x] =
              (uint16_t)((rnd() % (unsigned)(maxv + 1)));
        }
      }
      const uint16_t *dgd = dgd_buf + PAD * STRIDE + PAD;
      uint16_t *ref = ref_buf + PAD * STRIDE + PAD;
      uint16_t *gpu = gpu_buf + PAD * STRIDE + PAD;
      memcpy(ref_buf, dgd_buf, sizeof(uint16_t) * STRIDE * ROWS);
      memcpy(gpu_buf, dgd_buf, sizeof(uint16_t) * STRIDE * ROWS);

      /* Random class per 4x4 block, evenly spread over the 256 classes. */
      for (int by = 0; by < (h + 3) / 4; by++) {
        for (int bx = 0; bx < cid_stride; bx++) {
          class_id[by * cid_stride + bx] = (uint8_t)(rnd() % 256);
        }
      }

      const double t0 = (double)clock() / (double)(CLOCKS_PER_SEC / 1000);
      for (int by = 0; by < (h + 3) / 4; by++) {
        for (int bx = 0; bx < cid_stride; bx++) {
          const int cls = class_id[by * cid_stride + bx];
          const int16_t *filter = filters[selector[cls]];
          const int r0 = by * 4, c0 = bx * 4;
          const int r1 = r0 + 4 < h ? r0 + 4 : h;
          const int c1 = c0 + 4 < w ? c0 + 4 : w;
          av2_convolve_symmetric_highbd_c(dgd, STRIDE, &cfg, filter, ref, STRIDE,
                                          bd, r0, r1, c0, c1);
        }
      }
      const double cpu_ms = (double)clock() / (double)(CLOCKS_PER_SEC / 1000) - t0;

      avm_gpu_pcwiener_params params;
      memset(&params, 0, sizeof(params));
      params.width = w;
      params.height = h;
      params.stride = STRIDE;
      params.dst_stride = STRIDE;
      params.lead_rows = PAD + (MAXH - h) / 2;
      params.lead_cols = PAD;
      params.bd = bd;
      params.class_id_stride = cid_stride;
      params.num_sym = num_sym;
      params.singleton_index = singleton_index;
      params.prec_bits = cfg.prec_bits;
      params.num_taps = NUM_PC_WIENER_TAPS_LUMA;
      params.config = cfg.config;
      /* The upload covers the visible area plus its border. */
      params.rows = h + 2 * params.lead_rows;

      const double t1 = (double)clock() / (double)(CLOCKS_PER_SEC / 1000);
      const int rc = avm_gpu_pcwiener_filter(ctx, pc, &params, dgd, class_id,
                                             selector, &filters[0][0], gpu,
                                             err);
      const double gpu_ms = (double)clock() / (double)(CLOCKS_PER_SEC / 1000) - t1;
      if (rc != 0) {
        fprintf(stderr, "avm_gpu pcwiener: %s\n", err);
        return 1;
      }

      int diff = 0;
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          if (ref[y * STRIDE + x] != gpu[y * STRIDE + x]) {
            if (diff < 3) {
              printf("  set %d bd %d (%d,%d): gpu %u != cpu %u\n", set, bd, x, y,
                     gpu[y * STRIDE + x], ref[y * STRIDE + x]);
            }
            diff++;
          }
        }
      }
      /* The border must come back untouched. */
      for (int y = -PAD; y < h + PAD; y++) {
        for (int x = -PAD; x < w + PAD; x++) {
          if (y >= 0 && y < h && x >= 0 && x < w) continue;
          if (gpu[y * STRIDE + x] != dgd_buf[(y + PAD) * STRIDE + (x + PAD)]) {
            diff++;
          }
        }
      }
      failures += diff != 0;
      char name[32];
      snprintf(name, sizeof(name), "%dx%d set %d", w, h, set);
      printf("%-14s %5d %8.3f %8.3f %s\n", name, bd, cpu_ms, gpu_ms,
             diff ? "MISMATCH" : "ok");
    }
   }
  }

  free(dgd_buf);
  free(ref_buf);
  free(gpu_buf);
  free(class_id);
  avm_gpu_pcwiener_destroy(ctx, pc);
  avm_gpu_ctx_destroy(ctx);
  printf("\n%s\n", failures ? "FAILED" : "ALL PARITY CASES PASSED");
  return failures ? 1 : 0;
}
