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
 * Parity gate for the Vulkan inter convolve kernel against
 * avm_highbd_convolve8_c (avm_dsp/avm_convolve.c).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "config/avm_dsp_rtcd.h"
#include "avm_dsp/blend.h"
#include "av2/common/blockd.h"
#include "av2/common/filter.h"

#include "avm_gpu/vulkan_gpu.h"
#include "avm_gpu/vulkan_inter.h"

/* av2/common/convolve.c; the single-reference prediction path the encoder
 * uses. */
void av2_highbd_convolve_2d_sr_c(const uint16_t *src, int src_stride,
                                 uint16_t *dst, int dst_stride, int w, int h,
                                 const InterpFilterParams *filter_params_x,
                                 const InterpFilterParams *filter_params_y,
                                 const int subpel_x_qn, const int subpel_y_qn,
                                 ConvolveParams *conv_params, int bd);

#define SRC_W 128
#define SRC_H 128
#define DST_STRIDE 64
#define BD 10

static unsigned int rng_state = 0x7f4a7c15u;
static unsigned int rnd(void) {
  rng_state = rng_state * 1664525u + 1013904223u;
  return rng_state >> 8;
}

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(void) {
  static const int kPos[6][2] = { { 0, 0 },  { 1, 0 }, { 0, 7 },
                                  { 8, 8 },  { 15, 15 }, { 3, 9 } };
  static const int kSizes[5] = { 4, 8, 16, 32, 64 };

  char err[AVM_GPU_ERR_LEN];
  avm_gpu_ctx *ctx = NULL;
  if (avm_gpu_ctx_create(&ctx, err) != 0) {
    fprintf(stderr, "avm_gpu: %s\n", err);
    return 77; /* skip: no Vulkan device */
  }
  avm_gpu_inter *inter = NULL;
  if (avm_gpu_inter_create(ctx, &inter, err) != 0) {
    fprintf(stderr, "avm_gpu inter: %s\n", err);
    avm_gpu_ctx_destroy(ctx);
    return 1;
  }
  printf("device: %s\n", avm_gpu_device_name(ctx));
  printf("%-8s %-6s %10s %10s %s\n", "subpel", "block", "cpu ms", "gpu ms",
         "parity");

  uint16_t *plane = malloc(sizeof(uint16_t) * SRC_W * SRC_H);
  if (!plane) return 1;
  for (int i = 0; i < SRC_W * SRC_H; i++) plane[i] = (uint16_t)(rnd() & 0x3ff);

  int failures = 0;
  uint16_t cpu[64 * DST_STRIDE], gpu[64 * DST_STRIDE];

  for (size_t p = 0; p < sizeof(kPos) / sizeof(kPos[0]); p++) {
    for (size_t s = 0; s < sizeof(kSizes) / sizeof(kSizes[0]); s++) {
      const int n = kSizes[s];
      const int sx = 5 + (int)(rnd() % 7);
      const int sy = 5 + (int)(rnd() % 7);

      memset(cpu, 0, sizeof(cpu));
      memset(gpu, 0, sizeof(gpu));

      const double t0 = now_ms();
      avm_highbd_convolve8_c(plane + sy * SRC_W + sx, SRC_W, cpu, DST_STRIDE,
                             av2_sub_pel_filters_8, kPos[p][0], 16, kPos[p][1],
                             16, n, n, BD);
      const double cpu_ms = now_ms() - t0;

      const double t1 = now_ms();
      if (avm_gpu_inter_convolve(ctx, inter, plane, SRC_W, SRC_H, sx, sy,
                                 (const int16_t *)av2_sub_pel_filters_8,
                                 kPos[p][0], kPos[p][1], n, n, BD, gpu,
                                 DST_STRIDE, err) != 0) {
        fprintf(stderr, "avm_gpu inter: %s\n", err);
        return 1;
      }
      const double gpu_ms = now_ms() - t1;

      int diff = 0;
      for (int y = 0; y < n; y++) {
        for (int x = 0; x < n; x++) {
          if (cpu[y * DST_STRIDE + x] != gpu[y * DST_STRIDE + x]) {
            if (diff < 3) {
              printf("  subpel(%d,%d) %dx%d (%d,%d): gpu %u != cpu %u\n",
                     kPos[p][0], kPos[p][1], n, n, x, y,
                     gpu[y * DST_STRIDE + x], cpu[y * DST_STRIDE + x]);
            }
            diff++;
          }
        }
      }
      failures += diff != 0;
      printf("(%2d,%2d)  %-6d %10.3f %10.3f %s\n", kPos[p][0], kPos[p][1], n,
             cpu_ms, gpu_ms, diff ? "MISMATCH" : "ok");
    }
  }

  /* Scaled reference prediction: the subpel step is no longer 16, so each
   * output pixel uses its own fractional phase. */
  static const int kSteps[5][2] = { { 16, 16 }, { 20, 16 }, { 16, 20 },
                                    { 24, 24 }, { 32, 32 } };
  static const int kSxTab[5] = { 3, 5, 7, 11, 13 };
  for (size_t st = 0; st < 5; st++) {
    for (size_t p = 0; p < sizeof(kPos) / sizeof(kPos[0]); p++) {
      for (size_t s = 0; s < 3; s++) {
        const int n = kSizes[s];
        const int sx = kSxTab[st];
        const int sy = 4;
        memset(cpu, 0, sizeof(cpu));
        memset(gpu, 0, sizeof(gpu));

        avm_highbd_convolve8_c(plane + sy * SRC_W + sx, SRC_W, cpu, DST_STRIDE,
                               av2_sub_pel_filters_8, kPos[p][0],
                               kSteps[st][0], kPos[p][1], kSteps[st][1], n, n,
                               BD);
        if (avm_gpu_inter_convolve_scaled(
                ctx, inter, plane, SRC_W, SRC_H, sx, sy,
                (const int16_t *)av2_sub_pel_filters_8, kPos[p][0], kPos[p][1],
                kSteps[st][0], kSteps[st][1], n, n, BD, gpu, DST_STRIDE,
                err) != 0) {
          fprintf(stderr, "avm_gpu inter scaled: %s\n", err);
          return 1;
        }

        int diff = 0;
        for (int y = 0; y < n; y++) {
          for (int x = 0; x < n; x++) {
            if (cpu[y * DST_STRIDE + x] != gpu[y * DST_STRIDE + x]) diff++;
          }
        }
        failures += diff != 0;
        printf("scaled %2d/%2d pos(%2d,%2d) %-3d %9s %s\n", kSteps[st][0],
               kSteps[st][1], kPos[p][0], kPos[p][1], n, "-",
               diff ? "MISMATCH" : "ok");
      }
    }
  }

  /* Compound prediction: A64 mask blend of two predictors. */
  uint16_t *src1p = malloc(sizeof(uint16_t) * SRC_W * SRC_H);
  uint8_t *maskbuf = malloc(2 * 64 * 2 * 64);
  if (!src1p || !maskbuf) return 1;
  for (int i = 0; i < SRC_W * SRC_H; i++) src1p[i] = (uint16_t)(rnd() & 0x3ff);

  for (int subw = 0; subw < 2; subw++) {
    for (int subh = 0; subh < 2; subh++) {
      for (size_t si = 0; si < sizeof(kSizes) / sizeof(kSizes[0]); si++) {
        const int n = kSizes[si];
        const int mw = subw ? 2 * n : n;
        const int mh = subh ? 2 * n : n;
        for (int i = 0; i < mw * mh; i++) maskbuf[i] = (uint8_t)(rnd() % 65);
        memset(cpu, 0, sizeof(cpu));
        memset(gpu, 0, sizeof(gpu));

        avm_highbd_blend_a64_mask_c(cpu, DST_STRIDE, plane, SRC_W, src1p,
                                    SRC_W, maskbuf, mw, n, n, subw, subh, BD);
        if (avm_gpu_inter_blend_a64(ctx, inter, plane, SRC_W, src1p, SRC_W,
                                    maskbuf, mw, n, n, subw, subh, BD, gpu,
                                    DST_STRIDE, err) != 0) {
          fprintf(stderr, "avm_gpu blend: %s\n", err);
          return 1;
        }
        int diff = 0;
        for (int y = 0; y < n; y++) {
          for (int x = 0; x < n; x++) {
            if (cpu[y * DST_STRIDE + x] != gpu[y * DST_STRIDE + x]) diff++;
          }
        }
        failures += diff != 0;
        printf("blend sub%dx%d %-3d %9s %s\n", subw, subh, n, "-",
               diff ? "MISMATCH" : "ok");
      }
    }
  }
  free(src1p);

  /* Weighted (CWP) compound: avm_highbd_cwp_c(). */
  static const int kCwp[5][2] = { { 38, 26 }, { 46, 18 }, { 64, 0 },
                                  { 0, 64 },  { 20, 44 } };
  uint16_t *pred2 = malloc(sizeof(uint16_t) * SRC_W * SRC_H);
  if (!pred2) return 1;
  for (int i = 0; i < SRC_W * SRC_H; i++) pred2[i] = (uint16_t)(rnd() & 0x3ff);
  for (size_t ci = 0; ci < 5; ci++) {
    for (size_t si = 0; si < sizeof(kSizes) / sizeof(kSizes[0]); si++) {
      const int n = kSizes[si];
      CWP_PARAMS cwp;
      cwp.fwd_offset = kCwp[ci][0];
      cwp.bck_offset = kCwp[ci][1];
      memset(cpu, 0, sizeof(cpu));
      memset(gpu, 0, sizeof(gpu));
      avm_highbd_cwp_c(cpu, pred2, n, n, plane, SRC_W, &cwp);
      /* avm_highbd_cwp_c() advances pred by width, i.e. it expects a packed
       * w x h block, so the GPU call uses stride n for pred as well. */
      if (avm_gpu_inter_cwp(ctx, inter, pred2, n, plane, SRC_W, n, n,
                            cwp.fwd_offset, cwp.bck_offset, gpu, DST_STRIDE,
                            err) != 0) {
        fprintf(stderr, "avm_gpu cwp: %s\n", err);
        return 1;
      }
      int diff = 0;
      for (int y = 0; y < n; y++) {
        for (int x = 0; x < n; x++) {
          if (cpu[y * n + x] != gpu[y * DST_STRIDE + x]) diff++;
        }
      }
      failures += diff != 0;
      printf("cwp  %2d/%2d %-3d %9s %s\n", kCwp[ci][0], kCwp[ci][1], n, "-",
             diff ? "MISMATCH" : "ok");
    }
  }
  free(pred2);

  free(plane);
  /* Single-reference subpel prediction as av2_make_inter_predictor() does it
   * for an unscaled reference: av2_highbd_convolve_2d_sr_c() with the
   * encoder's ConvolveParams (round_0 = ROUND0_BITS, round_1 =
   * 2*FILTER_BITS - round_0 for one reference). */
  {
    InterpFilterParams fp_x;
    InterpFilterParams fp_y;
    memset(&fp_x, 0, sizeof(fp_x));
    memset(&fp_y, 0, sizeof(fp_y));
    fp_x.filter_ptr = (const int16_t *)av2_sub_pel_filters_8;
    fp_x.taps = 8;
    fp_y.filter_ptr = (const int16_t *)av2_sub_pel_filters_8;
    fp_y.taps = 8;
    static const int kSrSizes[3] = { 4, 16, 64 };
    for (int bd = 10; bd <= 12; bd += 2) {
      ConvolveParams conv = get_conv_params_no_round(0, 0, NULL, DST_STRIDE, 0, bd);
      for (size_t p = 0; p < sizeof(kPos) / sizeof(kPos[0]); p++) {
        for (size_t s = 0; s < 3; s++) {
          const int n = kSrSizes[s];
          const int sx = 6, sy = 5;
          memset(cpu, 0, sizeof(cpu));
          memset(gpu, 0, sizeof(gpu));
          av2_highbd_convolve_2d_sr_c(plane + sy * SRC_W + sx, SRC_W, cpu,
                                      DST_STRIDE, n, n, &fp_x, &fp_y, kPos[p][0],
                                      kPos[p][1], &conv, bd);
          if (avm_gpu_inter_convolve_sr(
                  ctx, inter, plane, SRC_W, SRC_H, sx, sy,
                  (const int16_t *)av2_sub_pel_filters_8,
                  (const int16_t *)av2_sub_pel_filters_8, kPos[p][0],
                  kPos[p][1], n, n, bd, conv.round_0, conv.round_1, gpu,
                  DST_STRIDE, err) != 0) {
            fprintf(stderr, "avm_gpu inter sr: %s\n", err);
            return 1;
          }
          int diff = 0;
          for (int y = 0; y < n; y++) {
            for (int x = 0; x < n; x++) {
              if (cpu[y * DST_STRIDE + x] != gpu[y * DST_STRIDE + x]) {
                if (diff < 3) {
                  printf("  sr bd %d subpel(%d,%d) %dx%d (%d,%d): gpu %u != "
                         "cpu %u\n",
                         bd, kPos[p][0], kPos[p][1], n, n, x, y,
                         gpu[y * DST_STRIDE + x], cpu[y * DST_STRIDE + x]);
                }
                diff++;
              }
            }
          }
          failures += diff != 0;
          printf("sr bd%-2d (%2d,%2d) %-6d %10.3f %10.3f %s\n", bd, kPos[p][0],
                 kPos[p][1], n, 0.0, 0.0, diff ? "MISMATCH" : "ok");
        }
      }
    }
  }

  avm_gpu_inter_destroy(ctx, inter);
  avm_gpu_ctx_destroy(ctx);
  printf("\n%s\n", failures ? "FAILED" : "ALL PARITY CASES PASSED");
  return failures ? 1 : 0;
}
