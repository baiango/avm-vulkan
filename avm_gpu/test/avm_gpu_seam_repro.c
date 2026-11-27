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
 * Focused reproducer for the inter seam hazard (not wired into ctest).
 *
 * avm_gpu_hook_inter() has been observed to disagree with
 * av2_highbd_convolve_2d_{sr,scale}_c() intermittently. This runs many
 * hook-vs-C comparisons; on the first disagreement it replays that exact case
 * (same plane, same parameters) N times, which separates the two possible
 * causes:
 *   - every replay fails  -> an index/window bug (deterministic, in the math)
 *   - some replays pass   -> the result depends on stale buffer contents
 *
 * Usage: avm_gpu_seam_repro [iterations]
 * Exit: 0 = all comparisons agree, 1 = a disagreement was reproduced.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config/avm_dsp_rtcd.h"
#include "av2/common/blockd.h"
#include "av2/common/convolve.h"
#include "av2/common/filter.h"

#include "avm_gpu/vulkan_gpu.h"
#include "avm_dsp/blend.h"
#include "avm_gpu/vulkan_hook_inter.h"

#include "avm_gpu/vulkan_inter.h"

void av2_highbd_convolve_2d_sr_c(const uint16_t *src, int src_stride,
                                 uint16_t *dst, int dst_stride, int w, int h,
                                 const InterpFilterParams *filter_params_x,
                                 const InterpFilterParams *filter_params_y,
                                 const int subpel_x_qn, const int subpel_y_qn,
                                 ConvolveParams *conv_params, int bd);
void av2_highbd_cwp_convolve_2d_c(const uint16_t *src, int src_stride,
                                  uint16_t *dst, int dst_stride, int w, int h,
                                  const InterpFilterParams *filter_params_x,
                                  const InterpFilterParams *filter_params_y,
                                  const int subpel_x_qn, const int subpel_y_qn,
                                  ConvolveParams *conv_params, int bd);
void av2_highbd_cwp_convolve_x_c(const uint16_t *src, int src_stride,
                                 uint16_t *dst, int dst_stride, int w, int h,
                                 const InterpFilterParams *filter_params_x,
                                 const int subpel_x_qn,
                                 ConvolveParams *conv_params, int bd);
void av2_highbd_cwp_convolve_y_c(const uint16_t *src, int src_stride,
                                 uint16_t *dst, int dst_stride, int w, int h,
                                 const InterpFilterParams *filter_params_y,
                                 const int subpel_y_qn,
                                 ConvolveParams *conv_params, int bd);
void av2_highbd_cwp_convolve_2d_copy_c(const uint16_t *src, int src_stride,
                                       uint16_t *dst, int dst_stride, int w,
                                       int h, ConvolveParams *conv_params,
                                       int bd);
void av2_highbd_convolve_2d_scale_c(const uint16_t *src, int src_stride,
                                    uint16_t *dst, int dst_stride, int w, int h,
                                    const InterpFilterParams *filter_params_x,
                                    const InterpFilterParams *filter_params_y,
                                    const int subpel_x_qn,
                                    const int x_step_qn, const int subpel_y_qn,
                                    const int y_step_qn,
                                    ConvolveParams *conv_params, int bd);

#define SRC_W 128
#define SRC_H 128
#define DST_STRIDE 64
#define BD 10

static unsigned int rng_state = 0x9e3779b9u;
static unsigned int rnd(void) {
  rng_state = rng_state * 1664525u + 1013904223u;
  return rng_state >> 8;
}

/* The plane the failing iteration saw, kept for the replay. */
static uint16_t plane[SRC_W * SRC_H];
static uint16_t cpu_out[DST_STRIDE * DST_STRIDE];
static uint16_t gpu_out[DST_STRIDE * DST_STRIDE];

/* Compound: two passes over the CWP high-precision buffer. With scaled the
 * reference is av2_highbd_convolve_2d_scale_c(), which is what the facade runs
 * for a scaled reference, and positions/steps are in 1/1024 units. */
static int run_compound(int n, int sub_x, int sub_y, int uneven, int scaled,
                        int step) {
  const InterpFilterParams fp_x = { (const int16_t *)av2_sub_pel_filters_8, 8 };
  const InterpFilterParams fp_y = { (const int16_t *)av2_sub_pel_filters_8, 8 };
  static uint16_t c_cpu[DST_STRIDE * 64], c_gpu[DST_STRIDE * 64];
  const int sx = 16, sy = 12;
  ConvolveParams conv =
      get_conv_params_no_round(0, 0, c_cpu, DST_STRIDE, 1, BD);
  if (uneven) {
    conv.fwd_offset = 10;
    conv.bck_offset = 6;
  }
  memset(c_cpu, 0, sizeof(c_cpu));
  memset(c_gpu, 0, sizeof(c_gpu));
  memset(cpu_out, 0, sizeof(cpu_out));
  memset(gpu_out, 0, sizeof(gpu_out));
  conv.do_average = 0;
  if (scaled) {
    av2_highbd_convolve_2d_scale_c(plane + sy * SRC_W + sx, SRC_W, cpu_out,
                                   DST_STRIDE, n, n, &fp_x, &fp_y, sub_x, step,
                                   sub_y, step, &conv, BD);
  } else if (sub_x == 0 && sub_y == 0) {
    av2_highbd_cwp_convolve_2d_copy_c(plane + sy * SRC_W + sx, SRC_W, cpu_out,
                                      DST_STRIDE, n, n, &conv, BD);
  } else if (sub_y == 0) {
    av2_highbd_cwp_convolve_x_c(plane + sy * SRC_W + sx, SRC_W, cpu_out,
                                DST_STRIDE, n, n, &fp_x, sub_x, &conv, BD);
  } else if (sub_x == 0) {
    av2_highbd_cwp_convolve_y_c(plane + sy * SRC_W + sx, SRC_W, cpu_out,
                                DST_STRIDE, n, n, &fp_y, sub_y, &conv, BD);
  } else {
    av2_highbd_cwp_convolve_2d_c(plane + sy * SRC_W + sx, SRC_W, cpu_out,
                                 DST_STRIDE, n, n, &fp_x, &fp_y, sub_x, sub_y,
                                 &conv, BD);
  }
  if (getenv("SEAM_NOGPU")) { /* CPU reference only, for bisecting */
    conv.do_average = 1;
    if (scaled) {
      av2_highbd_convolve_2d_scale_c(plane + sy * SRC_W + sx, SRC_W, cpu_out,
                                     DST_STRIDE, n, n, &fp_x, &fp_y, sub_x,
                                     step, sub_y, step, &conv, BD);
    } else if (sub_x == 0 && sub_y == 0) {
      av2_highbd_cwp_convolve_2d_copy_c(plane + sy * SRC_W + sx, SRC_W, cpu_out,
                                        DST_STRIDE, n, n, &conv, BD);
    } else if (sub_y == 0) {
      av2_highbd_cwp_convolve_x_c(plane + sy * SRC_W + sx, SRC_W, cpu_out,
                                  DST_STRIDE, n, n, &fp_x, sub_x, &conv, BD);
    } else if (sub_x == 0) {
      av2_highbd_cwp_convolve_y_c(plane + sy * SRC_W + sx, SRC_W, cpu_out,
                                  DST_STRIDE, n, n, &fp_y, sub_y, &conv, BD);
    } else {
      av2_highbd_cwp_convolve_2d_c(plane + sy * SRC_W + sx, SRC_W, cpu_out,
                                   DST_STRIDE, n, n, &fp_x, &fp_y, sub_x, sub_y,
                                   &conv, BD);
    }
    return 0;
  }
  if (!avm_gpu_hook_inter(plane + sy * SRC_W + sx, SRC_W, gpu_out, DST_STRIDE,
                          n, n, BD, (const int16_t *)av2_sub_pel_filters_8,
                          (const int16_t *)av2_sub_pel_filters_8, 8, 8, sub_x,
                          sub_y, conv.round_0, conv.round_1, scaled, step,
                          step, 1, 0, c_gpu, DST_STRIDE, conv.fwd_offset,
                          conv.bck_offset)) {
    return -1;
  }
  int diff = 0;
  for (int y = 0; y < n; y++) {
    for (int x = 0; x < n; x++) {
      if (c_cpu[y * DST_STRIDE + x] != c_gpu[y * DST_STRIDE + x]) diff++;
    }
  }
  conv.do_average = 1;
  if (scaled) {
    av2_highbd_convolve_2d_scale_c(plane + sy * SRC_W + sx, SRC_W, cpu_out,
                                   DST_STRIDE, n, n, &fp_x, &fp_y, sub_x, step,
                                   sub_y, step, &conv, BD);
  } else if (sub_x == 0 && sub_y == 0) {
    av2_highbd_cwp_convolve_2d_copy_c(plane + sy * SRC_W + sx, SRC_W, cpu_out,
                                      DST_STRIDE, n, n, &conv, BD);
  } else if (sub_y == 0) {
    av2_highbd_cwp_convolve_x_c(plane + sy * SRC_W + sx, SRC_W, cpu_out,
                                DST_STRIDE, n, n, &fp_x, sub_x, &conv, BD);
  } else if (sub_x == 0) {
    av2_highbd_cwp_convolve_y_c(plane + sy * SRC_W + sx, SRC_W, cpu_out,
                                DST_STRIDE, n, n, &fp_y, sub_y, &conv, BD);
  } else {
    av2_highbd_cwp_convolve_2d_c(plane + sy * SRC_W + sx, SRC_W, cpu_out,
                                 DST_STRIDE, n, n, &fp_x, &fp_y, sub_x, sub_y,
                                 &conv, BD);
  }
  if (!avm_gpu_hook_inter(plane + sy * SRC_W + sx, SRC_W, gpu_out, DST_STRIDE,
                          n, n, BD, (const int16_t *)av2_sub_pel_filters_8,
                          (const int16_t *)av2_sub_pel_filters_8, 8, 8, sub_x,
                          sub_y, conv.round_0, conv.round_1, scaled, step,
                          step, 1, 1, c_gpu, DST_STRIDE, conv.fwd_offset,
                          conv.bck_offset)) {
    return -1;
  }
  for (int y = 0; y < n; y++) {
    for (int x = 0; x < n; x++) {
      if (cpu_out[y * DST_STRIDE + x] != gpu_out[y * DST_STRIDE + x]) diff++;
    }
  }
  if (diff) printf("  compound n=%d subpel=(%d,%d) uneven=%d: %d samples\n",
                   n, sub_x, sub_y, uneven, diff);
  return diff;
}

/* Masked blend (inter-intra / wedge masks): the seam against
 * avm_highbd_blend_a64_mask(). */
static int run_blend(int n, int subw, int subh, int mask_stride) {
  static uint16_t l0[2 * SRC_W * SRC_H], l1[2 * SRC_W * SRC_H];
  static uint16_t d_cpu[2 * SRC_W * SRC_H], d_gpu[2 * SRC_W * SRC_H];
  static uint8_t mask[4 * SRC_W * SRC_H];
  const int stride = SRC_W;
  for (int i = 0; i < SRC_W * SRC_H; i++) {
    l0[i] = (uint16_t)(rnd() & 0x3ff);
    l1[i] = (uint16_t)(rnd() & 0x3ff);
    d_cpu[i] = 0;
    d_gpu[i] = 0;
  }
  /* A subsampled mask carries 2x2 samples per output sample, hence the
   * mask_stride the caller passes. */
  for (int i = 0; i < 4 * SRC_W * SRC_H; i++) mask[i] = (uint8_t)(rnd() & 63);
  avm_highbd_blend_a64_mask_c(d_cpu, stride, l0, stride, l1, stride, mask,
                            mask_stride, n, n, subw, subh, BD);
  if (!avm_gpu_hook_blend_a64(d_gpu, stride, l0, stride, l1, stride, mask,
                              mask_stride, n, n, subw, subh, BD)) {
    return -1;
  }
  int diff = 0;
  for (int y = 0; y < n; y++) {
    for (int x = 0; x < n; x++) {
      if (d_cpu[y * stride + x] != d_gpu[y * stride + x]) diff++;
    }
  }
  if (diff) {
    printf("  blend n=%d subw=%d subh=%d: %d samples\n", n, subw, subh, diff);
  }
  return diff;
}

static int run_case(int scaled, int n, int sub_x, int sub_y, int step) {
  const InterpFilterParams fp_x = { (const int16_t *)av2_sub_pel_filters_8, 8 };
  const InterpFilterParams fp_y = { (const int16_t *)av2_sub_pel_filters_8, 8 };
  ConvolveParams conv;
  get_conv_params_no_round(0, 0, NULL, DST_STRIDE, 0, BD);
  ConvolveParams c = get_conv_params_no_round(0, 0, NULL, DST_STRIDE, 0, BD);
  conv = c;
  const int sx = 16, sy = 12;
  memset(cpu_out, 0, sizeof(cpu_out));
  memset(gpu_out, 0, sizeof(gpu_out));
  if (scaled) {
    av2_highbd_convolve_2d_scale_c(plane + sy * SRC_W + sx, SRC_W, cpu_out,
                                   DST_STRIDE, n, n, &fp_x, &fp_y, sub_x, step,
                                   sub_y, step, &c, BD);
  } else {
    av2_highbd_convolve_2d_sr_c(plane + sy * SRC_W + sx, SRC_W, cpu_out,
                                DST_STRIDE, n, n, &fp_x, &fp_y, sub_x, sub_y,
                                &c, BD);
  }
  if (!avm_gpu_hook_inter(plane + sy * SRC_W + sx, SRC_W, gpu_out, DST_STRIDE,
                          n, n, BD, (const int16_t *)av2_sub_pel_filters_8,
                          (const int16_t *)av2_sub_pel_filters_8, 8, 8, sub_x,
                          sub_y, c.round_0, c.round_1, scaled, step, step, 0, 0,
                          NULL, 0, 0, 0)) {
    return -1; /* declined */
  }
  int diff = 0, first_x = -1, first_y = 0;
  for (int y = 0; y < n; y++) {
    for (int x = 0; x < n; x++) {
      if (cpu_out[y * DST_STRIDE + x] != gpu_out[y * DST_STRIDE + x]) {
        if (first_x < 0) {
          first_x = x;
          first_y = y;
        }
        diff++;
      }
    }
  }
  if (diff) {
    printf("  first diff at (%d,%d): cpu %u gpu %u, %d of %d pixels\n", first_x,
           first_y, cpu_out[first_y * DST_STRIDE + first_x],
           gpu_out[first_y * DST_STRIDE + first_x], diff, n * n);
  }
  return diff;
}

int main(int argc, char **argv) {
  const int iters = argc > 1 ? atoi(argv[1]) : 4000;
  char err[AVM_GPU_ERR_LEN];
  setenv("AV2_GPU_INTER", "1", 1);
  avm_gpu_ctx *ctx = NULL;
  if (avm_gpu_ctx_create(&ctx, err) != 0) {
    printf("SKIP: no Vulkan device\n");
    return 77;
  }
  avm_gpu_inter *inter = NULL;
  if (avm_gpu_inter_create(ctx, &inter, err) != 0) {
    printf("avm_gpu inter: %s\n", err);
    return 1;
  }

  int declined = 0;
  int first_declined_it = -1;
  const int verbose = getenv("SEAM_VERBOSE") != NULL;
  for (int it = 0; it < iters; it++) {
    const int mode = it % 4;   /* 0 unscaled, 1 scaled, 2 compound */
    const int scaled = mode == 1;
    const int n = 4 << (it % 5); /* 4, 8, 16, 32, 64 */
    const int step = scaled ? 910 : 16;
    const int sub_x = scaled ? (int)(rnd() % 1024) : (int)(rnd() % 16);
    const int sub_y = scaled ? (int)(rnd() % 1024) : (int)(rnd() % 16);
    for (int i = 0; i < SRC_W * SRC_H; i++) plane[i] = (uint16_t)(rnd() & 0x3ff);

    /* The seam takes compound only for the CWP y-only and 2D shapes; the copy
     * and x-only paths are kernel-gated but not taken in the encoder context,
     * so a compound case with a zero y phase is expected to be declined. */
    const int cwp_y = (sub_y % 15) + 1;   /* 1..15: the seam needs a y phase */
    const int cwp_x = sub_x % 16;
    const int expect_decline = mode == 2 && cwp_y == 0; /* never: y >= 1 */
    const int rc = mode == 3
                       ? run_blend(8 << (it % 4), it % 2, (it >> 1) % 2,
                                   ((it % 2) ? 2 : 1) * (8 << (it % 4)))
                       : mode == 2
                       /* Steps and phases go together: 1/1024 with a scaled
                        * reference, 1/16 otherwise. */
                       ? run_compound(n, (it % 6) >= 3 ? sub_x % 1024 : cwp_x,
                                      (it % 6) >= 3 ? sub_y % 1024 : cwp_y,
                                      it % 2, (it % 6) >= 3,
                                      (it % 6) >= 3 ? 910 : 16)
                       : run_case(scaled, n, sub_x, sub_y, step);
    if (expect_decline && rc < 0) continue;
    if (rc < 0) {
      /* A decline is not a pass: it would let a window that is too small look
       * like agreement, which is exactly how an earlier revision hid. */
      if (first_declined_it < 0) first_declined_it = it;
      declined++;
      continue;
    }
    if (rc > 0) {
      printf("FIRST FAILURE: it=%d scaled=%d n=%d subpel=(%d,%d) step=%d\n", it,
             scaled, n, sub_x, sub_y, step);
      printf("replaying the same case 50 times:\n");
      int fails = 0;
      for (int r = 0; r < 50; r++) {
        if (run_case(scaled, n, sub_x, sub_y, step) != 0) fails++;
      }
      printf("  %d of 50 replays failed -> %s\n", fails,
             fails == 50 ? "DETERMINISTIC (an index/window bug)"
                         : "INTERMITTENT (stale buffer contents)");
      avm_gpu_inter_destroy(ctx, inter);
      avm_gpu_ctx_destroy(ctx);
      return 1;
    }
  }
  if (declined) {
    printf("FAIL: the seam declined %d of %d cases (first at it=%d)\n", declined,
           iters, first_declined_it);
    avm_gpu_inter_destroy(ctx, inter);
    avm_gpu_ctx_destroy(ctx);
    return 1;
  }
  printf("ok: %d comparisons agree, 0 declined\n", iters);
  avm_gpu_inter_destroy(ctx, inter);
  avm_gpu_ctx_destroy(ctx);
  return 0;
}