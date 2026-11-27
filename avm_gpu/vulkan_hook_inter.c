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

#include "avm_gpu/vulkan_hook_inter.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "avm_dsp/avm_filter.h"
#include "av2/common/convolve.h"
#include "av2/common/mv.h"
#include "av2/common/warped_motion.h"
#include "avm_gpu/vulkan_gpu.h"
#include "avm_gpu/vulkan_inter.h"
#include "avm_gpu/vulkan_txfm.h"
#include "avm_gpu/vulkan_warp.h"

#define HOOK_SUBPEL_TAPS 8
#define HOOK_BORDER 3
#define HOOK_MAX_DIM 64

typedef struct inter_hook_state {
  int state; /* 0 = uninitialised, 1 = off, 2 = ready */
  avm_gpu_ctx *ctx;
  avm_gpu_inter *inter;
  avm_gpu_warp *warp;
  avm_gpu_txfm *txfm;
  unsigned long blocks;
  unsigned long compound;
  unsigned long blends;
  unsigned long warped;
  unsigned long fwd_txfm;
  unsigned long d16_blends;
  unsigned long warped_scaled;
  unsigned long compound_scaled;
  unsigned long warped_compound;
  unsigned long warped_cpu;
} inter_hook_state;

static inter_hook_state g_hook;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static void print_stats(void) {
  if (getenv("AV2_GPU_INTER_STATS")) {
    fprintf(stderr,
            "avm_gpu: %lu inter prediction blocks on the GPU (%lu compound, "
            "%lu masked blends, %lu warped, %lu forward transforms, %lu "
            "masked compound, %lu scaled warped, %lu compound warped, %lu "
            "scaled compound, %lu warped C)\n",
            g_hook.blocks, g_hook.compound, g_hook.blends, g_hook.warped,
            g_hook.fwd_txfm, g_hook.d16_blends, g_hook.warped_scaled,
            g_hook.warped_compound, g_hook.compound_scaled, g_hook.warped_cpu);
  }
}

/* Caller holds g_lock. */
static int hook_ready_locked(void) {
  if (g_hook.state) return g_hook.state == 2;

  const char *env = getenv("AV2_GPU_INTER");
  if (!env || !*env || !strcmp(env, "0")) {
    g_hook.state = 1;
    return 0;
  }
  char err[AVM_GPU_ERR_LEN];
  const char *txf = getenv("AV2_GPU_FWD_TXFM");
  const int want_txfm = txf && *txf && strcmp(txf, "0") != 0;
  (void)want_txfm;
  if (avm_gpu_ctx_create(&g_hook.ctx, err) != 0 ||
      avm_gpu_inter_create(g_hook.ctx, &g_hook.inter, err) != 0 ||
      avm_gpu_warp_create(g_hook.ctx, &g_hook.warp, err) != 0 ||
      avm_gpu_txfm_create(g_hook.ctx, &g_hook.txfm, err) != 0) {
    fprintf(stderr, "avm_gpu: inter hook disabled (%s)\n", err);
    avm_gpu_ctx_destroy(g_hook.ctx);
    g_hook.ctx = NULL;
    g_hook.state = 1;
    return 0;
  }
  atexit(print_stats);
  fprintf(stderr, "avm_gpu: inter hook enabled on %s\n",
          avm_gpu_device_name(g_hook.ctx));
  g_hook.state = 2;
  return 1;
}

int avm_gpu_hook_inter(const uint16_t *src, int src_stride, uint16_t *dst,
                       int dst_stride, int w, int h, int bd,
                       const int16_t *filter_x, const int16_t *filter_y,
                       int taps_x, int taps_y, int subpel_x_qn,
                       int subpel_y_qn, int round_0, int round_1, int scaled,
                       int x_step_qn, int y_step_qn, int is_compound,
                       int do_average, uint16_t *conv_dst, int conv_stride,
                       int fwd_offset, int bck_offset) {
  if (!src || !dst || !filter_x || !filter_y) return 0;
  if (w <= 0 || h <= 0 || w > HOOK_MAX_DIM || h > HOOK_MAX_DIM) return 0;
  if (taps_x != HOOK_SUBPEL_TAPS || taps_y != HOOK_SUBPEL_TAPS) return 0;
  const int subpel_max = scaled ? SCALE_SUBPEL_MASK : 15;
  if (subpel_x_qn < 0 || subpel_x_qn > subpel_max || subpel_y_qn < 0 ||
      subpel_y_qn > subpel_max) {
    return 0;
  }
  if (scaled && (x_step_qn <= 0 || y_step_qn <= 0 ||
                 x_step_qn > SCALE_SUBPEL_SHIFTS ||
                 y_step_qn > SCALE_SUBPEL_SHIFTS)) {
    return 0;
  }
  if (src_stride < w + 2 * HOOK_BORDER || dst_stride < w) return 0;

  /* Compound (two passes over the CWP high-precision buffer) is taken
   * unscaled; the kernel has a path for each filter shape, matching the
   * function the C dispatches to (2D, x only, y only, copy). */
  if (is_compound && !conv_dst) return 0;
  /* A scaled reference's compound prediction is not the CWP family: the C
   * facade sends every scaled call, compound or not, through
   * av2_highbd_convolve_2d_scale(), so it takes the sr kernel's compound
   * output stage (conv buffer + average), not the CWP kernel. */
  /* An unscaled call must carry 1/16 steps: anything else would make the window
   * arithmetic below (which divides by the subpel bit count) read far outside
   * the caller's plane. A scaled call's steps are bounded by 1 << 10. */
  if (scaled) {
    if (x_step_qn <= 0 || y_step_qn <= 0 || x_step_qn > SCALE_SUBPEL_SHIFTS ||
        y_step_qn > SCALE_SUBPEL_SHIFTS) {
      return 0;
    }
  } else if (x_step_qn != SUBPEL_SHIFTS || y_step_qn != SUBPEL_SHIFTS) {
    return 0;
  }
  /* Only the shapes the encoder reproduces bit-exactly are taken. The copy and
   * x-only CWP paths are kernel-gated against the C (avm_gpu_inter_test) but
   * change the encoder bitstream when the seam takes them, so they stay on the
   * CPU until that difference is understood; y-only and 2D are verified. */
  /* All four CWP shapes are taken: the facade's dispatch (copy / x-only /
   * y-only / 2D) is exactly the mode the kernel selects from the phases. */


  pthread_mutex_lock(&g_lock);
  int handled = 0;
  if (hook_ready_locked()) {
    char err[AVM_GPU_ERR_LEN];
    /* The kernels index the uploaded plane as (lead + base + tap - 3), so the
     * upload starts at the window (block origin minus the three-sample border)
     * and has to reach the last sample any tap touches: lead + last_base + 4,
     * i.e. last_base + 2 * lead + 2 rows. Being a row short reads whatever the
     * buffer held before, which only shows up once the previous contents differ
     * - avm_gpu_seam_repro exists to catch exactly that. */
    const uint16_t *window = src - HOOK_BORDER * src_stride - HOOK_BORDER;
    const int subpel_bits = scaled ? SCALE_SUBPEL_BITS : 4;
    const int last_by = (((h - 1) * y_step_qn) + subpel_y_qn) >> subpel_bits;
    /* Every kernel here reads rows base_y - 3 .. base_y + 4 (the 8-tap
     * vertical filter over the three border rows above), so the window must
     * reach last_base + 4: rows = last_base + 4 + HOOK_BORDER + 1 =
     * last_base + 8. With last_base + 7 the last row of the block was read
     * from whatever the buffer held before - invisible to avm_gpu_seam_repro,
     * which uploads the over-read row from the same fresh plane on every call,
     * but it changed the bottom row of blocks in a resized-reference stream,
     * where 4,144 blocks differed only in their last row. */
    const int rows = last_by + 2 * HOOK_BORDER + 2;
    /* Never hand the kernels a window larger than the block plus its border:
     * h + 8 covers every tap reach for both the unscaled and the scaled
     * mapping. */
    if (rows > h + 2 * HOOK_BORDER + 3) return 0;
    int rc;
    if (is_compound && scaled) {
      const int uneven =
          fwd_offset != (1 << (4 - 1)) || bck_offset != (1 << (4 - 1));
      rc = avm_gpu_inter_convolve_scaled_compound(
          g_hook.ctx, g_hook.inter, window, src_stride, rows, HOOK_BORDER,
          HOOK_BORDER, filter_x, filter_y, subpel_x_qn, x_step_qn, subpel_y_qn,
          y_step_qn, w, h, bd, round_0, round_1, conv_dst, conv_stride,
          do_average, uneven, fwd_offset, bck_offset, dst, dst_stride, err);
    } else if (is_compound) {
      const int uneven =
          fwd_offset != (1 << (4 - 1)) || bck_offset != (1 << (4 - 1));
      rc = avm_gpu_inter_cwp_convolve(
          g_hook.ctx, g_hook.inter, window, src_stride, rows, HOOK_BORDER,
          HOOK_BORDER, filter_x, filter_y, subpel_x_qn, subpel_y_qn, w, h, bd,
          round_0, round_1, conv_dst, conv_stride, do_average, do_average,
          uneven, fwd_offset, bck_offset, scaled, x_step_qn, y_step_qn, dst,
          dst_stride, err);
    } else {
    rc =
        scaled ? avm_gpu_inter_convolve_scaled_sr(
                     g_hook.ctx, g_hook.inter, window, src_stride, rows,
                     HOOK_BORDER, HOOK_BORDER, filter_x, filter_y, subpel_x_qn,
                     x_step_qn, subpel_y_qn, y_step_qn, w, h, bd, round_0,
                     round_1, dst, dst_stride, err)
               : avm_gpu_inter_convolve_sr(g_hook.ctx, g_hook.inter, window,
                                           src_stride, rows, HOOK_BORDER,
                                           HOOK_BORDER, filter_x, filter_y,
                                           subpel_x_qn, subpel_y_qn, w, h, bd,
                                           round_0, round_1, dst, dst_stride,
                                           err);
    }
    if (rc != 0 && getenv("AV2_GPU_INTER_DEBUG")) {
      fprintf(stderr, "avm_gpu: inter fallback %dx%d sub(%d,%d) rows=%d: %s\n", w,
              h, subpel_x_qn, subpel_y_qn, rows, err);
    }
    if (rc == 0) {
      g_hook.blocks++;
      if (is_compound) {
        g_hook.compound++;
        if (scaled) g_hook.compound_scaled++;
      }
      handled = 1;
    }
  }
  pthread_mutex_unlock(&g_lock);
  return handled;
}

int avm_gpu_hook_blend_a64(uint16_t *dst, int dst_stride,
                           const uint16_t *src0, int s0_stride,
                           const uint16_t *src1, int s1_stride,
                           const uint8_t *mask, int mask_stride, int w, int h,
                           int subw, int subh, int bd) {
  if (!dst || !src0 || !src1 || !mask) return 0;
  if (w <= 0 || h <= 0 || w > HOOK_MAX_DIM * 2 || h > HOOK_MAX_DIM * 2) return 0;

  pthread_mutex_lock(&g_lock);
  int handled = 0;
  if (hook_ready_locked()) {
    char err[AVM_GPU_ERR_LEN];
    const int brc =
        avm_gpu_inter_blend_a64(g_hook.ctx, g_hook.inter, src0, s0_stride, src1,
                                s1_stride, mask, mask_stride, w, h, subw, subh,
                                bd, dst, dst_stride, err);
    if (brc != 0 && getenv("AV2_GPU_INTER_DEBUG")) {
      fprintf(stderr, "avm_gpu: blend fallback %dx%d subw=%d subh=%d: %s\n", w,
              h, subw, subh, err);
    }
    if (brc == 0) {
      g_hook.blends++;
      handled = 1;
    }
  }
  pthread_mutex_unlock(&g_lock);
  return handled;
}

int avm_gpu_hook_blend_a64_d16(uint16_t *dst, int dst_stride,
                               const uint16_t *src0, int s0_stride,
                               const uint16_t *src1, int s1_stride,
                               const uint8_t *mask, int mask_stride, int w,
                               int h, int subw, int subh, int bd,
                               int round_offset, int round_bits) {
  if (!dst || !src0 || !src1 || !mask) return 0;
  if (w <= 0 || h <= 0 || w > HOOK_MAX_DIM * 2 || h > HOOK_MAX_DIM * 2) return 0;

  pthread_mutex_lock(&g_lock);
  int handled = 0;
  if (hook_ready_locked()) {
    char err[AVM_GPU_ERR_LEN];
    const int brc = avm_gpu_inter_blend_a64_d16(
        g_hook.ctx, g_hook.inter, src0, s0_stride, src1, s1_stride, mask,
        mask_stride, w, h, subw, subh, bd, round_offset, round_bits, dst,
        dst_stride, err);
    if (brc != 0 && getenv("AV2_GPU_INTER_DEBUG")) {
      fprintf(stderr, "avm_gpu: d16 blend fallback %dx%d subw=%d subh=%d: %s\n",
              w, h, subw, subh, err);
    }
    if (brc == 0) {
      g_hook.d16_blends++;
      handled = 1;
    }
  }
  pthread_mutex_unlock(&g_lock);
  return handled;
}

static int warp_try(const int32_t wmmat[8], int wmtype, int alpha, int beta,
                    int gamma, int delta, int use_affine_filter, int bd,
                    const uint16_t *ref, int width, int height, int stride,
                    uint16_t *pred, int p_col, int p_row, int p_width,
                    int p_height, int p_stride, int ss_x, int ss_y, int round_0,
                    int round_1, const struct scale_factors *sf,
                    int use_warp_bd_box, const PadBlock *warp_bd_box,
                    int is_compound, int do_average, uint16_t *conv_dst,
                    int conv_stride, int fwd_offset, int bck_offset) {
  if (!wmmat || !ref || !pred || p_width <= 0 || p_height <= 0) return 0;
  /* A compound call passes the high-precision buffer and its averaging state
   * to the kernels; a single reference passes NULL. */
  if (is_compound && !conv_dst) return 0;
  const int scaled = sf != NULL && av2_is_scaled(sf);
  const int use_affine =
      !scaled && use_affine_filter && p_width >= 8 && p_height >= 8;
  if (use_warp_bd_box && !warp_bd_box) return 0;
  if (p_width > 128 || p_height > 128) return 0;

  int32_t mat[8];
  memcpy(mat, wmmat, sizeof(mat));
  if (wmtype == ROTZOOM) {  /* av2_highbd_warp_plane() expands ROTZOOM */
    mat[5] = mat[2];
    mat[4] = -mat[3];
  }

  avm_gpu_warp_compound cmp;
  memset(&cmp, 0, sizeof(cmp));
  cmp.conv = conv_dst;
  cmp.conv_stride = conv_stride;
  cmp.do_average = do_average;
  cmp.uneven = fwd_offset != (1 << (4 - 1)) || bck_offset != (1 << (4 - 1));
  cmp.fwd = fwd_offset;
  cmp.bck = bck_offset;
  const avm_gpu_warp_compound *cmp_ptr = is_compound ? &cmp : NULL;

  pthread_mutex_lock(&g_lock);
  int handled = 0;
  if (hook_ready_locked()) {
    char err[AVM_GPU_ERR_LEN];
    const int rc =
        scaled
            ? avm_gpu_warp_affine_scaled(
                  g_hook.ctx, g_hook.warp, mat, ref, width, height, stride,
                  pred, p_col, p_row, p_width, p_height, p_stride, bd, round_0,
                  round_1, ss_x, ss_y, sf->x_scale_fp, sf->y_scale_fp, sf->x_step_q4,
                  sf->y_step_q4, (const int32_t *)warp_bd_box,
                  &av2_ext_warped_filter[0][0], cmp_ptr, err)
        : use_affine
            ? avm_gpu_warp_affine(g_hook.ctx, g_hook.warp, mat, ref, width,
                                  height, stride, pred, p_col, p_row, p_width,
                                  p_height, p_stride, bd, round_0, round_1,
                                  alpha, beta, gamma, delta, ss_x, ss_y,
                                  &av2_warped_filter[0][0], cmp_ptr, err)
            : avm_gpu_warp_affine_ext(g_hook.ctx, g_hook.warp, mat, ref, width,
                                      height, stride, pred, p_col, p_row,
                                      p_width, p_height, p_stride, bd, round_0,
                                      round_1, ss_x, ss_y,
                                      &av2_ext_warped_filter[0][0],
                                      (const int32_t *)warp_bd_box,
                                      use_warp_bd_box, cmp_ptr, err);
    if (rc == 0) {
      g_hook.warped++;
      if (scaled) g_hook.warped_scaled++;
      if (is_compound) g_hook.warped_compound++;
      handled = 1;
    } else if (getenv("AV2_GPU_INTER_DEBUG")) {
      fprintf(stderr, "avm_gpu: warp fallback %dx%d: %s\n", p_width, p_height,
              err);
    }
  }
  pthread_mutex_unlock(&g_lock);
  return handled;
}

int avm_gpu_hook_warp(const int32_t wmmat[8], int wmtype, int alpha, int beta,
                      int gamma, int delta, int use_affine_filter, int bd,
                      const uint16_t *ref, int width, int height, int stride,
                      uint16_t *pred, int p_col, int p_row, int p_width,
                      int p_height, int p_stride, int ss_x, int ss_y,
                      int round_0, int round_1, const struct scale_factors *sf,
                      int use_warp_bd_box, const PadBlock *warp_bd_box,
                      int is_compound, int do_average, uint16_t *conv_dst,
                      int conv_stride, int fwd_offset, int bck_offset) {
  const int handled = warp_try(wmmat, wmtype, alpha, beta, gamma, delta,
                               use_affine_filter, bd, ref, width, height,
                               stride, pred, p_col, p_row, p_width, p_height,
                               p_stride, ss_x, ss_y, round_0, round_1, sf,
                               use_warp_bd_box, warp_bd_box, is_compound,
                               do_average, conv_dst, conv_stride, fwd_offset,
                               bck_offset);
  if (!handled) {
    pthread_mutex_lock(&g_lock);
    if (hook_ready_locked()) g_hook.warped_cpu++;
    pthread_mutex_unlock(&g_lock);
  }
  return handled;
}

int avm_gpu_hook_fwd_txfm(const int16_t *resi, int32_t *coeff, int diff_stride,
                          int tx_w, int tx_h, int shift1, int shift2,
                          const int *mat_w, const int *mat_h) {
  if (!resi || !coeff || !mat_w || !mat_h) return 0;
  if (tx_w < 4 || tx_h < 4 || tx_w > 64 || tx_h > 64 || (tx_w & 3) ||
      (tx_h & 3)) {
    return 0;
  }

  pthread_mutex_lock(&g_lock);
  int handled = 0;
  if (hook_ready_locked()) {
    char err[AVM_GPU_ERR_LEN];
    const avm_gpu_txfm_pos pos = { 0, 0 };
    if (avm_gpu_txfm_fwd(g_hook.ctx, g_hook.txfm, resi, diff_stride, &pos, 1,
                         tx_w, tx_h, shift1, shift2, mat_w, mat_h, coeff,
                         err) == 0) {
      g_hook.fwd_txfm++;
      handled = 1;
    }
  }
  pthread_mutex_unlock(&g_lock);
  return handled;
}
