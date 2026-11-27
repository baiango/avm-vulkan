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
#include "avm_gpu/vulkan_gpu.h"
#include "avm_gpu/vulkan_inter.h"

#define HOOK_SUBPEL_TAPS 8
#define HOOK_BORDER 3
#define HOOK_MAX_DIM 64

typedef struct inter_hook_state {
  int state; /* 0 = uninitialised, 1 = off, 2 = ready */
  avm_gpu_ctx *ctx;
  avm_gpu_inter *inter;
  unsigned long blocks;
} inter_hook_state;

static inter_hook_state g_hook;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static void print_stats(void) {
  if (getenv("AV2_GPU_INTER_STATS")) {
    fprintf(stderr, "avm_gpu: %lu inter prediction blocks on the GPU\n",
            g_hook.blocks);
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
  if (avm_gpu_ctx_create(&g_hook.ctx, err) != 0 ||
      avm_gpu_inter_create(g_hook.ctx, &g_hook.inter, err) != 0) {
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
                       int x_step_qn, int y_step_qn) {
  if (!src || !dst || !filter_x || !filter_y) return 0;
  if (w <= 0 || h <= 0 || w > HOOK_MAX_DIM || h > HOOK_MAX_DIM) return 0;
  if (taps_x != HOOK_SUBPEL_TAPS || taps_y != HOOK_SUBPEL_TAPS) return 0;
  const int subpel_max = scaled ? SCALE_SUBPEL_MASK : 15;
  if (subpel_x_qn < 0 || subpel_x_qn > subpel_max || subpel_y_qn < 0 ||
      subpel_y_qn > subpel_max) {
    return 0;
  }
  if (scaled && (x_step_qn <= 0 || y_step_qn <= 0 || x_step_qn > 1024 ||
                 y_step_qn > 1024)) {
    return 0;
  }
  if (src_stride < w + 2 * HOOK_BORDER || dst_stride < w) return 0;

  pthread_mutex_lock(&g_lock);
  int handled = 0;
  if (hook_ready_locked()) {
    char err[AVM_GPU_ERR_LEN];
    /* Hand the kernel the window the C reads: the block plus its three-sample
     * border, addressed as a small plane with the block at (3, 3). A scaled
     * reference reaches further down/right, by the scale factor. */
    const uint16_t *window = src - HOOK_BORDER * src_stride - HOOK_BORDER;
    const int rows =
        scaled ? HOOK_BORDER + (((h - 1) * y_step_qn + subpel_y_qn) >>
                                SCALE_SUBPEL_BITS) +
                     5
               : h + 2 * HOOK_BORDER;
    const int rc =
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
    if (rc == 0) {
      g_hook.blocks++;
      handled = 1;
    }
  }
  pthread_mutex_unlock(&g_lock);
  return handled;
}
