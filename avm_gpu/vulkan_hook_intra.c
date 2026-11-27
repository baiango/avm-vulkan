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

#include "avm_gpu/vulkan_hook_intra.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "av2/common/enums.h"
#include "avm_gpu/vulkan_gpu.h"
#include "avm_gpu/vulkan_intra.h"

typedef struct intra_hook_state {
  int state; /* 0 = uninitialised, 1 = off, 2 = ready */
  avm_gpu_ctx *ctx;
  avm_gpu_intra *intra;
  unsigned long blocks;
} intra_hook_state;

static intra_hook_state g_hook;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static void print_stats(void) {
  if (getenv("AV2_GPU_INTRA_STATS")) {
    fprintf(stderr, "avm_gpu: %lu intra blocks on the GPU\n", g_hook.blocks);
  }
}

/* Caller holds g_lock. */
static int hook_ready_locked(void) {
  if (g_hook.state) return g_hook.state == 2;

  const char *env = getenv("AV2_GPU_INTRA");
  if (!env || !*env || !strcmp(env, "0")) {
    g_hook.state = 1;
    return 0;
  }
  char err[AVM_GPU_ERR_LEN];
  if (avm_gpu_ctx_create(&g_hook.ctx, err) != 0 ||
      avm_gpu_intra_create(g_hook.ctx, &g_hook.intra, err) != 0) {
    fprintf(stderr, "avm_gpu: intra hook disabled (%s)\n", err);
    avm_gpu_ctx_destroy(g_hook.ctx);
    g_hook.ctx = NULL;
    g_hook.state = 1;
    return 0;
  }
  atexit(print_stats);
  fprintf(stderr, "avm_gpu: intra hook enabled on %s\n",
          avm_gpu_device_name(g_hook.ctx));
  g_hook.state = 2;
  return 1;
}

/* The modes dc_pred_high[][] / pred_high[][tx_size] select, mapped onto the
 * kernel's mode numbering. PAETH and SMOOTH* read the top-left sample and the
 * far edges (above[bw], left[bh]), which the builder has already filled. */
static int kernel_mode(int mode, int n_top_px, int n_left_px, int w, int h) {
  switch (mode) {
    /* A rectangular block's DC uses AVM's reciprocal-division form
     * (highbd_dc_predictor_rect -> resolve_divisor_32), which rounds
     * differently from the square (sum + count/2) / count; those stay on the
     * CPU until the kernel models it. */
    case DC_PRED:
      if (w != h) return -1;
      return avm_gpu_intra_auto_mode(n_top_px > 0, n_left_px > 0);
    case SMOOTH_PRED: return AVM_GPU_INTRA_SMOOTH;
    case SMOOTH_V_PRED: return AVM_GPU_INTRA_SMOOTH_V;
    case SMOOTH_H_PRED: return AVM_GPU_INTRA_SMOOTH_H;
    case PAETH_PRED: return AVM_GPU_INTRA_PAETH;
    default: return -1;
  }
}

int avm_gpu_hook_intra(int mode, uint16_t *dst, int dst_stride,
                       const uint16_t *above, const uint16_t *left, int w,
                       int h, int bd, int n_top_px, int n_left_px) {
  if (!dst || !above || !left || w <= 0 || h <= 0 || dst_stride < w) return 0;
  const int kmode = kernel_mode(mode, n_top_px, n_left_px, w, h);
  if (kmode < 0) return 0;

  pthread_mutex_lock(&g_lock);
  int handled = 0;
  if (hook_ready_locked()) {
    char err[AVM_GPU_ERR_LEN];
    const int ok =
        kmode == AVM_GPU_INTRA_DC || kmode == AVM_GPU_INTRA_DC_LEFT ||
                kmode == AVM_GPU_INTRA_DC_TOP || kmode == AVM_GPU_INTRA_DC_128
            ? avm_gpu_intra_predict(g_hook.ctx, g_hook.intra, above, left, w, h,
                                    kmode, bd, dst, dst_stride, err)
            : avm_gpu_intra_predict_tl(g_hook.ctx, g_hook.intra, above, left,
                                       above[-1], w, h, kmode, bd, dst,
                                       dst_stride, err);
    if (ok == 0) {
      g_hook.blocks++;
      handled = 1;
    } else if (getenv("AV2_GPU_INTRA_DEBUG")) {
      fprintf(stderr, "avm_gpu: intra fallback %dx%d mode %d: %s\n", w, h, kmode,
              err);
    }
  }
  pthread_mutex_unlock(&g_lock);
  return handled;
}
