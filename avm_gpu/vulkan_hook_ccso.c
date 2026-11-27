/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at aomedia.org/license/software-license/bsd-3-c/.  If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * aomedia.org/license/patent-license/.
 */

#include "avm_gpu/vulkan_hook_ccso.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "avm_gpu/vulkan_ccso.h"
#include "avm_gpu/vulkan_gpu.h"

typedef struct ccso_hook_state {
  int state; /* 0 = uninitialised, 1 = off, 2 = ready */
  avm_gpu_ctx *ctx;
  avm_gpu_ccso *ccso;
  unsigned long blocks;
  unsigned long edge;
} ccso_hook_state;

static ccso_hook_state g_hook;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static void print_stats(void) {
  if (getenv("AV2_GPU_CCSO_STATS")) {
    fprintf(stderr,
            "avm_gpu: %lu CCSO blocks on the GPU (%lu declined)\n",
            g_hook.blocks, g_hook.edge);
  }
}

/* Caller holds g_lock. */
static int hook_ready_locked(void) {
  if (g_hook.state) return g_hook.state == 2;

  const char *env = getenv("AV2_GPU_CCSO");
  if (!env || !*env || !strcmp(env, "0")) {
    g_hook.state = 1;
    return 0;
  }
  char err[AVM_GPU_ERR_LEN];
  if (avm_gpu_ctx_create(&g_hook.ctx, err) != 0 ||
      avm_gpu_ccso_create(g_hook.ctx, &g_hook.ccso, err) != 0) {
    fprintf(stderr, "avm_gpu: CCSO hook disabled (%s)\n", err);
    avm_gpu_ctx_destroy(g_hook.ctx);
    g_hook.ctx = NULL;
    g_hook.state = 1;
    return 0;
  }
  atexit(print_stats);
  fprintf(stderr, "avm_gpu: CCSO hook enabled on %s\n",
          avm_gpu_device_name(g_hook.ctx));
  g_hook.state = 2;
  return 1;
}

int avm_gpu_hook_ccso(const uint16_t *src_y, uint16_t *dst_yuv, int x, int y,
                      int pic_width, int pic_height, const int8_t *offset_buf,
                      int src_stride, int dst_stride, int y_uv_hscale,
                      int y_uv_vscale, int thr, int neg_thr, int src_loc_0,
                      int src_loc_1, int max_val, int blk_w, int blk_h,
                      int single_band, int shift_bits, int edge_clf,
                      int bo_only) {
  if (!src_y || !dst_yuv || !offset_buf) return 0;

  pthread_mutex_lock(&g_lock);
  int handled = 0;
  if (hook_ready_locked() && (x < 2 || y < 1)) {
    /* The window needs one row above and two columns beside the block, which
     * the padded reference has but the destination plane does not. */
    g_hook.edge++;
  } else if (hook_ready_locked()) {
    char err[AVM_GPU_ERR_LEN];
    if (avm_gpu_ccso_block_window(
            g_hook.ctx, g_hook.ccso, src_y, src_stride, dst_yuv, dst_stride,
            pic_width, pic_height, x, y, offset_buf, src_loc_0, src_loc_1,
            y_uv_hscale, y_uv_vscale, thr, neg_thr, max_val, blk_w, blk_h,
            single_band, shift_bits, edge_clf, bo_only, err) == 0) {
      g_hook.blocks++;
      handled = 1;
    } else if (getenv("AV2_GPU_CCSO_DEBUG")) {
      fprintf(stderr, "avm_gpu: CCSO fallback at (%d,%d): %s\n", x, y, err);
    }
  }
  pthread_mutex_unlock(&g_lock);
  return handled;
}
