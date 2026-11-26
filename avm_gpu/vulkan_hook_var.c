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

#include "avm_gpu/vulkan_hook_var.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "avm_gpu/vulkan_gpu.h"
#include "avm_gpu/vulkan_var.h"

#define HOOK_MAX_DIM 128

typedef struct var_hook_state {
  int state; /* 0 = uninitialised, 1 = off, 2 = ready */
  avm_gpu_ctx *ctx;
  avm_gpu_var *var;
  uint16_t a[HOOK_MAX_DIM * HOOK_MAX_DIM];
  uint16_t b[HOOK_MAX_DIM * HOOK_MAX_DIM];
  unsigned long blocks;
} var_hook_state;

static var_hook_state g_hook;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static void print_stats(void) {
  if (g_hook.blocks && getenv("AV2_GPU_VAR_STATS")) {
    fprintf(stderr, "avm_gpu: %lu variance blocks on the GPU\n", g_hook.blocks);
  }
}

/* Returns 1 when the GPU path is ready. Caller holds g_lock. */
static int hook_init_locked(void) {
  if (g_hook.state) return g_hook.state == 2;

  const char *env = getenv("AV2_GPU_VAR");
  if (!env || !*env || !strcmp(env, "0")) {
    g_hook.state = 1;
    return 0;
  }

  char err[AVM_GPU_ERR_LEN];
  if (avm_gpu_ctx_create(&g_hook.ctx, err) != 0 ||
      avm_gpu_var_create(g_hook.ctx, &g_hook.var, err) != 0) {
    fprintf(stderr, "avm_gpu: variance hook disabled (%s)\n", err);
    avm_gpu_ctx_destroy(g_hook.ctx);
    g_hook.ctx = NULL;
    g_hook.state = 1;
    return 0;
  }
  atexit(print_stats);
  fprintf(stderr, "avm_gpu: variance hook enabled on %s\n",
          avm_gpu_device_name(g_hook.ctx));
  g_hook.state = 2;
  return 1;
}

int avm_gpu_hook_variance64(const uint16_t *a, int a_stride, const uint16_t *b,
                            int b_stride, int w, int h, uint64_t *sse,
                            int64_t *sum) {
  if (!a || !b || !sse || !sum) return 0;
  if (w <= 0 || h <= 0 || w > HOOK_MAX_DIM || h > HOOK_MAX_DIM) return 0;

  pthread_mutex_lock(&g_lock);
  int handled = 0;
  if (hook_init_locked()) {
    for (int y = 0; y < h; y++) {
      memcpy(g_hook.a + (size_t)y * w, a + (size_t)y * a_stride,
             (size_t)w * sizeof(uint16_t));
      memcpy(g_hook.b + (size_t)y * w, b + (size_t)y * b_stride,
             (size_t)w * sizeof(uint16_t));
    }
    char err[AVM_GPU_ERR_LEN];
    if (avm_gpu_var_set_planes(g_hook.ctx, g_hook.var, g_hook.a, w, h, g_hook.b,
                               w, h, err) == 0) {
      const avm_gpu_var_pos pos = { 0, 0 };
      uint64_t gsse = 0;
      int64_t gsum = 0;
      if (avm_gpu_var_run_raw(g_hook.ctx, g_hook.var, &pos, 1, w, h, &gsse,
                              &gsum, err) == 0) {
        *sse = gsse;
        *sum = gsum;
        g_hook.blocks++;
        handled = 1;
      }
    }
  }
  pthread_mutex_unlock(&g_lock);
  return handled;
}
