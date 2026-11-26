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

#include "avm_gpu/vulkan_hook.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "avm_gpu/vulkan_gpu.h"
#include "avm_gpu/vulkan_sad.h"

/* Largest block the encoder can hand to a single SAD call (256x256 AV2 block). */
#define HOOK_MAX_BLOCK 256

typedef struct hook_state {
  int state; /* 0 = uninitialised, 1 = off, 2 = ready */
  avm_gpu_ctx *ctx;
  avm_gpu_sad *sad;
  uint16_t *src_copy;
  uint16_t *ref_copy;
  size_t cap;
  unsigned long calls;
} hook_state;

static hook_state g_hook;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static void print_stats(void) {
  if (g_hook.calls && getenv("AV2_GPU_SAD_STATS")) {
    fprintf(stderr, "avm_gpu: %lu SAD blocks priced on the GPU\n", g_hook.calls);
  }
}

/* Returns 1 when the GPU path is ready. Caller holds g_lock. */
static int hook_init_locked(void) {
  if (g_hook.state) return g_hook.state == 2;

  const char *env = getenv("AV2_GPU_SAD");
  if (!env || !*env || !strcmp(env, "0")) {
    g_hook.state = 1;
    return 0;
  }

  char err[AVM_GPU_ERR_LEN];
  if (avm_gpu_ctx_create(&g_hook.ctx, err) != 0) {
    fprintf(stderr, "avm_gpu: SAD hook disabled (%s)\n", err);
    g_hook.state = 1;
    return 0;
  }
  if (avm_gpu_sad_create(g_hook.ctx, &g_hook.sad, err) != 0) {
    fprintf(stderr, "avm_gpu: SAD hook disabled (%s)\n", err);
    avm_gpu_ctx_destroy(g_hook.ctx);
    g_hook.ctx = NULL;
    g_hook.state = 1;
    return 0;
  }
  atexit(print_stats);
  fprintf(stderr, "avm_gpu: SAD hook enabled on %s\n",
          avm_gpu_device_name(g_hook.ctx));
  g_hook.state = 2;
  return 1;
}

/* Caller holds g_lock. */
static int hook_run_locked(const uint16_t *src, int src_stride,
                           const uint16_t *ref, int ref_stride, int width,
                           int height, unsigned int *sad) {
  const size_t n = (size_t)width * height;
  if (n > g_hook.cap) {
    free(g_hook.src_copy);
    free(g_hook.ref_copy);
    g_hook.src_copy = malloc(n * sizeof(uint16_t));
    g_hook.ref_copy = malloc(n * sizeof(uint16_t));
    g_hook.cap = (g_hook.src_copy && g_hook.ref_copy) ? n : 0;
    if (!g_hook.cap) return 0;
  }
  for (int y = 0; y < height; y++) {
    memcpy(g_hook.src_copy + (size_t)y * width, src + (size_t)y * src_stride,
           (size_t)width * sizeof(uint16_t));
    memcpy(g_hook.ref_copy + (size_t)y * width, ref + (size_t)y * ref_stride,
           (size_t)width * sizeof(uint16_t));
  }

  char err[AVM_GPU_ERR_LEN];
  if (avm_gpu_sad_set_planes(g_hook.ctx, g_hook.sad, g_hook.src_copy, width,
                             height, g_hook.ref_copy, width, height,
                             err) != 0) {
    return 0;
  }
  const avm_gpu_sad_pos pos = { 0, 0 };
  uint32_t cost = 0;
  if (avm_gpu_sad_run(g_hook.ctx, g_hook.sad, &pos, 1, width, height, &cost,
                      err) != 0) {
    return 0;
  }
  *sad = cost;
  g_hook.calls++;
  return 1;
}

int avm_gpu_hook_sad(const uint16_t *src, int src_stride, const uint16_t *ref,
                     int ref_stride, int width, int height,
                     unsigned int *sad) {
  if (!src || !ref || !sad) return 0;
  if (width <= 0 || height <= 0 || width > HOOK_MAX_BLOCK ||
      height > HOOK_MAX_BLOCK) {
    return 0;
  }

  pthread_mutex_lock(&g_lock);
  int handled = 0;
  if (hook_init_locked()) {
    handled = hook_run_locked(src, src_stride, ref, ref_stride, width, height,
                              sad);
  }
  pthread_mutex_unlock(&g_lock);
  return handled;
}
