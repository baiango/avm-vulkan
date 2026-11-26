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
#define HOOK_MAX_BATCH AVM_GPU_HOOK_MAX_BATCH

/* AV2_GPU_SAD values: 0/off, 1/batch, single, all. */
#define HOOK_MODE_OFF 0
#define HOOK_MODE_BATCH 1
#define HOOK_MODE_SINGLE 2
#define HOOK_MODE_ALL 3

typedef struct hook_state {
  int state; /* 0 = uninitialised, 1 = off, 2 = ready */
  int mode;
  avm_gpu_ctx *ctx;
  avm_gpu_sad *sad;
  uint16_t *src_copy;
  uint16_t *ref_copy;
  avm_gpu_sad_pos *pos;
  size_t cap;
  size_t ref_cap;
  size_t pos_cap;
  unsigned long blocks;
  unsigned long dispatches;
} hook_state;

static hook_state g_hook;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static void print_stats(void) {
  if (g_hook.blocks && getenv("AV2_GPU_SAD_STATS")) {
    fprintf(stderr,
            "avm_gpu: %lu SAD blocks in %lu dispatches (%.1f blocks/dispatch)\n",
            g_hook.blocks, g_hook.dispatches,
            g_hook.dispatches ? (double)g_hook.blocks / g_hook.dispatches : 0.0);
  }
}

static int parse_mode(const char *env) {
  if (!env || !*env || !strcmp(env, "0") || !strcmp(env, "off")) {
    return HOOK_MODE_OFF;
  }
  if (!strcmp(env, "single")) return HOOK_MODE_SINGLE;
  if (!strcmp(env, "all")) return HOOK_MODE_ALL;
  return HOOK_MODE_BATCH;
}

/* Returns 1 when the GPU path is ready. Caller holds g_lock. */
static int hook_init_locked(void) {
  if (g_hook.state) return g_hook.state == 2;

  g_hook.mode = parse_mode(getenv("AV2_GPU_SAD"));
  if (g_hook.mode == HOOK_MODE_OFF) {
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
  fprintf(stderr, "avm_gpu: SAD hook enabled on %s (%s mode)\n",
          avm_gpu_device_name(g_hook.ctx),
          g_hook.mode == HOOK_MODE_SINGLE ? "single" : "batch");
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
  g_hook.blocks++;
  g_hook.dispatches++;
  return 1;
}

/* Caller holds g_lock. Packs n candidate blocks and prices them in one
 * dispatch. Entries with a NULL reference pointer are left untouched. */
static int hook_multi_locked(const uint16_t *src, int src_stride,
                             const uint16_t *const *ref_ptrs, int ref_stride,
                             int width, int height, int n,
                             unsigned int *sad) {
  const size_t block = (size_t)width * height;
  const size_t ref_need = block * (size_t)n;
  if (block > g_hook.cap) {
    free(g_hook.src_copy);
    g_hook.src_copy = malloc(block * sizeof(uint16_t));
    g_hook.cap = g_hook.src_copy ? block : 0;
    if (!g_hook.cap) return 0;
  }
  if (ref_need > g_hook.ref_cap) {
    free(g_hook.ref_copy);
    g_hook.ref_copy = malloc(ref_need * sizeof(uint16_t));
    g_hook.ref_cap = g_hook.ref_copy ? ref_need : 0;
    if (!g_hook.ref_cap) return 0;
  }
  if ((size_t)n > g_hook.pos_cap) {
    free(g_hook.pos);
    g_hook.pos = malloc((size_t)n * sizeof(*g_hook.pos));
    g_hook.pos_cap = g_hook.pos ? (size_t)n : 0;
    if (!g_hook.pos_cap) return 0;
  }

  for (int y = 0; y < height; y++) {
    memcpy(g_hook.src_copy + (size_t)y * width, src + (size_t)y * src_stride,
           (size_t)width * sizeof(uint16_t));
  }
  for (int i = 0; i < n; i++) {
    if (!ref_ptrs[i]) continue;
    for (int y = 0; y < height; y++) {
      memcpy(g_hook.ref_copy + block * (size_t)i + (size_t)y * width,
             ref_ptrs[i] + (size_t)y * ref_stride,
             (size_t)width * sizeof(uint16_t));
    }
    g_hook.pos[i].src_off = 0;
    g_hook.pos[i].ref_off = (uint32_t)(block * (size_t)i);
  }

  char err[AVM_GPU_ERR_LEN];
  if (avm_gpu_sad_set_planes(g_hook.ctx, g_hook.sad, g_hook.src_copy, width,
                             height, g_hook.ref_copy, width, height * n,
                             err) != 0) {
    return 0;
  }
  uint32_t *costs = malloc((size_t)n * sizeof(*costs));
  if (!costs) return 0;
  const int rc = avm_gpu_sad_run(g_hook.ctx, g_hook.sad, g_hook.pos, n, width,
                                 height, costs, err);
  if (rc == 0) {
    for (int i = 0; i < n; i++) {
      if (!ref_ptrs[i]) continue;
      sad[i] = costs[i];
      g_hook.blocks++;
    }
    g_hook.dispatches++;
  }
  free(costs);
  return rc == 0;
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
  if (hook_init_locked() &&
      (g_hook.mode == HOOK_MODE_SINGLE || g_hook.mode == HOOK_MODE_ALL)) {
    handled = hook_run_locked(src, src_stride, ref, ref_stride, width, height,
                              sad);
  }
  pthread_mutex_unlock(&g_lock);
  return handled;
}

int avm_gpu_hook_sad_multi(const uint16_t *src, int src_stride,
                           const uint16_t *const *ref_ptrs, int ref_stride,
                           int width, int height, int n, unsigned int *sad) {
  if (!src || !ref_ptrs || !sad) return 0;
  if (width <= 0 || height <= 0 || width > HOOK_MAX_BLOCK ||
      height > HOOK_MAX_BLOCK || n <= 0 || n > HOOK_MAX_BATCH) {
    return 0;
  }

  pthread_mutex_lock(&g_lock);
  int handled = 0;
  if (hook_init_locked() &&
      (g_hook.mode == HOOK_MODE_BATCH || g_hook.mode == HOOK_MODE_ALL)) {
    handled = hook_multi_locked(src, src_stride, ref_ptrs, ref_stride, width,
                                height, n, sad);
  }
  pthread_mutex_unlock(&g_lock);
  return handled;
}
