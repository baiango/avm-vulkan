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

/* Below this block area a dispatch costs far more than the CPU loop (a fence
 * round trip is ~0.2 ms through MoltenVK, a 16x16 variance is well under a
 * microsecond), so those calls are left to the CPU.
 * AV2_GPU_VAR_MIN_AREA overrides it, 0 sends every block to the GPU. */
#define HOOK_DEFAULT_MIN_AREA 1024

typedef struct var_hook_state {
  int state; /* 0 = uninitialised, 1 = off, 2 = ready */
  avm_gpu_ctx *ctx;
  avm_gpu_var *var;
  uint16_t a[HOOK_MAX_DIM * HOOK_MAX_DIM];
  uint16_t b[HOOK_MAX_DIM * HOOK_MAX_DIM];
  unsigned long blocks;
  unsigned long memo_hits, memo_misses;
  unsigned long size_calls[256]; /* calls by block area, power of two */
  int min_area;                  /* calls below this area stay on the CPU */
  int memo_on;
} var_hook_state;

/*
 * Content-keyed memo.
 *
 * sse and sum are a pure function of the two blocks and their shape, so a hit
 * cannot go stale: the key *is* the content. That makes it sound to reuse a
 * result across calls without a scope or a generation token. Blocks are
 * hashed with FNV-1a over 64-bit words plus a 32-bit checksum, so a false hit
 * needs a 96-bit coincidence.
 *
 * The RD and motion-search loops ask for the same block pairs many times
 * (the same source block against the same predictor in several trials), and
 * every avoided call is a whole dispatch saved.
 */
#define MEMO_SIZE (1u << 18)
typedef struct memo_entry {
  uint64_t key;   /* FNV-1a of both blocks and the shape */
  uint32_t check; /* independent checksum */
  int32_t w, h;
  uint64_t sse;
  int64_t sum;
} memo_entry;

static memo_entry g_memo[MEMO_SIZE];

static uint64_t fnv_words(uint64_t h, const uint16_t *p, int stride, int w,
                          int hgt, uint32_t *check) {
  for (int y = 0; y < hgt; y++) {
    const uint16_t *row = p + (size_t)y * stride;
    int x = 0;
    for (; x + 4 <= w; x += 4) {
      uint64_t v;
      memcpy(&v, row + x, sizeof(v));
      h = (h ^ v) * 0x100000001b3ull;
      *check = (*check ^ (uint32_t)v) * 0x9e3779b1u;
    }
    for (; x < w; x++) {
      h = (h ^ row[x]) * 0x100000001b3ull;
      *check = (*check ^ (uint32_t)row[x]) * 0x9e3779b1u;
    }
  }
  return h;
}

static var_hook_state g_hook;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static void print_stats(void) {
  if (g_hook.blocks && getenv("AV2_GPU_VAR_STATS")) {
    fprintf(stderr, "avm_gpu: %lu variance blocks on the GPU\n", g_hook.blocks);
  }
  if (getenv("AV2_GPU_VAR_STATS")) {
    for (int b = 4; b < 256; b++) {
      if (!g_hook.size_calls[b]) continue;
      fprintf(stderr, "avm_gpu:   area %d: %lu calls\n", 1 << b,
              g_hook.size_calls[b]);
    }
  }
  if (getenv("AV2_GPU_VAR_STATS") && (g_hook.memo_hits || g_hook.memo_misses)) {
    fprintf(stderr, "avm_gpu: variance memo %lu hits / %lu misses\n",
            g_hook.memo_hits, g_hook.memo_misses);
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
  const char *area_env = getenv("AV2_GPU_VAR_MIN_AREA");
  g_hook.min_area = area_env && *area_env ? atoi(area_env)
                                          : HOOK_DEFAULT_MIN_AREA;
  const char *memo_env = getenv("AV2_GPU_VAR_MEMO");
  g_hook.memo_on = !(memo_env && !strcmp(memo_env, "0"));
  atexit(print_stats);
  fprintf(stderr, "avm_gpu: variance hook enabled on %s (min area %d)\n",
          avm_gpu_device_name(g_hook.ctx), g_hook.min_area);
  g_hook.state = 2;
  return 1;
}

/* Below this block area a dispatch costs far more than the CPU loop (a fence
 * round trip is ~0.2 ms through MoltenVK, a 16x16 variance is well under a
 * microsecond), so those calls are left to the CPU.
 * AV2_GPU_VAR_MIN_AREA overrides it, 0 sends every block to the GPU. */
#define HOOK_DEFAULT_MIN_AREA 1024

int avm_gpu_hook_variance64(const uint16_t *a, int a_stride, const uint16_t *b,
                            int b_stride, int w, int h, uint64_t *sse,
                            int64_t *sum) {
  if (!a || !b || !sse || !sum) return 0;
  if (w <= 0 || h <= 0 || w > HOOK_MAX_DIM || h > HOOK_MAX_DIM) return 0;
  /* Cheap pre-check so a disabled hook costs nothing: the real threshold is
   * applied after initialisation. */
  if (w * h < 16) return 0;

  pthread_mutex_lock(&g_lock);
  int handled = 0;
  if (hook_init_locked() && w * h >= g_hook.min_area) {
    {
      int area = w * h, bit = 0;
      while ((1 << bit) < area) bit++;
      g_hook.size_calls[bit]++;
    }
    const int memo_on = g_hook.memo_on;
    uint32_t check = 0x811c9dc5u;
    uint64_t key = 0;
    memo_entry *slot = NULL;
    if (memo_on) {
      key = fnv_words(0xcbf29ce484222325ull, a, a_stride, w, h, &check);
      key = fnv_words(key, b, b_stride, w, h, &check);
      key = (key ^ (uint64_t)(uint32_t)w ^ ((uint64_t)(uint32_t)h << 32)) *
            0x100000001b3ull;
      slot = &g_memo[key & (MEMO_SIZE - 1)];
      if (slot->key == key && slot->check == check && slot->w == w &&
          slot->h == h) {
        *sse = slot->sse;
        *sum = slot->sum;
        g_hook.memo_hits++;
        g_hook.blocks++;
        handled = 1;
      }
    }
    if (handled) {
      pthread_mutex_unlock(&g_lock);
      return 1;
    }
    if (memo_on) g_hook.memo_misses++;
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
        if (slot) {
          slot->key = key;
          slot->check = check;
          slot->w = w;
          slot->h = h;
          slot->sse = gsse;
          slot->sum = gsum;
        }
      }
    }
  }
  pthread_mutex_unlock(&g_lock);
  return handled;
}
