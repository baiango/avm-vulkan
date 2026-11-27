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

#include "avm_gpu/vulkan_hook_deblock.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "avm_gpu/vulkan_deblock.h"
#include "avm_gpu/vulkan_gpu.h"

/* The C reference, used by AV2_GPU_DEBLOCK_VERIFY. */
void avm_highbd_lpf_vertical_generic_c(uint16_t *s, int pitch,
                                       int filt_width_neg, int filt_width_pos,
                                       const uint16_t *q_thresh,
                                       const uint16_t *side_thresh, int bd,
                                       int is_lossless_neg,
                                       int is_lossless_pos);
void avm_highbd_lpf_horizontal_generic_c(uint16_t *s, int pitch,
                                         int filt_width_neg,
                                         int filt_width_pos,
                                         const uint16_t *q_thresh,
                                         const uint16_t *side_thresh, int bd,
                                         int is_lossless_neg,
                                         int is_lossless_pos);

/* The filter can read MAX_DBL_FLT_LEN (8) samples across the edge plus two for
 * its second-derivative tests, on either side. */
#define HOOK_REACH (8 + 2)
#define HOOK_WSPAN (2 * HOOK_REACH + 1)
#define HOOK_COUNT 4
#define HOOK_MAX_WINDOW (HOOK_WSPAN * HOOK_COUNT)

typedef struct deblock_hook_state {
  int state; /* 0 = uninitialised, 1 = off, 2 = ready */
  avm_gpu_ctx *ctx;
  avm_gpu_deblock *db;
  unsigned long edges;
  uint16_t in[HOOK_MAX_WINDOW];   /* the window before filtering */
  uint16_t c[HOOK_MAX_WINDOW];    /* the C result, same layout as in */
  uint16_t tr[HOOK_MAX_WINDOW];   /* transpose scratch for the horizontal C */
} deblock_hook_state;

static deblock_hook_state g_hook;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static void print_stats(void) {
  if (getenv("AV2_GPU_DEBLOCK_STATS")) {
    fprintf(stderr, "avm_gpu: %lu deblock edges on the GPU\n", g_hook.edges);
  }
}

/* Caller holds g_lock. */
static int hook_ready_locked(void) {
  if (g_hook.state) return g_hook.state == 2;

  const char *env = getenv("AV2_GPU_DEBLOCK");
  if (!env || !*env || !strcmp(env, "0")) {
    g_hook.state = 1;
    return 0;
  }
  char err[AVM_GPU_ERR_LEN];
  if (avm_gpu_ctx_create(&g_hook.ctx, err) != 0 ||
      avm_gpu_deblock_create(g_hook.ctx, &g_hook.db, err) != 0) {
    fprintf(stderr, "avm_gpu: deblock hook disabled (%s)\n", err);
    avm_gpu_ctx_destroy(g_hook.ctx);
    g_hook.ctx = NULL;
    g_hook.state = 1;
    return 0;
  }
  atexit(print_stats);
  fprintf(stderr, "avm_gpu: deblock hook enabled on %s\n",
          avm_gpu_device_name(g_hook.ctx));
  g_hook.state = 2;
  return 1;
}

/*
 * Runs the C filter on the snapshot in `g_hook.in` and leaves the result in
 * `g_hook.c`, both in the compact window layout of `w` (4 rows of wspan, the
 * edge at column HOOK_REACH). The two C orientations disagree on what `pitch`
 * means: the vertical one steps rows by it and filters across with 1, the
 * horizontal one steps rows by 1 and filters across with it, so the horizontal
 * case needs the window transposed first.
 */
static void run_c_window(int filt_neg, int filt_pos, int q_thresh,
                         int side_thresh, int bd, int ll_neg, int ll_pos,
                         int vertical) {
  const uint16_t q = (uint16_t)q_thresh;
  const uint16_t side = (uint16_t)side_thresh;
  if (vertical) {
    memcpy(g_hook.c, g_hook.in, sizeof(g_hook.in));
    avm_highbd_lpf_vertical_generic_c(g_hook.c + HOOK_REACH, HOOK_WSPAN,
                                      filt_neg, filt_pos, &q, &side, bd,
                                      ll_neg, ll_pos);
  } else {
    for (int j = 0; j < HOOK_COUNT; j++) {
      for (int i = 0; i < HOOK_WSPAN; i++) {
        g_hook.tr[i * HOOK_COUNT + j] = g_hook.in[j * HOOK_WSPAN + i];
      }
    }
    avm_highbd_lpf_horizontal_generic_c(g_hook.tr + HOOK_REACH * HOOK_COUNT,
                                        HOOK_COUNT, filt_neg, filt_pos, &q,
                                        &side, bd, ll_neg, ll_pos);
    for (int j = 0; j < HOOK_COUNT; j++) {
      for (int i = 0; i < HOOK_WSPAN; i++) {
        g_hook.c[j * HOOK_WSPAN + i] = g_hook.tr[i * HOOK_COUNT + j];
      }
    }
  }
}

int avm_gpu_hook_deblock(uint16_t *p, int across_pitch, int step, int filt_neg,
                         int filt_pos, int q_thresh, int side_thresh, int bd,
                         int ll_neg, int ll_pos, int vertical) {
  if (!p || across_pitch <= 0 || step <= 0) return 0;
  /* The filter never reaches wider than its own length on the positive side. */
  if (filt_pos > 8 || filt_neg > 8) return 0;

  pthread_mutex_lock(&g_lock);
  int handled = 0;
  const int verify =
      hook_ready_locked() && getenv("AV2_GPU_DEBLOCK_VERIFY") != NULL;
  if (hook_ready_locked()) {
    if (verify) {
      /* Snapshot first: deblocking is not idempotent, so the C has to see the
       * same input the GPU sees, and the GPU writes `p` in place. */
      for (int j = 0; j < HOOK_COUNT; j++) {
        for (int i = 0; i < HOOK_WSPAN; i++) {
          g_hook.in[j * HOOK_WSPAN + i] =
              p[j * step + (i - HOOK_REACH) * across_pitch];
        }
      }
    }
    char err[AVM_GPU_ERR_LEN];
    if (avm_gpu_deblock_window(g_hook.ctx, g_hook.db, p, across_pitch, step,
                               filt_neg, filt_pos, q_thresh, side_thresh, bd,
                               ll_neg, ll_pos, p, err) == 0) {
      g_hook.edges++;
      handled = 1;
    } else if (getenv("AV2_GPU_DEBLOCK_DEBUG")) {
      fprintf(stderr, "avm_gpu: deblock fallback: %s\n", err);
    }
  }

  if (handled && verify) {
    run_c_window(filt_neg, filt_pos, q_thresh, side_thresh, bd, ll_neg, ll_pos,
                 vertical);
    int diff = 0, fi = 0, fj = 0;
    for (int j = 0; j < HOOK_COUNT; j++) {
      for (int i = 0; i < HOOK_WSPAN; i++) {
        const uint16_t got = p[j * step + (i - HOOK_REACH) * across_pitch];
        if (got != g_hook.c[j * HOOK_WSPAN + i]) {
          if (!diff) {
            fi = i;
            fj = j;
          }
          diff++;
        }
      }
    }
    if (diff) {
      fprintf(stderr,
              "avm_gpu deblock verify: %s filt %d/%d q %d side %d bd %d "
              "ll %d/%d: %d samples differ, first at %+d gpu %u cpu %u\n",
              vertical ? "vertical" : "horizontal", filt_neg, filt_pos,
              q_thresh, side_thresh, bd, ll_neg, ll_pos, diff,
              fi - HOOK_REACH,
              p[fj * step + (fi - HOOK_REACH) * across_pitch],
              g_hook.c[fj * HOOK_WSPAN + fi]);
      /* Keep the stream on the C while debugging. */
      for (int j = 0; j < HOOK_COUNT; j++) {
        for (int i = 0; i < HOOK_WSPAN; i++) {
          p[j * step + (i - HOOK_REACH) * across_pitch] =
              g_hook.c[j * HOOK_WSPAN + i];
        }
      }
    }
  }
  pthread_mutex_unlock(&g_lock);
  return handled;
}
