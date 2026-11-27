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

#include "avm_gpu/vulkan_hook_cdef.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "avm_gpu/vulkan_cdef.h"
#include "avm_gpu/vulkan_gpu.h"

/* The C reference, used by AV2_GPU_CDEF_VERIFY. */
void av2_cdef_filter_16_0_c(uint16_t *const dst16, int dstride,
                            const uint16_t *in, int pri_strength,
                            int sec_strength, int dir, int pri_damping,
                            int sec_damping, int coeff_shift, int block_width,
                            int block_height);
void av2_cdef_filter_16_1_c(uint16_t *const dst16, int dstride,
                            const uint16_t *in, int pri_strength,
                            int sec_strength, int dir, int pri_damping,
                            int sec_damping, int coeff_shift, int block_width,
                            int block_height);
void av2_cdef_filter_16_2_c(uint16_t *const dst16, int dstride,
                            const uint16_t *in, int pri_strength,
                            int sec_strength, int dir, int pri_damping,
                            int sec_damping, int coeff_shift, int block_width,
                            int block_height);
void av2_cdef_filter_16_3_c(uint16_t *const dst16, int dstride,
                            const uint16_t *in, int pri_strength,
                            int sec_strength, int dir, int pri_damping,
                            int sec_damping, int coeff_shift, int block_width,
                            int block_height);

#define HOOK_VBORDER 2
#define HOOK_HBORDER 8
#define HOOK_MAX_BLOCK 8
#define HOOK_MAX_STRIDE 16

typedef struct cdef_hook_state {
  int state; /* 0 = uninitialised, 1 = off, 2 = ready */
  avm_gpu_ctx *ctx;
  avm_gpu_cdef *cdef;
  unsigned long blocks;
  uint16_t ref[HOOK_MAX_BLOCK * HOOK_MAX_STRIDE];
} cdef_hook_state;

static cdef_hook_state g_hook;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static void print_stats(void) {
  if (getenv("AV2_GPU_CDEF_STATS")) {
    fprintf(stderr, "avm_gpu: %lu CDEF blocks on the GPU\n", g_hook.blocks);
  }
}

/* Caller holds g_lock. */
static int hook_ready_locked(void) {
  if (g_hook.state) return g_hook.state == 2;

  const char *env = getenv("AV2_GPU_CDEF");
  if (!env || !*env || !strcmp(env, "0")) {
    g_hook.state = 1;
    return 0;
  }
  char err[AVM_GPU_ERR_LEN];
  if (avm_gpu_ctx_create(&g_hook.ctx, err) != 0 ||
      avm_gpu_cdef_create(g_hook.ctx, &g_hook.cdef, err) != 0) {
    fprintf(stderr, "avm_gpu: CDEF hook disabled (%s)\n", err);
    avm_gpu_ctx_destroy(g_hook.ctx);
    g_hook.ctx = NULL;
    g_hook.state = 1;
    return 0;
  }
  atexit(print_stats);
  fprintf(stderr, "avm_gpu: CDEF hook enabled on %s\n",
          avm_gpu_device_name(g_hook.ctx));
  g_hook.state = 2;
  return 1;
}

static void run_c(int variant, uint16_t *dst, int dstride, const uint16_t *in,
                  int pri_strength, int sec_strength, int dir, int pri_damping,
                  int sec_damping, int coeff_shift, int w, int h) {
  switch (variant) {
    case 0:
      av2_cdef_filter_16_0_c(dst, dstride, in, pri_strength, sec_strength, dir,
                             pri_damping, sec_damping, coeff_shift, w, h);
      break;
    case 1:
      av2_cdef_filter_16_1_c(dst, dstride, in, pri_strength, sec_strength, dir,
                             pri_damping, sec_damping, coeff_shift, w, h);
      break;
    case 2:
      av2_cdef_filter_16_2_c(dst, dstride, in, pri_strength, sec_strength, dir,
                             pri_damping, sec_damping, coeff_shift, w, h);
      break;
    default:
      av2_cdef_filter_16_3_c(dst, dstride, in, pri_strength, sec_strength, dir,
                             pri_damping, sec_damping, coeff_shift, w, h);
      break;
  }
}

int avm_gpu_hook_cdef(const uint16_t *in, int in_rows, int stride, int i0,
                      int j0, int w, int h, int dir, int pri_strength,
                      int sec_strength, int pri_damping, int sec_damping,
                      int coeff_shift, int enable_pri, int enable_sec,
                      uint16_t *dst, int dstride) {
  (void)in_rows;
  if (!in || !dst || stride <= 0 || w <= 0 || h <= 0 || w > HOOK_MAX_BLOCK ||
      h > HOOK_MAX_BLOCK || dstride > HOOK_MAX_STRIDE) {
    return 0;
  }
  /* The filter reaches CDEF_VBORDER (2) rows and CDEF_HBORDER (8) samples
   * around the block, and the caller's source buffer has exactly those borders
   * around it, so the whole CDEF buffer never has to be uploaded. */
  const uint16_t *win = in + (size_t)(i0 - HOOK_VBORDER) * stride +
                        (j0 - HOOK_HBORDER);

  pthread_mutex_lock(&g_lock);
  int handled = 0;
  const int verify = hook_ready_locked() && getenv("AV2_GPU_CDEF_VERIFY");
  const int variant = (enable_pri ? 0 : 2) | (enable_sec ? 0 : 1);
  if (hook_ready_locked()) {
    char err[AVM_GPU_ERR_LEN];
    const int rc = avm_gpu_cdef_filter16(
        g_hook.ctx, g_hook.cdef, win, h + 2 * HOOK_VBORDER, stride, HOOK_VBORDER,
        HOOK_HBORDER, w, h, dir, pri_strength, sec_strength, pri_damping,
        sec_damping, coeff_shift, enable_pri, enable_sec, dst, dstride, err);
    if (rc == 0) {
      g_hook.blocks++;
      handled = 1;
    } else if (getenv("AV2_GPU_CDEF_DEBUG")) {
      fprintf(stderr, "avm_gpu: CDEF fallback: %s\n", err);
    }
  }

  if (handled && verify) {
    run_c(variant, g_hook.ref, HOOK_MAX_STRIDE, in + (size_t)i0 * stride + j0,
          pri_strength, sec_strength, dir, pri_damping, sec_damping,
          coeff_shift, w, h);
    int diff = 0, fx = 0, fy = 0;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        if (dst[y * dstride + x] != g_hook.ref[y * HOOK_MAX_STRIDE + x]) {
          if (!diff) {
            fx = x;
            fy = y;
          }
          diff++;
        }
      }
    }
    if (diff) {
      fprintf(stderr,
              "avm_gpu CDEF verify: variant %d dir %d pri %d sec %d damp %d/%d "
              "shift %d %dx%d: %d samples differ, first (%d,%d) gpu %u cpu %u\n",
              variant, dir, pri_strength, sec_strength, pri_damping, sec_damping,
              coeff_shift, w, h, diff, fx, fy, dst[fy * dstride + fx],
              g_hook.ref[fy * HOOK_MAX_STRIDE + fx]);
      for (int y = 0; y < h; y++) {
        memcpy(dst + y * dstride, g_hook.ref + y * HOOK_MAX_STRIDE,
               (size_t)w * sizeof(*dst));
      }
    }
  }
  pthread_mutex_unlock(&g_lock);
  return handled;
}
