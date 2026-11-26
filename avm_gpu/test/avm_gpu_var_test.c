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

/*
 * Parity gate for the Vulkan variance/SSE kernel against
 * avm_highbd_{8,10,12}_varianceWxH_c, over three bitdepths.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "config/avm_dsp_rtcd.h"

#include "avm_gpu/vulkan_gpu.h"
#include "avm_gpu/vulkan_var.h"

#define PLANE_W 512
#define PLANE_H 512
#define BLOCKS_PER_CASE 128

typedef uint32_t (*var_fn)(const uint16_t *, int, const uint16_t *, int,
                           uint32_t *);

typedef struct {
  const char *name;
  int bw, bh;
  var_fn ref[3]; /* 8, 10, 12 bit */
} case_t;

#define VCASE(w, h)                                                            \
  {                                                                            \
    #w "x" #h, w, h, {                                                         \
      avm_highbd_8_variance##w##x##h##_c, avm_highbd_10_variance##w##x##h##_c, \
          avm_highbd_12_variance##w##x##h##_c                                   \
    }                                                                          \
  }

static const case_t kCases[] = {
  VCASE(4, 4),   VCASE(4, 8),   VCASE(8, 8),   VCASE(8, 16), VCASE(16, 16),
  VCASE(16, 32), VCASE(32, 32), VCASE(32, 64), VCASE(64, 64),
};

static const int kBitdepths[3] = { 8, 10, 12 };

static unsigned int rng_state = 0x2468aceu;
static unsigned int rnd(void) {
  rng_state = rng_state * 1664525u + 1013904223u;
  return rng_state >> 8;
}

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(void) {
  char err[AVM_GPU_ERR_LEN];
  avm_gpu_ctx *ctx = NULL;
  if (avm_gpu_ctx_create(&ctx, err) != 0) {
    fprintf(stderr, "avm_gpu: %s\n", err);
    return 77; /* skip: no Vulkan device */
  }
  avm_gpu_var *var = NULL;
  if (avm_gpu_var_create(ctx, &var, err) != 0) {
    fprintf(stderr, "avm_gpu var: %s\n", err);
    avm_gpu_ctx_destroy(ctx);
    return 1;
  }
  printf("device: %s\n", avm_gpu_device_name(ctx));
  printf("%-3s %-7s %8s %10s %10s %s\n", "bit", "block", "blocks", "cpu ms",
         "gpu ms", "parity");

  uint16_t *src = malloc(sizeof(uint16_t) * PLANE_W * PLANE_H);
  uint16_t *ref = malloc(sizeof(uint16_t) * PLANE_W * PLANE_H);
  if (!src || !ref) return 1;

  avm_gpu_var_pos pos[BLOCKS_PER_CASE];
  uint32_t gpu_var[BLOCKS_PER_CASE], gpu_sse[BLOCKS_PER_CASE];
  uint32_t cpu_var[BLOCKS_PER_CASE];
  int failures = 0;

  /* Warm the pipeline once. */
  {
    for (int i = 0; i < PLANE_W * PLANE_H; i++) {
      src[i] = (uint16_t)(rnd() & 0xff);
      ref[i] = (uint16_t)(rnd() & 0xff);
    }
    if (avm_gpu_var_set_planes(ctx, var, src, PLANE_W, PLANE_H, ref, PLANE_W,
                               PLANE_H, err) != 0) {
      fprintf(stderr, "avm_gpu var planes: %s\n", err);
      return 1;
    }
    avm_gpu_var_pos w0 = { 0, 0 };
    uint32_t wv = 0, ws = 0;
    if (avm_gpu_var_run(ctx, var, &w0, 1, 64, 64, 8, &wv, &ws, err) != 0) {
      fprintf(stderr, "avm_gpu var warm-up: %s\n", err);
      return 1;
    }
  }

  for (int b = 0; b < 3; b++) {
    const int bd = kBitdepths[b];
    const unsigned mask = (1u << bd) - 1;
    for (int i = 0; i < PLANE_W * PLANE_H; i++) {
      src[i] = (uint16_t)(rnd() & mask);
      ref[i] = (uint16_t)(rnd() & mask);
    }
    if (avm_gpu_var_set_planes(ctx, var, src, PLANE_W, PLANE_H, ref, PLANE_W,
                               PLANE_H, err) != 0) {
      fprintf(stderr, "avm_gpu var planes: %s\n", err);
      return 1;
    }

    for (size_t c = 0; c < sizeof(kCases) / sizeof(kCases[0]); c++) {
      const case_t *tc = &kCases[c];
      const int max_x = PLANE_W - tc->bw;
      const int max_y = PLANE_H - tc->bh;
      for (int i = 0; i < BLOCKS_PER_CASE; i++) {
        pos[i].src_off =
            (uint32_t)((rnd() % max_y) * PLANE_W + (rnd() % max_x));
        pos[i].ref_off =
            (uint32_t)((rnd() % max_y) * PLANE_W + (rnd() % max_x));
      }

      uint32_t cpu_sse[BLOCKS_PER_CASE];
      double t0 = now_ms();
      for (int i = 0; i < BLOCKS_PER_CASE; i++) {
        cpu_var[i] = tc->ref[b](src + pos[i].src_off, PLANE_W,
                                ref + pos[i].ref_off, PLANE_W, &cpu_sse[i]);
      }
      const double cpu_ms = now_ms() - t0;

      t0 = now_ms();
      if (avm_gpu_var_run(ctx, var, pos, BLOCKS_PER_CASE, tc->bw, tc->bh, bd,
                          gpu_var, gpu_sse, err) != 0) {
        fprintf(stderr, "avm_gpu var run: %s\n", err);
        return 1;
      }
      const double gpu_ms = now_ms() - t0;

      int diff = 0;
      for (int i = 0; i < BLOCKS_PER_CASE; i++) {
        if (gpu_var[i] != cpu_var[i] || gpu_sse[i] != cpu_sse[i]) {
          if (diff < 3) {
            printf("  %dbd %s block %d: var %u/%u sse %u/%u\n", bd, tc->name, i,
                   gpu_var[i], cpu_var[i], gpu_sse[i], cpu_sse[i]);
          }
          diff++;
        }
      }
      failures += diff != 0;
      printf("%-3d %-7s %8d %10.3f %10.3f %s\n", bd, tc->name, BLOCKS_PER_CASE,
             cpu_ms, gpu_ms, diff ? "MISMATCH" : "ok");
    }
  }

  free(src);
  free(ref);
  avm_gpu_var_destroy(ctx, var);
  avm_gpu_ctx_destroy(ctx);

  printf("\n%s\n", failures ? "FAILED" : "ALL PARITY CASES PASSED");
  return failures ? 1 : 0;
}
