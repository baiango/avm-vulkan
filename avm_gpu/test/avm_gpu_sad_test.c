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
 * Parity gate for the Vulkan SAD kernel: every block priced on the GPU must be
 * bit-identical to AVM's own avm_highbd_sadWxH_c reference.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "config/avm_dsp_rtcd.h"

#include "avm_gpu/vulkan_gpu.h"
#include "avm_gpu/vulkan_sad.h"

#define PLANE_W 512
#define PLANE_H 512
#define BLOCKS_PER_CASE 128
#define RAND_MAX_VAL 4095

typedef unsigned int (*sad_fn)(const uint16_t *, int, const uint16_t *, int);

typedef struct {
  const char *name;
  int bw, bh;
  sad_fn ref;
} case_t;

#define CASE(w, h) { #w "x" #h, w, h, avm_highbd_sad##w##x##h##_c }

static const case_t kCases[] = {
  CASE(4, 4),   CASE(4, 8),   CASE(8, 8),   CASE(8, 16),
  CASE(16, 16), CASE(16, 32), CASE(32, 32), CASE(32, 64),
  CASE(64, 64),
};

static unsigned int rng_state = 0x12345678u;
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
  avm_gpu_sad *sad = NULL;
  if (avm_gpu_sad_create(ctx, &sad, err) != 0) {
    fprintf(stderr, "avm_gpu sad: %s\n", err);
    avm_gpu_ctx_destroy(ctx);
    return 1;
  }
  printf("device: %s\n", avm_gpu_device_name(ctx));
  printf("%-8s %10s %10s %10s %s\n", "block", "blocks", "cpu ms", "gpu ms",
         "parity");

  uint16_t *src = malloc(sizeof(uint16_t) * PLANE_W * PLANE_H);
  uint16_t *ref = malloc(sizeof(uint16_t) * PLANE_W * PLANE_H);
  if (!src || !ref) return 1;
  for (int i = 0; i < PLANE_W * PLANE_H; i++) {
    src[i] = (uint16_t)(rnd() % (RAND_MAX_VAL + 1));
    ref[i] = (uint16_t)(rnd() % (RAND_MAX_VAL + 1));
  }

  if (avm_gpu_sad_set_planes(ctx, sad, src, PLANE_W, PLANE_H, ref, PLANE_W,
                             PLANE_H, err) != 0) {
    fprintf(stderr, "avm_gpu sad planes: %s\n", err);
    return 1;
  }

  avm_gpu_sad_pos pos[BLOCKS_PER_CASE];
  uint32_t gpu_out[BLOCKS_PER_CASE];
  unsigned int cpu_out[BLOCKS_PER_CASE];
  int failures = 0;

  /* Warm the pipeline so the first timed case is not a page-fault charge. */
  {
    avm_gpu_sad_pos w0 = { 0, 0 };
    uint32_t wcost = 0;
    if (avm_gpu_sad_run(ctx, sad, &w0, 1, 64, 64, &wcost, err) != 0) {
      fprintf(stderr, "avm_gpu sad warm-up: %s\n", err);
      return 1;
    }
  }

  for (size_t c = 0; c < sizeof(kCases) / sizeof(kCases[0]); c++) {
    const case_t *tc = &kCases[c];
    const int max_x = PLANE_W - tc->bw;
    const int max_y = PLANE_H - tc->bh;
    for (int i = 0; i < BLOCKS_PER_CASE; i++) {
      pos[i].src_off = (uint32_t)((rnd() % max_y) * PLANE_W + (rnd() % max_x));
      pos[i].ref_off = (uint32_t)((rnd() % max_y) * PLANE_W + (rnd() % max_x));
    }

    double t0 = now_ms();
    for (int i = 0; i < BLOCKS_PER_CASE; i++) {
      cpu_out[i] = tc->ref(src + pos[i].src_off, PLANE_W, ref + pos[i].ref_off,
                           PLANE_W);
    }
    double cpu_ms = now_ms() - t0;

    t0 = now_ms();
    if (avm_gpu_sad_run(ctx, sad, pos, BLOCKS_PER_CASE, tc->bw, tc->bh,
                        gpu_out, err) != 0) {
      fprintf(stderr, "avm_gpu sad run: %s\n", err);
      return 1;
    }
    double gpu_ms = now_ms() - t0;

    int diff = 0;
    for (int i = 0; i < BLOCKS_PER_CASE; i++) {
      if (gpu_out[i] != cpu_out[i]) {
        if (diff < 3) {
          printf("  %s block %d: gpu %u != cpu %u\n", tc->name, i, gpu_out[i],
                 cpu_out[i]);
        }
        diff++;
      }
    }
    failures += diff != 0;
    printf("%-8s %10d %10.3f %10.3f %s\n", tc->name, BLOCKS_PER_CASE, cpu_ms,
           gpu_ms, diff ? "MISMATCH" : "ok");
  }

  free(src);
  free(ref);
  avm_gpu_sad_destroy(ctx, sad);
  avm_gpu_ctx_destroy(ctx);

  printf("\n%s\n", failures ? "FAILED" : "ALL PARITY CASES PASSED");
  return failures ? 1 : 0;
}
