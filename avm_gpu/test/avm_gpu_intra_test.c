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
 * Parity gate for the Vulkan intra predictors against
 * avm_highbd_{dc,v,h,dc_left,dc_top,dc_128}_predictor_WxH_c, plus checks for
 * the availability rule and the frame-clamped edge builder.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "config/avm_dsp_rtcd.h"

#include "avm_gpu/vulkan_gpu.h"
#include "avm_gpu/vulkan_intra.h"

#define STRIDE 80
#define BD 10

typedef void (*pred_fn)(uint16_t *, ptrdiff_t, const uint16_t *,
                        const uint16_t *, int);

typedef struct {
  const char *name;
  int mode;
  pred_fn ref[5]; /* 4x4, 8x8, 16x16, 32x32, 64x64 */
} pred_case;

#define PCASE(NAME, fn)                                                       \
  {                                                                           \
    #NAME, AVM_GPU_INTRA_##NAME, {                                             \
      avm_highbd_##fn##_predictor_4x4_c, avm_highbd_##fn##_predictor_8x8_c,    \
          avm_highbd_##fn##_predictor_16x16_c,                                 \
          avm_highbd_##fn##_predictor_32x32_c,                                 \
          avm_highbd_##fn##_predictor_64x64_c                                  \
    }                                                                         \
  }

static const pred_case kModes[] = {
  PCASE(DC, dc),         PCASE(V, v),           PCASE(H, h),
  PCASE(DC_LEFT, dc_left), PCASE(DC_TOP, dc_top), PCASE(DC_128, dc_128)
};
static const int kSizes[5] = { 4, 8, 16, 32, 64 };

static unsigned int rng_state = 0x51ed270bu;
static unsigned int rnd(void) {
  rng_state = rng_state * 1664525u + 1013904223u;
  return rng_state >> 8;
}

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int check_availability_rules(void) {
  struct {
    int above, left, want;
  } cases[] = {
    { 0, 0, AVM_GPU_INTRA_DC_128 },
    { 0, 1, AVM_GPU_INTRA_DC_LEFT },
    { 1, 0, AVM_GPU_INTRA_DC_TOP },
    { 1, 1, AVM_GPU_INTRA_DC },
  };
  int bad = 0;
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
    const int got = avm_gpu_intra_auto_mode(cases[i].above, cases[i].left);
    if (got != cases[i].want) {
      printf("  availability(%d,%d) = %d, want %d\n", cases[i].above,
             cases[i].left, got, cases[i].want);
      bad++;
    }
  }
  printf("%-8s %-28s %s\n", "rule", "has_above/has_left -> mode",
         bad ? "MISMATCH" : "ok");
  return bad;
}

static int check_edges(const uint16_t *plane, int w, int h) {
  /* Bottom-right block: the edge builder must clamp to the frame border. */
  const int bw = 8, bh = 8;
  const int x = w - 4, y = h - 4;
  uint16_t above[8], left[8];
  int bad = 0;

  avm_gpu_intra_edges(plane, STRIDE, x, y, bw, bh, w, h, BD, 1, 1, above, left);
  for (int i = 0; i < bw; i++) {
    int px = x + i;
    if (px > w - 1) px = w - 1;
    if (above[i] != plane[(y - 1) * STRIDE + px]) bad++;
  }
  for (int j = 0; j < bh; j++) {
    int py = y + j;
    if (py > h - 1) py = h - 1;
    if (left[j] != plane[py * STRIDE + (x - 1)]) bad++;
  }

  /* Unavailable sides must carry the neutral value. */
  uint16_t a2[8], l2[8];
  avm_gpu_intra_edges(plane, STRIDE, x, y, bw, bh, w, h, BD, 0, 0, a2, l2);
  for (int i = 0; i < bw; i++) if (a2[i] != (1 << (BD - 1))) bad++;
  for (int j = 0; j < bh; j++) if (l2[j] != (1 << (BD - 1))) bad++;

  printf("%-8s %-28s %s\n", "edges", "clamp + neutral fill",
         bad ? "MISMATCH" : "ok");
  return bad;
}

int main(void) {
  char err[AVM_GPU_ERR_LEN];
  avm_gpu_ctx *ctx = NULL;
  if (avm_gpu_ctx_create(&ctx, err) != 0) {
    fprintf(stderr, "avm_gpu: %s\n", err);
    return 77; /* skip: no Vulkan device */
  }
  avm_gpu_intra *intra = NULL;
  if (avm_gpu_intra_create(ctx, &intra, err) != 0) {
    fprintf(stderr, "avm_gpu intra: %s\n", err);
    avm_gpu_ctx_destroy(ctx);
    return 1;
  }
  printf("device: %s\n", avm_gpu_device_name(ctx));
  printf("%-8s %-6s %10s %10s %s\n", "mode", "block", "cpu ms", "gpu ms",
         "parity");

  int failures = 0;
  failures += check_availability_rules();

  const int frame_w = 200, frame_h = 200;
  uint16_t *plane = malloc(sizeof(uint16_t) * STRIDE * frame_h);
  if (!plane) return 1;
  for (int y = 0; y < frame_h; y++) {
    for (int x = 0; x < STRIDE; x++) plane[y * STRIDE + x] = (uint16_t)(rnd() & 0x3ff);
  }
  failures += check_edges(plane, frame_w, frame_h);

  uint16_t above[66], left[66];
  uint16_t cpu[64 * STRIDE], gpu[64 * STRIDE];

  for (size_t m = 0; m < sizeof(kModes) / sizeof(kModes[0]); m++) {
    for (size_t s = 0; s < 5; s++) {
      const int n = kSizes[s];
      for (int i = 0; i < n; i++) {
        above[i] = (uint16_t)(rnd() & 0x3ff);
        left[i] = (uint16_t)(rnd() & 0x3ff);
      }

      memset(cpu, 0, sizeof(cpu));
      const double t0 = now_ms();
      kModes[m].ref[s](cpu, STRIDE, above, left, BD);
      const double cpu_ms = now_ms() - t0;

      const double t1 = now_ms();
      if (avm_gpu_intra_predict(ctx, intra, above, left, n, n, kModes[m].mode,
                                BD, gpu, STRIDE, err) != 0) {
        fprintf(stderr, "avm_gpu intra: %s\n", err);
        return 1;
      }
      const double gpu_ms = now_ms() - t1;

      int diff = 0;
      for (int y = 0; y < n; y++) {
        for (int x = 0; x < n; x++) {
          if (cpu[y * STRIDE + x] != gpu[y * STRIDE + x]) {
            if (diff < 3) {
              printf("  %s %dx%d (%d,%d): gpu %u != cpu %u\n", kModes[m].name, n,
                     n, x, y, gpu[y * STRIDE + x], cpu[y * STRIDE + x]);
            }
            diff++;
          }
        }
      }
      failures += diff != 0;
      printf("%-8s %-6d %10.3f %10.3f %s\n", kModes[m].name, n, cpu_ms, gpu_ms,
             diff ? "MISMATCH" : "ok");
    }
  }

  free(plane);
  avm_gpu_intra_destroy(ctx, intra);
  avm_gpu_ctx_destroy(ctx);
  printf("\n%s\n", failures ? "FAILED" : "ALL PARITY CASES PASSED");
  return failures ? 1 : 0;
}
