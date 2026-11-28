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
 * Gate for the encoder's variance seam (avm_gpu/vulkan_hook_var.c).
 *
 * The seam has two shortcuts that the kernel gate (avm_gpu_var_test) does not
 * see: a content-keyed memo shared across calls, and a block-area threshold
 * below which calls stay on the CPU. This test drives the seam the way the
 * encoder does - repeated block pairs, every block size - and compares the
 * (sse, sum) it returns with the reference loop from avm_dsp/variance.c,
 * including the memo's hit path.
 *
 * Exit 0 = every case exact, 77 = no Vulkan device, 1 = a mismatch.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "avm_gpu/vulkan_hook_var.h"

#define PLANE_W 192
#define PLANE_H 192
#define CASES 256

static uint16_t src[PLANE_W * PLANE_H];
static uint16_t ref[PLANE_W * PLANE_H];

static unsigned int rnd_state = 12345u;

static unsigned int rnd(void) {
  rnd_state = rnd_state * 1103515245u + 12345u;
  return (rnd_state >> 8);
}

/* avm_dsp/variance.c::highbd_variance64(), verbatim. */
static void ref_variance64(const uint16_t *a, int a_stride, const uint16_t *b,
                           int b_stride, int w, int h, uint64_t *sse,
                           int64_t *sum) {
  int64_t tsum = 0;
  uint64_t tsse = 0;
  for (int i = 0; i < h; ++i) {
    int32_t lsum = 0;
    for (int j = 0; j < w; ++j) {
      const int diff = a[j] - b[j];
      lsum += diff;
      tsse += (uint32_t)(diff * diff);
    }
    tsum += lsum;
    a += a_stride;
    b += b_stride;
  }
  *sum = tsum;
  *sse = tsse;
}

static const struct {
  int w, h;
} kSizes[] = {
  { 4, 4 },   { 4, 8 },   { 8, 4 },   { 8, 8 },   { 8, 16 },  { 16, 8 },
  { 16, 16 }, { 16, 32 }, { 32, 16 }, { 32, 32 }, { 32, 64 }, { 64, 32 },
  { 64, 64 },
};

/* The gates call AVM's own reference functions, several of which are RTCD
 * pointers (NULL on x86 until the tables are set up - arm64 resolves them to C
 * at link time, which is why MoltenVK never showed this). The encoder calls
 * these at init; the gates do the same. */
void av2_rtcd(void);
void avm_dsp_rtcd(void);

int main(void) {
  av2_rtcd();
  avm_dsp_rtcd();
  /* Every block goes to the GPU and the memo stays on, so both shortcuts run. */
  setenv("AV2_GPU_VAR", "1", 1);
  setenv("AV2_GPU_VAR_MIN_AREA", "0", 1);
  setenv("AV2_GPU_VAR_MEMO", "1", 1);
  unsetenv("AV2_GPU_VAR_STATS");

  for (int i = 0; i < PLANE_W * PLANE_H; i++) {
    src[i] = (uint16_t)(rnd() & 0x3ff);
    ref[i] = (uint16_t)(rnd() & 0x3ff);
  }

  printf("variance seam: %d calls, sizes 4x4..64x64, repeats included\n", CASES);

  int failures = 0;
  int handled = 0;
  int repeated = 0;
  int last_w = 0, last_h = 0;
  uint32_t last_src = 0, last_ref = 0;
  uint64_t last_sse = 0;
  int64_t last_sum = 0;

  for (int c = 0; c < CASES; c++) {
    const int s = c % (int)(sizeof(kSizes) / sizeof(kSizes[0]));
    int w = kSizes[s].w;
    int h = kSizes[s].h;
    /* Row/column offsets inside the plane, so a block never reads past it. */
    uint32_t so = (rnd() % (PLANE_H - h)) * PLANE_W + (rnd() % (PLANE_W - w));
    uint32_t ro = (rnd() % (PLANE_H - h)) * PLANE_W + (rnd() % (PLANE_W - w));

    /* Every fourth call repeats the previous block pair exactly, which is the
     * memo's hit path (44% of the encoder's calls hit it). */
    if (c > 0 && (c % 4) == 0) {
      w = last_w;
      h = last_h;
      so = last_src;
      ro = last_ref;
      repeated++;
    }

    const uint16_t *a = src + so;
    const uint16_t *b = ref + ro;
    uint64_t got_sse = 0, want_sse = 0;
    int64_t got_sum = 0, want_sum = 0;

    const int ok =
        avm_gpu_hook_variance64(a, PLANE_W, b, PLANE_W, w, h, &got_sse, &got_sum);
    if (ok) {
      handled++;
      if (repeated && last_sse && c % 4 == 0) {
        if (got_sse != last_sse || got_sum != last_sum) {
          printf("  %2dx%-3d memo hit returned (%llu, %lld), expected "
                 "(%llu, %lld)\n",
                 w, h, (unsigned long long)got_sse, (long long)got_sum,
                 (unsigned long long)last_sse, (long long)last_sum);
          failures++;
        }
      }
      ref_variance64(a, PLANE_W, b, PLANE_W, w, h, &want_sse, &want_sum);
      if (got_sse != want_sse || got_sum != want_sum) {
        printf("  %2dx%-3d gpu (%llu, %lld) != cpu (%llu, %lld)\n", w, h,
               (unsigned long long)got_sse, (long long)got_sum,
               (unsigned long long)want_sse, (long long)want_sum);
        failures++;
      }
      last_sse = got_sse;
      last_sum = got_sum;
      last_w = w;
      last_h = h;
      last_src = so;
      last_ref = ro;
    }
  }

  if (!handled) {
    printf("SKIP: no Vulkan device\n");
    return 77;
  }
  printf("handled %d/%d calls (%d repeated)\n", handled, CASES, repeated);
  printf("\n%s\n", failures ? "FAILED" : "ALL PARITY CASES PASSED");
  return failures ? 1 : 0;
}
