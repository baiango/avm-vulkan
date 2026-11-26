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
 * Parity gate for the Vulkan inverse DCT against inv_txfm_c()
 * (av2/common/idct.c), DCT_DCT, square and rectangular sizes.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "avm_dsp/txfm_common.h"
#include "av2/common/enums.h"
#include "av2/common/idct.h"

#include "avm_gpu/vulkan_gpu.h"
#include "avm_gpu/vulkan_itxfm.h"

extern const int tx_kernel_dct2_size4[4][4][4];
extern const int tx_kernel_dct2_size8[4][8][8];
extern const int tx_kernel_dct2_size16[4][16][16];
extern const int tx_kernel_dct2_size32[4][32][32];

#define W 128
#define H 128
#define STRIDE 128
#define BLOCKS 64
#define BD 10

static unsigned int rng_state = 0x1d872b41u;
static unsigned int rnd(void) {
  rng_state = rng_state * 1664525u + 1013904223u;
  return rng_state >> 8;
}

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static const int *matrix_for(int n) {
  switch (n) {
    case 4: return &tx_kernel_dct2_size4[0][0][0];
    case 8: return &tx_kernel_dct2_size8[0][0][0];
    case 16: return &tx_kernel_dct2_size16[0][0][0];
    default: return &tx_kernel_dct2_size32[0][0][0];
  }
}

static int check_block(avm_gpu_ctx *ctx, avm_gpu_itxfm *it, int w, int h,
                       TX_SIZE tx_size, int *failures) {
  char err[AVM_GPU_ERR_LEN];
  const int n = w * h;
  uint16_t *ref = malloc(sizeof(uint16_t) * STRIDE * H);
  uint16_t *gpu = malloc(sizeof(uint16_t) * STRIDE * H);
  int32_t *coeffs = malloc((size_t)BLOCKS * n * sizeof(*coeffs));
  uint32_t *pos = malloc(BLOCKS * sizeof(*pos));
  int *slots = NULL;
  if (!ref || !gpu || !coeffs || !pos) return -1;

  for (int y = 0; y < H; y++) {
    for (int x = 0; x < STRIDE; x++) {
      ref[y * STRIDE + x] = (uint16_t)(rnd() & 0x3ff);
    }
  }
  memcpy(gpu, ref, sizeof(uint16_t) * STRIDE * H);

  /* Blocks must not overlap: inv_txfm_c adds into the prediction in call
   * order, so a later block covering the same pixels depends on the earlier
   * one, which a single parallel dispatch cannot reproduce. */
  const int gx = W / w, gy = H / h;
  const int blocks = BLOCKS < gx * gy ? BLOCKS : gx * gy;
  slots = malloc((size_t)gx * gy * sizeof(*slots));
  if (!slots) return -1;
  for (int i = 0; i < gx * gy; i++) slots[i] = i;
  for (int i = 0; i < blocks; i++) {
    const int pick = i + (int)(rnd() % (unsigned)(gx * gy - i));
    const int t = slots[i];
    slots[i] = slots[pick];
    slots[pick] = t;
    const int x = (slots[i] % gx) * w;
    const int y = (slots[i] / gx) * h;
    pos[i] = (uint32_t)(y * STRIDE + x);
    for (int k = 0; k < n; k++) {
      coeffs[(size_t)i * n + k] = (int32_t)(rnd() % 8193) - 4096;
    }
  }

  TxfmParam param;
  memset(&param, 0, sizeof(param));
  param.tx_size = tx_size;
  param.primary_tx_type = DCT_DCT;
  param.bd = BD;
  param.eob = n;
  param.lossless = 0;
  param.use_ddt = 0;
  param.is_inter = 0;

  const double t0 = now_ms();
  for (int i = 0; i < blocks; i++) {
    inv_txfm_c(coeffs + (size_t)i * n, ref + pos[i], STRIDE, &param);
  }
  const double cpu_ms = now_ms() - t0;

  const double t1 = now_ms();
  if (avm_gpu_itxfm_add_rect(ctx, it, coeffs, pos, blocks, w, h, BD,
                             matrix_for(w), matrix_for(h), gpu, STRIDE, H,
                             err) != 0) {
    fprintf(stderr, "avm_gpu itxfm: %s\n", err);
    free(ref);
    free(gpu);
    free(coeffs);
    free(pos);
    free(slots);
    return -1;
  }
  const double gpu_ms = now_ms() - t1;

  int diff = 0;
  for (int y = 0; y < H; y++) {
    for (int x = 0; x < STRIDE; x++) {
      if (ref[y * STRIDE + x] != gpu[y * STRIDE + x]) {
        if (diff < 3) {
          printf("  %dx%d (%d,%d): gpu %u != cpu %u\n", w, h, x, y,
                 gpu[y * STRIDE + x], ref[y * STRIDE + x]);
        }
        diff++;
      }
    }
  }
  *failures += diff != 0;
  printf("%2dx%-3d %8d %9.3f %9.3f %s\n", w, h, blocks, cpu_ms, gpu_ms,
         diff ? "MISMATCH" : "ok");

  free(ref);
  free(gpu);
  free(coeffs);
  free(pos);
  free(slots);
  return 0;
}

int main(void) {
  static const struct {
    int w, h;
    TX_SIZE tx;
  } kSizes[] = {
    { 4, 4, TX_4X4 },     { 8, 8, TX_8X8 },     { 16, 16, TX_16X16 },
    { 32, 32, TX_32X32 }, { 4, 8, TX_4X8 },     { 8, 4, TX_8X4 },
    { 8, 16, TX_8X16 },   { 16, 8, TX_16X8 },   { 4, 16, TX_4X16 },
    { 16, 4, TX_16X4 },   { 16, 32, TX_16X32 }, { 32, 16, TX_32X16 },
    { 4, 32, TX_4X32 },   { 32, 4, TX_32X4 },   { 8, 32, TX_8X32 },
    { 32, 8, TX_32X8 },
  };

  char err[AVM_GPU_ERR_LEN];
  avm_gpu_ctx *ctx = NULL;
  if (avm_gpu_ctx_create(&ctx, err) != 0) {
    fprintf(stderr, "avm_gpu: %s\n", err);
    return 77; /* skip: no Vulkan device */
  }
  avm_gpu_itxfm *it = NULL;
  if (avm_gpu_itxfm_create(ctx, &it, err) != 0) {
    fprintf(stderr, "avm_gpu itxfm: %s\n", err);
    avm_gpu_ctx_destroy(ctx);
    return 1;
  }
  printf("device: %s\n", avm_gpu_device_name(ctx));
  printf("%-5s %8s %9s %9s %s\n", "block", "blocks", "cpu ms", "gpu ms",
         "parity");

  int failures = 0;
  for (size_t i = 0; i < sizeof(kSizes) / sizeof(kSizes[0]); i++) {
    if (check_block(ctx, it, kSizes[i].w, kSizes[i].h, kSizes[i].tx,
                    &failures) != 0) {
      return 1;
    }
  }

  avm_gpu_itxfm_destroy(ctx, it);
  avm_gpu_ctx_destroy(ctx);
  printf("\n%s\n", failures ? "FAILED" : "ALL PARITY CASES PASSED");
  return failures ? 1 : 0;
}
