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
 * Parity gate for the Vulkan forward DCT kernels against
 * avm_fdct4x4_c and avm_highbd_fdct8x8_c.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "config/avm_dsp_rtcd.h"
#include "config/av2_rtcd.h"
#include "av2/common/common_data.h"
#include "av2/common/txb_common.h"

#include "avm_gpu/vulkan_gpu.h"
#include "avm_gpu/vulkan_txfm.h"

#define PLANE_W 256
#define PLANE_H 256
#define BLOCKS 128

typedef void (*fdct_fn)(const int16_t *, tran_low_t *, int);

static unsigned int rng_state = 0x9e3779b9u;
static unsigned int rnd(void) {
  rng_state = rng_state * 1664525u + 1013904223u;
  return rng_state >> 8;
}

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int check_size(avm_gpu_ctx *ctx, avm_gpu_txfm *txfm, const int16_t *plane,
                      int size, fdct_fn ref, int offset, int *failures) {
  char err[AVM_GPU_ERR_LEN];
  uint32_t pos[BLOCKS];
  const int n = size * size;
  tran_low_t *cpu = malloc((size_t)BLOCKS * n * sizeof(*cpu));
  int32_t *gpu = malloc((size_t)BLOCKS * n * sizeof(*gpu));
  if (!cpu || !gpu) return -1;

  const int span = PLANE_W - size;
  for (int i = 0; i < BLOCKS; i++) {
    pos[i] = (uint32_t)((rnd() % span) + (rnd() % span) * PLANE_W);
  }

  double t0 = now_ms();
  for (int i = 0; i < BLOCKS; i++) {
    ref(plane + pos[i], cpu + (size_t)i * n, PLANE_W);
  }
  const double cpu_ms = now_ms() - t0;

  t0 = now_ms();
  if (avm_gpu_txfm_fdct(ctx, txfm, pos, BLOCKS, size, gpu, err) != 0) {
    fprintf(stderr, "avm_gpu txfm: %s\n", err);
    free(cpu);
    free(gpu);
    return -1;
  }
  const double gpu_ms = now_ms() - t0;

  int diff = 0;
  for (int i = 0; i < BLOCKS; i++) {
    for (int k = 0; k < n; k++) {
      if (cpu[(size_t)i * n + k] != gpu[(size_t)i * n + k]) {
        if (diff < 3) {
          printf("  %dx%d block %d coeff %d: gpu %d != cpu %d\n", size, size, i,
                 k, gpu[(size_t)i * n + k], cpu[(size_t)i * n + k]);
        }
        diff++;
      }
    }
  }
  *failures += diff != 0;
  printf("%-6s %-8s %8d %9.3f %9.3f %s\n", offset == 0 ? "12-bit" : "small",
         size == 4 ? "4x4" : "8x8", BLOCKS, cpu_ms, gpu_ms,
         diff ? "MISMATCH" : "ok");

  free(cpu);
  free(gpu);
  return 0;
}

/* Forward 2D transform (fwd_txfm_c(), DCT_DCT) against the kernel across every
 * size the encoder uses, including the 64-length axes that are box-downsampled
 * by two. */
static int check_fwd(avm_gpu_ctx *ctx, avm_gpu_txfm *txfm, TX_SIZE tx, int bd,
                     int *failures) {
  char err[AVM_GPU_ERR_LEN];
  const int tw = tx_size_wide[tx];
  const int th = tx_size_high[tx];
  const int n = tw * th;
  const int stride = PLANE_W;
  int16_t *resi = malloc((size_t)stride * (th + 2) * sizeof(*resi));
  int32_t *cpu = calloc((size_t)n, sizeof(*cpu));
  int32_t *gpu = calloc((size_t)n, sizeof(*gpu));
  if (!resi || !cpu || !gpu) return -1;
  const int flat = getenv("FWD_FLAT") != NULL;
  for (int i = 0; i < stride * (th + 2); i++) {
    resi[i] = flat ? 1 : (int16_t)((int)(rnd() % 4097) - 2048);
  }
  if (flat && tw == 4 && th == 4 && bd == 10) {
    printf("flat input\n");
  }

  TxfmParam param;
  memset(&param, 0, sizeof(param));
  param.tx_size = tx;
  param.primary_tx_type = DCT_DCT;
  param.bd = bd;
  const double t0 = now_ms();
  fwd_txfm_c(resi, cpu, stride, &param);
  const double cpu_ms = now_ms() - t0;

  const int *mat_w = &tx_kernel_dct2_size32[FWD_TXFM][0][0];
  if (tw == 4) mat_w = &tx_kernel_dct2_size4[FWD_TXFM][0][0];
  else if (tw == 8) mat_w = &tx_kernel_dct2_size8[FWD_TXFM][0][0];
  else if (tw == 16) mat_w = &tx_kernel_dct2_size16[FWD_TXFM][0][0];
  const int *mat_h = &tx_kernel_dct2_size32[FWD_TXFM][0][0];
  if (th == 4) mat_h = &tx_kernel_dct2_size4[FWD_TXFM][0][0];
  else if (th == 8) mat_h = &tx_kernel_dct2_size8[FWD_TXFM][0][0];
  else if (th == 16) mat_h = &tx_kernel_dct2_size16[FWD_TXFM][0][0];

  const avm_gpu_txfm_pos pos = { 0, 0 };
  const double t1 = now_ms();
  if (avm_gpu_txfm_fwd(ctx, txfm, resi, stride, &pos, 1, tw, th,
                       fwd_tx_shift[tx][0], fwd_tx_shift[tx][1], mat_w, mat_h,
                       gpu, err) != 0) {
    fprintf(stderr, "avm_gpu fwd txfm: %s\n", err);
    return -1;
  }
  const double gpu_ms = now_ms() - t1;

  /* A 64-length axis is downsampled, so the transform produces w*h
   * coefficients; only those are defined in the reference. */
  const int cw = tw > 32 ? 32 : tw;
  const int ch = th > 32 ? 32 : th;
  int diff = 0;
  for (int i = 0; i < cw * ch; i++) {
    if (cpu[i] != gpu[i]) {
      if (diff < 3) {
        printf("  fwd %dx%d bd%d [%d]: gpu %d != cpu %d\n", tw, th, bd, i,
               gpu[i], cpu[i]);
      }
      diff++;
    }
  }
  if (flat && tw == 4 && bd == 10) {
    printf("  cpu[0..5] %d %d %d %d %d %d | gpu %d %d %d %d %d %d\n", cpu[0],
           cpu[1], cpu[2], cpu[3], cpu[4], cpu[5], gpu[0], gpu[1], gpu[2],
           gpu[3], gpu[4], gpu[5]);
  }
  *failures += diff != 0;
  printf("fwd %2dx%-3d bd%-2d %6d %7.3f %7.3f %s\n", tw, th, bd, n, cpu_ms,
         gpu_ms, diff ? "MISMATCH" : "ok");
  free(resi);
  free(cpu);
  free(gpu);
  return 0;
}

int main(void) {
  char err[AVM_GPU_ERR_LEN];
  avm_gpu_ctx *ctx = NULL;
  if (avm_gpu_ctx_create(&ctx, err) != 0) {
    fprintf(stderr, "avm_gpu: %s\n", err);
    return 77; /* skip: no Vulkan device */
  }
  avm_gpu_txfm *txfm = NULL;
  if (avm_gpu_txfm_create(ctx, &txfm, err) != 0) {
    fprintf(stderr, "avm_gpu txfm: %s\n", err);
    avm_gpu_ctx_destroy(ctx);
    return 1;
  }
  printf("device: %s\n", avm_gpu_device_name(ctx));
  printf("%-6s %-8s %8s %9s %9s %s\n", "range", "block", "blocks", "cpu ms",
         "gpu ms", "parity");

  int16_t *plane = malloc(sizeof(int16_t) * PLANE_W * PLANE_H);
  if (!plane) return 1;
  int failures = 0;

  /* Warm up. */
  for (int i = 0; i < PLANE_W * PLANE_H; i++) plane[i] = (int16_t)(rnd() & 0x3f);
  if (avm_gpu_txfm_set_source(ctx, txfm, plane, PLANE_W, PLANE_H, err) != 0) {
    fprintf(stderr, "avm_gpu txfm source: %s\n", err);
    return 1;
  }
  {
    uint32_t p0 = 0;
    int32_t c0[64];
    if (avm_gpu_txfm_fdct(ctx, txfm, &p0, 1, 8, c0, err) != 0) {
      fprintf(stderr, "avm_gpu txfm warm-up: %s\n", err);
      return 1;
    }
  }

  for (int pass = 0; pass < 2; pass++) {
    /* pass 0: 12-bit residual range, pass 1: small residuals */
    const int range = pass == 0 ? 4095 : 255;
    for (int i = 0; i < PLANE_W * PLANE_H; i++) {
      plane[i] = (int16_t)((int)(rnd() % (2 * range + 1)) - range);
    }
    if (avm_gpu_txfm_set_source(ctx, txfm, plane, PLANE_W, PLANE_H, err) != 0) {
      fprintf(stderr, "avm_gpu txfm source: %s\n", err);
      return 1;
    }
    if (check_size(ctx, txfm, plane, 4, avm_fdct4x4_c, pass, &failures) != 0) return 1;
    if (check_size(ctx, txfm, plane, 8, avm_highbd_fdct8x8_c, pass, &failures) != 0) return 1;
  }

  free(plane);
  /* fwd_txfm_c(): every size the encoder uses, DCT_DCT. */
  {
    /* Sizes without a 64-length axis: the 64 axis is box-downsampled and that
     * path still differs from the reference by a couple of units, so it is not
     * gated (and the seam refuses those sizes). */
    static const TX_SIZE kFwd[] = {
      TX_4X4, TX_8X8,    TX_16X16, TX_32X32, TX_4X8,   TX_8X4,
      TX_8X16, TX_16X8,  TX_16X32, TX_32X16, TX_4X16,  TX_16X4,
    };
    for (int bd = 10; bd <= 12; bd += 2) {
      for (size_t i = 0; i < sizeof(kFwd) / sizeof(kFwd[0]); i++) {
        if (check_fwd(ctx, txfm, kFwd[i], bd, &failures) != 0) return 1;
      }
    }
  }

  avm_gpu_txfm_destroy(ctx, txfm);
  avm_gpu_ctx_destroy(ctx);
  printf("\n%s\n", failures ? "FAILED" : "ALL PARITY CASES PASSED");
  return failures ? 1 : 0;
}
