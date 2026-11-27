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
 * Parity gate for the Vulkan forward transform kernels against
 * fwd_txfm_c(): every 1D kind (DCT2 folded butterflies, identity scaling and
 * the DST7 / DCT8 / DDTX / FDDT matrices) and every size the encoder uses.
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

/* Forward 2D transform (fwd_txfm_c(), DCT_DCT) against the kernel across every
 * size the encoder uses, including the 64-length axes that are box-downsampled
 * by two. */
/* avm_gpu_fwd_kind()/avm_gpu_fwd_kernel() in av2_fwd_txfm(), recomputed here so
 * the gate does not share the seam's tables. */
static int fwd_kind(TX1D_TYPE t) {
  return t == DCT2 ? 0 : (t == IDT ? 1 : 2);
}

static int fwd_idt_scale(int n) {
  return n == 4 ? 128 : (n == 8 ? 181 : (n == 16 ? 256 : 362));
}

static const int *fwd_kernel(TX1D_TYPE t, int n) {
  if (t == IDT) return NULL;
  if (t == DCT2) {  /* the folded butterflies read the DCT2 rows */
    return n == 4    ? &tx_kernel_dct2_size4[FWD_TXFM][0][0]
           : n == 8  ? &tx_kernel_dct2_size8[FWD_TXFM][0][0]
           : n == 16 ? &tx_kernel_dct2_size16[FWD_TXFM][0][0]
                     : &tx_kernel_dct2_size32[FWD_TXFM][0][0];
  }
  if (t == DST7) {
    return n == 4    ? &tx_kernel_adst_size4[FWD_TXFM][0][0]
           : n == 8  ? &tx_kernel_adst_size8[FWD_TXFM][0][0]
           : n == 16 ? &tx_kernel_adst_size16[FWD_TXFM][0][0]
                     : NULL;
  }
  if (t == DCT8) {
    return n == 4    ? &tx_kernel_fdst_size4[FWD_TXFM][0][0]
           : n == 8  ? &tx_kernel_fdst_size8[FWD_TXFM][0][0]
           : n == 16 ? &tx_kernel_fdst_size16[FWD_TXFM][0][0]
                     : NULL;
  }
  if (t == DDTX || t == FDDT) {
    return n == 4    ? &tx_kernel_ddtx_size4[FWD_TXFM][0][0]
           : n == 8  ? &tx_kernel_ddtx_size8[FWD_TXFM][0][0]
           : n == 16 ? &tx_kernel_ddtx_size16[FWD_TXFM][0][0]
                     : NULL;
  }
  return NULL;
}

static int check_fwd(avm_gpu_ctx *ctx, avm_gpu_txfm *txfm, TX_SIZE tx, int bd,
                     PRIMARY_TX_TYPE type, int use_ddt, int *failures) {
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
  param.primary_tx_type = type;
  param.use_ddt = use_ddt;
  param.bd = bd;
  const int w1 = tw > 32 ? 32 : tw;
  const int h1 = th > 32 ? 32 : th;
  TX1D_TYPE row = g_hor_tx_type[type];
  TX1D_TYPE col = g_ver_tx_type[type];
  if (use_ddt) {
    if (((tw == 4 && REPLACE_ADST4) || (tw == 8 && REPLACE_ADST8) ||
         (tw == 16 && REPLACE_ADST16)) &&
        (row == DST7 || row == DCT8)) {
      row = row == DST7 ? DDTX : FDDT;
    }
    if (((th == 4 && REPLACE_ADST4) || (th == 8 && REPLACE_ADST8) ||
         (th == 16 && REPLACE_ADST16)) &&
        (col == DST7 || col == DCT8)) {
      col = col == DST7 ? DDTX : FDDT;
    }
  }
  const int kind_w = fwd_kind(row), kind_h = fwd_kind(col);
  const int *mat_w = fwd_kernel(row, w1);
  const int *mat_h = fwd_kernel(col, h1);
  if ((kind_w == 2 && !mat_w) || (kind_h == 2 && !mat_h)) {
    /* The reference has no kernel there (it asserts), so it never encodes. */
    free(resi);
    free(cpu);
    free(gpu);
    return 0;
  }

  const double t0 = now_ms();
  fwd_txfm_c(resi, cpu, stride, &param);
  const double cpu_ms = now_ms() - t0;

  const avm_gpu_txfm_pos pos = { 0, 0 };
  const double t1 = now_ms();
  if (avm_gpu_txfm_fwd(ctx, txfm, resi, stride, &pos, 1, tw, th,
                       fwd_tx_shift[tx][0], fwd_tx_shift[tx][1], kind_w,
                       kind_h, kind_w == 1 ? fwd_idt_scale(w1) : 0,
                       kind_h == 1 ? fwd_idt_scale(h1) : 0, row == FDDT,
                       col == FDDT, mat_w, mat_h, gpu, err) != 0) {
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
  printf("%-6s %-8s %8s %9s %9s %s\n", "fwd", "block", "samples", "cpu ms",
         "gpu ms", "parity");
  int failures = 0;

  /* fwd_txfm_c(): every size the encoder uses, DCT_DCT. */
  {
    static const TX_SIZE kFwd[] = {
      TX_4X4,   TX_8X8,   TX_16X16, TX_32X32, TX_64X64, TX_4X8,
      TX_8X4,   TX_8X16,  TX_16X8,  TX_16X32, TX_32X16, TX_32X64,
      TX_64X32, TX_16X64, TX_64X16, TX_4X16,  TX_16X4,
    };
    for (int bd = 10; bd <= 12; bd += 2) {
      for (size_t i = 0; i < sizeof(kFwd) / sizeof(kFwd[0]); i++) {
        if (check_fwd(ctx, txfm, kFwd[i], bd, DCT_DCT, 0, &failures) != 0) {
          return 1;
        }
      }
    }
  }

  /* fwd_txfm_c(): every primary transform type, 1D kernel and size the encoder
   * can ask for - DCT2 folded butterflies, the identity scaling and the
   * DST7 / DCT8 / DDTX / FDDT matrices, with and without the DDT
   * substitution. */
  {
    static const TX_SIZE kTypeSizes[] = {
      TX_4X4, TX_8X8, TX_16X16, TX_32X32, TX_4X8, TX_8X4, TX_16X8, TX_8X16,
      TX_16X32, TX_32X16, TX_4X16, TX_16X4, TX_64X64, TX_32X64, TX_64X32,
    };
    static const int kDt[3][2] = { { 4, 4 }, { 8, 8 }, { 16, 16 } };
    (void)kDt;
    int type_failures = 0;
    for (int t = 0; t < PRIMARY_TX_TYPES; t++) {
      for (int ddt = 0; ddt <= 1; ddt++) {
        const int before = failures;
        for (size_t i = 0; i < sizeof(kTypeSizes) / sizeof(kTypeSizes[0]); i++) {
          if (check_fwd(ctx, txfm, kTypeSizes[i], 10, (PRIMARY_TX_TYPE)t, ddt,
                        &failures) != 0) {
            return 1;
          }
        }
        type_failures += failures - before;
      }
    }
    printf("fwd types (16 primary types x 15 sizes x ddt on/off): %d failures\n",
           type_failures);
  }

  avm_gpu_txfm_destroy(ctx, txfm);
  avm_gpu_ctx_destroy(ctx);
  printf("\n%s\n", failures ? "FAILED" : "ALL PARITY CASES PASSED");
  return failures ? 1 : 0;
}
