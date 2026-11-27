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

#include "avm_gpu/vulkan_hook_itxfm.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "av2/common/enums.h"
#include "av2/common/txb_common.h"
#include "avm_gpu/vulkan_gpu.h"
#include "avm_gpu/vulkan_itxfm.h"

#define ITX_MAX_1D 32

typedef struct itxfm_hook_state {
  int state; /* 0 = uninitialised, 1 = off, 2 = ready */
  avm_gpu_ctx *ctx;
  avm_gpu_itxfm *it;
  unsigned long blocks;
  unsigned long cpu_blocks;
  /* inv_txfm_idtx_sizeN_c() is a scaled identity, so the identity pass is a
   * matrix with the scale on the diagonal: 128 / 181 / 256 / 362. */
  int idtx[ITX_MAX_1D * ITX_MAX_1D];
  /* FDDT is the DDTX matrix with the columns reversed (inv_txfm_fddt_sizeN_c). */
  int fddt[16 * 16];
  int idtx_n;
} itxfm_hook_state;

static itxfm_hook_state g_hook;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static void print_stats(void) {
  if (getenv("AV2_GPU_ITXFM_STATS")) {
    fprintf(stderr,
            "avm_gpu: %lu inverse transforms on the GPU (%lu left to the C)\n",
            g_hook.blocks, g_hook.cpu_blocks);
  }
}

/* Caller holds g_lock. */
static int hook_ready_locked(void) {
  if (g_hook.state) return g_hook.state == 2;

  const char *env = getenv("AV2_GPU_ITXFM");
  if (!env || !*env || !strcmp(env, "0")) {
    g_hook.state = 1;
    return 0;
  }
  char err[AVM_GPU_ERR_LEN];
  if (avm_gpu_ctx_create(&g_hook.ctx, err) != 0 ||
      avm_gpu_itxfm_create(g_hook.ctx, &g_hook.it, err) != 0) {
    fprintf(stderr, "avm_gpu: inverse transform hook disabled (%s)\n", err);
    avm_gpu_ctx_destroy(g_hook.ctx);
    g_hook.ctx = NULL;
    g_hook.state = 1;
    return 0;
  }
  atexit(print_stats);
  fprintf(stderr, "avm_gpu: inverse transform hook enabled on %s\n",
          avm_gpu_device_name(g_hook.ctx));
  g_hook.state = 2;
  return 1;
}

/* Caller holds g_lock. The 1D kernel for one axis, or NULL when the reference
 * has none at this length (it asserts there, so the encoder never asks). */
static const int *itx_kernel_locked(int type, int n) {
  switch (type) {
    case DCT2:
      return n == 4    ? &tx_kernel_dct2_size4[INV_TXFM][0][0]
             : n == 8  ? &tx_kernel_dct2_size8[INV_TXFM][0][0]
             : n == 16 ? &tx_kernel_dct2_size16[INV_TXFM][0][0]
             : n == 32 ? &tx_kernel_dct2_size32[INV_TXFM][0][0]
                       : NULL;
    case IDT: {
      static const int kScale[6] = { 0, 0, 128, 181, 256, 362 };
      const int size_log2 = n == 4 ? 2 : (n == 8 ? 3 : (n == 16 ? 4 : 5));
      if (n != g_hook.idtx_n) {
        memset(g_hook.idtx, 0, sizeof(g_hook.idtx));
        for (int i = 0; i < n; i++) {
          g_hook.idtx[i * n + i] = kScale[size_log2];
        }
        g_hook.idtx_n = n;
      }
      return g_hook.idtx;
    }
    case DST7:
      return n == 4    ? &tx_kernel_adst_size4[INV_TXFM][0][0]
             : n == 8  ? &tx_kernel_adst_size8[INV_TXFM][0][0]
             : n == 16 ? &tx_kernel_adst_size16[INV_TXFM][0][0]
                       : NULL;
    case DCT8:
      return n == 4    ? &tx_kernel_fdst_size4[INV_TXFM][0][0]
             : n == 8  ? &tx_kernel_fdst_size8[INV_TXFM][0][0]
             : n == 16 ? &tx_kernel_fdst_size16[INV_TXFM][0][0]
                       : NULL;
    case DDTX:
      return n == 8    ? &tx_kernel_ddtx_size8[INV_TXFM][0][0]
             : n == 16 ? &tx_kernel_ddtx_size16[INV_TXFM][0][0]
                       : NULL;
    case FDDT: {
      const int *m = n == 8    ? &tx_kernel_ddtx_size8[INV_TXFM][0][0]
                     : n == 16 ? &tx_kernel_ddtx_size16[INV_TXFM][0][0]
                               : NULL;
      if (!m) return NULL;
      for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
          g_hook.fddt[r * n + c] = m[r * n + (n - 1 - c)];
        }
      }
      return g_hook.fddt;
    }
    default: return NULL;
  }
}

int avm_gpu_hook_itxfm(const int32_t *coeff, uint16_t *dest, int stride,
                       int tx_w, int tx_h, int bd, int row_type, int col_type) {
  if (!coeff || !dest || stride < tx_w || tx_w < 4 || tx_h < 4 ||
      tx_w > 64 || tx_h > 64) {
    return 0;
  }

  pthread_mutex_lock(&g_lock);
  int handled = 0;
  if (hook_ready_locked()) {
    const int w1 = tx_w > 32 ? 32 : tx_w;
    const int h1 = tx_h > 32 ? 32 : tx_h;
    const int *mat_w = itx_kernel_locked(row_type, w1);
    const int *mat_h = itx_kernel_locked(col_type, h1);
    if (mat_w && mat_h) {
      char err[AVM_GPU_ERR_LEN];
      const uint32_t pos = 0;
      if (avm_gpu_itxfm_add_tx(g_hook.ctx, g_hook.it, coeff, &pos, 1, tx_w,
                               tx_h, bd, mat_w, mat_h, dest, stride, tx_h,
                               err) == 0) {
        g_hook.blocks++;
        handled = 1;
      } else if (getenv("AV2_GPU_ITXFM_DEBUG")) {
        fprintf(stderr, "avm_gpu: inverse transform fallback %dx%d: %s\n", tx_w,
                tx_h, err);
      }
    } else {
      g_hook.cpu_blocks++;
    }
  }
  pthread_mutex_unlock(&g_lock);
  return handled;
}
