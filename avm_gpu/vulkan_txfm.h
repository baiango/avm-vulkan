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

#ifndef AVM_AVM_GPU_VULKAN_TXFM_H_
#define AVM_AVM_GPU_VULKAN_TXFM_H_

#include <stdint.h>

#include "avm_gpu/vulkan_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct avm_gpu_txfm avm_gpu_txfm;

typedef struct avm_gpu_txfm_pos {
  uint32_t resi_off;  /* residual sample offset */
  uint32_t coef_off;  /* coefficient offset  */
} avm_gpu_txfm_pos;

int avm_gpu_txfm_create(avm_gpu_ctx *ctx, avm_gpu_txfm **out,
                        char err[AVM_GPU_ERR_LEN]);
void avm_gpu_txfm_destroy(avm_gpu_ctx *ctx, avm_gpu_txfm *txfm);

/* Uploads a residual plane once; it stays resident. */
int avm_gpu_txfm_set_source(avm_gpu_ctx *ctx, avm_gpu_txfm *txfm,
                            const int16_t *src, int stride, int rows,
                            char err[AVM_GPU_ERR_LEN]);

/*
 * Batched forward DCT: n blocks of size x size, block i starting at
 * offsets[i] samples in the resident source plane.
 *
 * coeffs receives n * size * size coefficients, row-major per block, and is
 * bit-identical to avm_fdct4x4_c (size 4) or avm_highbd_fdct8x8_c (size 8).
 */
/*
 * Forward 2D DCT2 transform mirroring fwd_txfm_c()'s folded butterflies:
 * tx_w/tx_h are the transform size (4..64; a 64-length axis is box-downsampled
 * first), shift1/shift2 are fwd_tx_shift[tx], mat_w/mat_h are
 * tx_kernel_dct2_size{tx}[FWD_TXFM]. pos holds one (residual sample offset,
 * coefficient offset) per block.
 */
int avm_gpu_txfm_fwd(avm_gpu_ctx *ctx, avm_gpu_txfm *t, const int16_t *resi,
                     int diff_stride, const avm_gpu_txfm_pos *pos, int n,
                     int tx_w, int tx_h, int shift1, int shift2,
                     const int *mat_w, const int *mat_h, int32_t *coeff,
                     char err[AVM_GPU_ERR_LEN]);

int avm_gpu_txfm_fdct(avm_gpu_ctx *ctx, avm_gpu_txfm *txfm,
                      const uint32_t *offsets, int n, int size,
                      int32_t *coeffs, char err[AVM_GPU_ERR_LEN]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AVM_AVM_GPU_VULKAN_TXFM_H_
