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
int avm_gpu_txfm_fdct(avm_gpu_ctx *ctx, avm_gpu_txfm *txfm,
                      const uint32_t *offsets, int n, int size,
                      int32_t *coeffs, char err[AVM_GPU_ERR_LEN]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AVM_AVM_GPU_VULKAN_TXFM_H_
