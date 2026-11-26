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

#ifndef AVM_AVM_GPU_VULKAN_ITXFM_H_
#define AVM_AVM_GPU_VULKAN_ITXFM_H_

#include <stdint.h>

#include "avm_gpu/vulkan_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct avm_gpu_itxfm avm_gpu_itxfm;

int avm_gpu_itxfm_create(avm_gpu_ctx *ctx, avm_gpu_itxfm **out,
                         char err[AVM_GPU_ERR_LEN]);
void avm_gpu_itxfm_destroy(avm_gpu_ctx *ctx, avm_gpu_itxfm *itxfm);

/*
 * Batched inverse DCT + reconstruction add, DCT_DCT only, size 4 or 8.
 *
 * coeffs holds n blocks of size x size tran_low_t values, dst_off the n
 * (y * stride + x) positions in the dest plane. dest is the prediction plane
 * (stride x rows samples, 16-bit) and is updated in place, bit-identically to
 * inv_txfm_c() from av2/common/idct.c for the same TXFM_PARAM.
 */
int avm_gpu_itxfm_add(avm_gpu_ctx *ctx, avm_gpu_itxfm *itxfm,
                      const int32_t *coeffs, const uint32_t *dst_off, int n,
                      int size, int bd, uint16_t *dest, int stride, int rows,
                      char err[AVM_GPU_ERR_LEN]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AVM_AVM_GPU_VULKAN_ITXFM_H_
