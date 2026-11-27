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

/*
 * Forward 2D transform mirroring fwd_txfm_c(): tx_w/tx_h are the transform size
 * (4..64; a 64-length axis is box-downsampled first), shift1/shift2 are
 * fwd_tx_shift[tx]. Each axis carries a kind - 0 = DCT2 folded butterflies over
 * tx_kernel_dct2_size{tx}, 1 = identity with scale_w/scale_h, 2 = a matrix
 * product over mat_w/mat_h (DST7/DCT8/DDTX/FDDT), read reversed when flip is
 * set. pos holds one (residual sample offset, coefficient offset) per block.
 */
int avm_gpu_txfm_fwd(avm_gpu_ctx *ctx, avm_gpu_txfm *t, const int16_t *resi,
                     int diff_stride, const avm_gpu_txfm_pos *pos, int n,
                     int tx_w, int tx_h, int shift1, int shift2, int kind_w,
                     int kind_h, int scale_w, int scale_h, int flip_w,
                     int flip_h, const int *mat_w, const int *mat_h,
                     int32_t *coeff, char err[AVM_GPU_ERR_LEN]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AVM_AVM_GPU_VULKAN_TXFM_H_
