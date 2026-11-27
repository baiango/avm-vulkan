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

#ifndef AVM_AVM_GPU_VULKAN_HOOK_ITXFM_H_
#define AVM_AVM_GPU_VULKAN_HOOK_ITXFM_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Inverse transform + reconstruction add seam: inv_txfm_c()'s 1D kernels for
 * one block. row_type / col_type are the TX1D_TYPE values g_hor_tx_type /
 * g_ver_tx_type select (DCT2, IDT, DST7, DCT8, DDTX, FDDT) *after* the DDT
 * substitution, exactly as the reference resolves them. Returns 1 when the GPU
 * added the block, 0 to let the C run (unsupported type/size, or the hook is
 * off). Inert unless AV2_GPU_ITXFM=1; AV2_GPU_ITXFM_STATS=1 prints the number
 * of blocks at exit.
 */
int avm_gpu_hook_itxfm(const int32_t *coeff, uint16_t *dest, int stride,
                       int tx_w, int tx_h, int bd, int row_type, int col_type);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AVM_AVM_GPU_VULKAN_HOOK_ITXFM_H_
