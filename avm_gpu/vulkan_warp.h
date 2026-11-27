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

#ifndef AVM_AVM_GPU_VULKAN_WARP_H_
#define AVM_AVM_GPU_VULKAN_WARP_H_

#include <stdint.h>

#include "avm_gpu/vulkan_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AVM_GPU_WARP_FILTER_ROWS (64 * 7 + 1)
#define AVM_GPU_WARP_EXT_FILTER_ROWS (64 + 1)

typedef struct avm_gpu_warp avm_gpu_warp;

int avm_gpu_warp_create(avm_gpu_ctx *ctx, avm_gpu_warp **out,
                        char err[AVM_GPU_ERR_LEN]);
void avm_gpu_warp_destroy(avm_gpu_ctx *ctx, avm_gpu_warp *warp);

/*
 * Affine (warped) prediction, single-reference case, matching
 * av2_highbd_warp_affine_c() including the chroma projection shifts
 * (ss_x / ss_y) and conv_params->is_compound == 0. mat is the 6-entry warp matrix, filter the
 * av2_warped_filter table (AVM_GPU_WARP_FILTER_ROWS x 8 int16 values).
 * pred is the whole prediction plane; only the p_col/p_row window is written.
 */
int avm_gpu_warp_affine(avm_gpu_ctx *ctx, avm_gpu_warp *warp,
                        const int32_t mat[6], const uint16_t *ref, int width,
                        int height, int stride, uint16_t *pred, int p_col,
                        int p_row, int p_width, int p_height, int p_stride,
                        int bd, int round_0, int alpha, int beta, int gamma,
                        int delta, int ss_x, int ss_y, const int16_t *filter,
                        char err[AVM_GPU_ERR_LEN]);

/*
 * Extended-range affine prediction, matching av2_ext_highbd_warp_affine_c()
 * for the single-prediction case: 4x4 chunks and the 65-row
 * av2_ext_warped_filter table. Blocks are multiples of 4.
 */
int avm_gpu_warp_affine_ext(avm_gpu_ctx *ctx, avm_gpu_warp *warp,
                            const int32_t mat[6], const uint16_t *ref,
                            int width, int height, int stride, uint16_t *pred,
                            int p_col, int p_row, int p_width, int p_height,
                            int p_stride, int bd, int round_0, int ss_x,
                            int ss_y, const int16_t *filter,
                            char err[AVM_GPU_ERR_LEN]);

/*
 * Scaled (resized-reference) affine prediction, matching
 * av2_ext_highbd_warp_affine_scaled_c() for the single-prediction case: the
 * chunk corner is rescaled with x_scale_fp / y_scale_fp (REF_SCALE_SHIFT) and
 * walked with x_step_q4 / y_step_q4, and every read is clamped to the 8x8
 * sub-block limits in box (MAX_WARP_BD_SIZE^2 entries of x0, x1, y0, y1).
 * Blocks are multiples of 4 in neither direction: partial chunks are written.
 */
int avm_gpu_warp_affine_scaled(avm_gpu_ctx *ctx, avm_gpu_warp *warp,
                               const int32_t mat[6], const uint16_t *ref,
                               int width, int height, int stride,
                               uint16_t *pred, int p_col, int p_row,
                               int p_width, int p_height, int p_stride, int bd,
                               int round_0, int ss_x, int ss_y, int x_scale_fp,
                               int y_scale_fp, int x_step_q4, int y_step_q4,
                               const int32_t *box, const int16_t *filter,
                               char err[AVM_GPU_ERR_LEN]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AVM_AVM_GPU_VULKAN_WARP_H_
