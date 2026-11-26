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

#ifndef AVM_AVM_GPU_VULKAN_INTER_H_
#define AVM_AVM_GPU_VULKAN_INTER_H_

#include <stdint.h>

#include "avm_gpu/vulkan_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct avm_gpu_inter avm_gpu_inter;

int avm_gpu_inter_create(avm_gpu_ctx *ctx, avm_gpu_inter **out,
                         char err[AVM_GPU_ERR_LEN]);
void avm_gpu_inter_destroy(avm_gpu_ctx *ctx, avm_gpu_inter *inter);

/*
 * 8-tap separable subpel prediction (whole-pel when x0_q4 == y0_q4 == 0),
 * matching avm_highbd_convolve8_c() with x_step_q4 = y_step_q4 = 16.
 *
 * src points at the reference plane, (sx, sy) at the block origin; the caller
 * must ensure the 3-sample filter border around the block is inside the plane,
 * as AVM does. filter is the 16 x 8 InterpKernel table.
 */
/*
 * Same with an explicit subpel step (16 = unscaled reference). Any
 * x_step_q4 / y_step_q4 in 1..64 is supported; the source border needed by the
 * 8-tap filter grows with the scale factor and is checked on entry.
 */
/*
 * Compound prediction: A64 mask blend of two predictors, matching
 * avm_highbd_blend_a64_mask_c() (mask subsampling subw/subh included).
 */
int avm_gpu_inter_blend_a64(avm_gpu_ctx *ctx, avm_gpu_inter *inter,
                            const uint16_t *src0, int s0_stride,
                            const uint16_t *src1, int s1_stride,
                            const uint8_t *mask, int mask_stride, int w, int h,
                            int subw, int subh, int bd, uint16_t *dst,
                            int dst_stride, char err[AVM_GPU_ERR_LEN]);

int avm_gpu_inter_convolve_scaled(avm_gpu_ctx *ctx, avm_gpu_inter *inter,
                                  const uint16_t *src, int src_stride,
                                  int src_rows, int sx, int sy,
                                  const int16_t *filter, int x0_q4,
                                  int y0_q4, int x_step_q4, int y_step_q4,
                                  int w, int h, int bd, uint16_t *dst,
                                  int dst_stride, char err[AVM_GPU_ERR_LEN]);

int avm_gpu_inter_convolve(avm_gpu_ctx *ctx, avm_gpu_inter *inter,
                           const uint16_t *src, int src_stride, int src_rows,
                           int sx, int sy, const int16_t *filter, int x0_q4,
                           int y0_q4, int w, int h, int bd, uint16_t *dst,
                           int dst_stride, char err[AVM_GPU_ERR_LEN]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AVM_AVM_GPU_VULKAN_INTER_H_
