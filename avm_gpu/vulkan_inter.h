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
/*
 * av2_highbd_convolve_2d_sr_c() for one reference, unscaled: filter_x and
 * filter_y are the 16 x 8-tap kernel tables (the row at the subpel position is
 * used), round0/round1 come
 * from the ConvolveParams of the call (ROUND0_BITS and 2*FILTER_BITS-round_0
 * for a single reference), and the intermediate is 16-bit without a clip.
 * src must point at the block origin with 3 samples of border on each side.
 */
int avm_gpu_inter_convolve_sr(avm_gpu_ctx *ctx, avm_gpu_inter *in,
                              const uint16_t *src, int src_stride, int src_rows,
                              int sx, int sy, const int16_t *filter_x,
                              const int16_t *filter_y, int subpel_x_qn,
                              int subpel_y_qn, int w, int h, int bd, int round0,
                              int round1, uint16_t *dst, int dst_stride,
                              char err[AVM_GPU_ERR_LEN]);

/*
 * av2_highbd_convolve_2d_scale_c() for one reference: subpel positions and
 * steps in 1/1024 units (SCALE_SUBPEL_BITS), filter rows resolved per pixel,
 * same rounding as the unscaled sr path.
 */
int avm_gpu_inter_convolve_scaled_sr(
    avm_gpu_ctx *ctx, avm_gpu_inter *in, const uint16_t *src, int src_stride,
    int src_rows, int sx, int sy, const int16_t *filter_x,
    const int16_t *filter_y, int subpel_x_qn, int x_step_qn, int subpel_y_qn,
    int y_step_qn, int w, int h, int bd, int round0, int round1, uint16_t *dst,
    int dst_stride, char err[AVM_GPU_ERR_LEN]);

/*
 * Compound prediction, the CWP family: av2_highbd_cwp_convolve_2d_c() for one
 * reference and one pass. conv (uint16, conv_stride) is the high-precision
 * buffer; conv_index 0 with do_average 0 writes only that, conv_index 1
 * combines with it, subtracts the two pass biases and writes pixels.
 * With scaled set, positions and steps are in 1/1024 units and the 1/1024
 * phase path of av2_highbd_convolve_2d_scale_c() is used, which is what the
 * facade runs for a scaled compound reference.
 */
int avm_gpu_inter_cwp_convolve(
    avm_gpu_ctx *ctx, avm_gpu_inter *in, const uint16_t *src, int src_stride,
    int src_rows, int sx, int sy, const int16_t *filter_x,
    const int16_t *filter_y, int subpel_x_qn, int subpel_y_qn, int w, int h,
    int bd, int round0, int round1, uint16_t *conv, int conv_stride,
    int conv_index, int do_average, int uneven, int fwd, int bck, int scaled,
    int x_step_qn, int y_step_qn, uint16_t *dst, int dst_stride,
    char err[AVM_GPU_ERR_LEN]);

int avm_gpu_inter_blend_a64(avm_gpu_ctx *ctx, avm_gpu_inter *inter,
                            const uint16_t *src0, int s0_stride,
                            const uint16_t *src1, int s1_stride,
                            const uint8_t *mask, int mask_stride, int w, int h,
                            int subw, int subh, int bd, uint16_t *dst,
                            int dst_stride, char err[AVM_GPU_ERR_LEN]);

/*
 * Weighted (CWP) compound: comp = ROUND_POWER_OF_TWO(pred * bck_offset +
 * ref * fwd_offset, 4), matching avm_highbd_cwp_c().
 */
int avm_gpu_inter_cwp(avm_gpu_ctx *ctx, avm_gpu_inter *inter,
                      const uint16_t *pred, int pred_stride,
                      const uint16_t *ref, int ref_stride, int w, int h,
                      int fwd_offset, int bck_offset, uint16_t *out,
                      int out_stride, char err[AVM_GPU_ERR_LEN]);

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
