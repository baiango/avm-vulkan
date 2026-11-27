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

#ifndef AVM_AVM_GPU_VULKAN_HOOK_INTER_H_
#define AVM_AVM_GPU_VULKAN_HOOK_INTER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Inter prediction seam for single-reference, unscaled subpel prediction.
 *
 * Called from av2/common/reconinter.c::av2_make_inter_predictor() when the
 * library is built with AVM_GPU_HOOKS, before the RTCD convolve facade. It
 * handles what the kernels model bit-exactly - one reference, scaled or
 * unscaled, 8-tap x and y filters, no intrabc, blocks up to 64x64 - and
 * returns 0 for everything else so the caller runs the C path. Positions and
 * steps follow the caller's convention: 1/16 units after
 * revert_scale_extra_bits() for an unscaled reference, 1/1024
 * (SCALE_SUBPEL_BITS) for a scaled one.
 *
 * src points at the block origin and must have 3 samples of border on every
 * side; filter_x / filter_y are the 16 x 8-tap kernel tables.
 *
 * Inert unless AV2_GPU_INTER=1. AV2_GPU_INTER_STATS=1 prints the number of
 * blocks predicted on the GPU at exit.
 */
int avm_gpu_hook_inter(const uint16_t *src, int src_stride, uint16_t *dst,
                       int dst_stride, int w, int h, int bd,
                       const int16_t *filter_x, const int16_t *filter_y,
                       int taps_x, int taps_y, int subpel_x_qn,
                       int subpel_y_qn, int round_0, int round_1, int scaled,
                       int x_step_qn, int y_step_qn, int is_compound,
                       int do_average, uint16_t *conv_dst, int conv_stride,
                       int fwd_offset, int bck_offset);

/*
 * Masked blend seam (inter-intra and wedge masks):
 * avm_highbd_blend_a64_mask(), already gated against the kernel by
 * avm_gpu_inter_test. Same argument order as the C. Returns 1 when the GPU
 * blended the block. Inert unless AV2_GPU_INTER=1.
 */
int avm_gpu_hook_blend_a64(uint16_t *dst, int dst_stride,
                           const uint16_t *src0, int s0_stride,
                           const uint16_t *src1, int s1_stride,
                           const uint8_t *mask, int mask_stride, int w, int h,
                           int subw, int subh, int bd);

/*
 * Masked compound between two inter predictors:
 * avm_highbd_blend_a64_d16_mask() with round_offset / round_bits from the
 * call's ConvolveParams. Same return convention as the blend above.
 */
int avm_gpu_hook_blend_a64_d16(uint16_t *dst, int dst_stride,
                               const uint16_t *src0, int s0_stride,
                               const uint16_t *src1, int s1_stride,
                               const uint8_t *mask, int mask_stride, int w,
                               int h, int subw, int subh, int bd,
                               int round_offset, int round_bits);

/*
 * Warped (affine) prediction seam: av2_highbd_warp_plane() for a single
 * reference, unscaled. The warp parameters arrive field by field (wmtype is
 * the TransformationType, so ROTZOOM can be expanded) so this header stays free
 * of av2 types. Picks av2_highbd_warp_affine() or av2_ext_highbd_warp_affine()
 * exactly as that function does (affine filter and blocks of at least 8x8, else
 * the extended-range one, which the kernel only models without a boundary box).
 * Returns 1 when the GPU wrote the block. Inert unless AV2_GPU_INTER=1.
 */
int avm_gpu_hook_warp(const int32_t wmmat[8], int wmtype, int alpha, int beta,
                      int gamma, int delta, int use_affine_filter, int bd,
                      const uint16_t *ref, int width, int height, int stride,
                      uint16_t *pred, int p_col, int p_row, int p_width,
                      int p_height, int p_stride, int ss_x, int ss_y,
                      int round_0, int is_scaled, int use_warp_bd_box);

/*
 * Forward transform seam: fwd_txfm_c()'s DCT2 path, every size up to 64 (a
 * 64-length axis is box-downsampled first, as the reference does). mat_w/mat_h
 * are
 * tx_kernel_dct2_size{tx}[FWD_TXFM], shift1/shift2 fwd_tx_shift[tx].
 * Inert unless AV2_GPU_FWD_TXFM=1.
 */
int avm_gpu_hook_fwd_txfm(const int16_t *resi, int32_t *coeff, int diff_stride,
                          int tx_w, int tx_h, int shift1, int shift2,
                          const int *mat_w, const int *mat_h);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AVM_AVM_GPU_VULKAN_HOOK_INTER_H_
