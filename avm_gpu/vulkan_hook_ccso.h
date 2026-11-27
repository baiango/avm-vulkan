/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at aomedia.org/license/software-license/bsd-3-c/.  If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * aomedia.org/license/patent-license/.
 */

#ifndef AVM_AVM_GPU_VULKAN_HOOK_CCSO_H_
#define AVM_AVM_GPU_VULKAN_HOOK_CCSO_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * CCSO seam for one processing unit, the work ccso_filter_block_hbd_wo_buf()
 * does: src_y is the luma reference (padded by CCSO_PADDING_SIZE on every
 * side) the band and class decisions use, dst_yuv is the plane being offset and
 * is updated in place, x/y name the block, and the rest are the C function's
 * own arguments. A block closer than 2 columns or 1 row to the left/top edge is
 * left to the C, whose buffers carry CCSO_PADDING_SIZE there.
 *
 * Returns 1 when the GPU filtered the block, 0 to let the C run. Inert unless
 * AV2_GPU_CCSO=1; AV2_GPU_CCSO_STATS=1 prints the block count at exit.
 */
int avm_gpu_hook_ccso(const uint16_t *src_y, uint16_t *dst_yuv, int x, int y,
                      int pic_width, int pic_height, const int8_t *offset_buf,
                      int src_stride, int dst_stride, int y_uv_hscale,
                      int y_uv_vscale, int thr, int neg_thr, int src_loc_0,
                      int src_loc_1, int max_val, int blk_w, int blk_h,
                      int single_band, int shift_bits, int edge_clf,
                      int bo_only);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AVM_AVM_GPU_VULKAN_HOOK_CCSO_H_
