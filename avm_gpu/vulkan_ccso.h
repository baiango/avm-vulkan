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

#ifndef AVM_AVM_GPU_VULKAN_CCSO_H_
#define AVM_AVM_GPU_VULKAN_CCSO_H_

#include <stdint.h>

#include "avm_gpu/vulkan_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* filter_offset is CCSO_BAND_NUM (64) x 16 entries in AVM. */
#define AVM_GPU_CCSO_LUT_ENTRIES (64 * 16)

typedef struct avm_gpu_ccso avm_gpu_ccso;

int avm_gpu_ccso_create(avm_gpu_ctx *ctx, avm_gpu_ccso **out,
                        char err[AVM_GPU_ERR_LEN]);
void avm_gpu_ccso_destroy(avm_gpu_ctx *ctx, avm_gpu_ccso *ccso);

/*
 * One CCSO block pass, matching ccso_filter_block_hbd_wo_buf_c()
 * (av2/common/ccso.c): src_y is the luma reference used for band and class
 * decisions, dst is the plane being offset and is updated in place, offset_buf
 * is the 64-entry LUT, src_loc_0/1 come from derive_ccso_sample_pos().
 */
int avm_gpu_ccso_block(avm_gpu_ctx *ctx, avm_gpu_ccso *ccso,
                       const uint16_t *src_y, int src_stride,
                       const uint16_t *dst, int dst_stride, int pic_w,
                       int pic_h, int x, int y, const int8_t *offset_buf,
                       int src_loc_0, int src_loc_1, int y_uv_hscale,
                       int y_uv_vscale, int thr, int neg_thr, int max_val,
                       int blk_w, int blk_h, int single_band, int shift_bits,
                       int edge_clf, int bo_only, uint16_t *out,
                       char err[AVM_GPU_ERR_LEN]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AVM_AVM_GPU_VULKAN_CCSO_H_
