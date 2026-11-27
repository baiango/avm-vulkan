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

#ifndef AVM_AVM_GPU_VULKAN_DEBLOCK_H_
#define AVM_AVM_GPU_VULKAN_DEBLOCK_H_

#include <stdint.h>

#include "avm_gpu/vulkan_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct avm_gpu_deblock avm_gpu_deblock;

int avm_gpu_deblock_create(avm_gpu_ctx *ctx, avm_gpu_deblock **out,
                           char err[AVM_GPU_ERR_LEN]);
void avm_gpu_deblock_destroy(avm_gpu_ctx *ctx, avm_gpu_deblock *db);

/*
 * One 4-sample deblock call, matching avm_highbd_lpf_{horizontal,vertical}
 * generic_c(): the 4 samples start at (x, y) of a stride x rows plane.
 * vertical != 0 selects the vertical filter (pitch 1). out receives the whole
 * filtered plane.
 */
int avm_gpu_deblock_generic(avm_gpu_ctx *ctx, avm_gpu_deblock *db,
                            const uint16_t *plane, int stride, int rows, int x,
                            int y, int filt_neg, int filt_pos, int q_thresh,
                            int side_thresh, int bd, int ll_neg, int ll_pos,
                            int vertical, uint16_t *out,
                            char err[AVM_GPU_ERR_LEN]);

/*
 * Windowed variant for the encode seam: p points at the edge inside a real
 * frame buffer, across_pitch is the stride of the delta direction (1 for a
 * vertical filter, the frame stride for a horizontal one) and step is the
 * distance between the four filtered samples (the frame stride for a vertical
 * filter, 1 for a horizontal one). Only the samples the filter can read or
 * write are uploaded and written back, so a caller's plane or band extent never
 * matters and neighbouring rows cannot be disturbed.
 */
int avm_gpu_deblock_window(avm_gpu_ctx *ctx, avm_gpu_deblock *db,
                           const uint16_t *p, int across_pitch, int step,
                           int filt_neg, int filt_pos, int q_thresh,
                           int side_thresh, int bd, int ll_neg, int ll_pos,
                           uint16_t *out, char err[AVM_GPU_ERR_LEN]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AVM_AVM_GPU_VULKAN_DEBLOCK_H_
