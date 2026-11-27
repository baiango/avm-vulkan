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

#ifndef AVM_AVM_GPU_VULKAN_HOOK_DEBLOCK_H_
#define AVM_AVM_GPU_VULKAN_HOOK_DEBLOCK_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Deblock seam for one 4-sample edge of avm_highbd_lpf_{vertical,horizontal}
 * generic_c(): p points at the edge inside the caller's frame buffer,
 * across_pitch is the delta direction (1 for a vertical filter, the frame
 * stride for a horizontal one) and step the distance between the four filtered
 * samples (the stride for vertical, 1 for horizontal). Only the window the
 * filter can read or write is uploaded, so the caller's plane or band extent
 * does not matter. The rest are the C's own arguments. Returns 1 when the GPU
 * filtered the edge in place, 0 to let the C run. Inert unless
 * AV2_GPU_DEBLOCK=1; AV2_GPU_DEBLOCK_STATS=1 prints the number of edges at
 * exit, AV2_GPU_DEBLOCK_VERIFY=1 re-runs the C filter on the same window,
 * reports the first differing sample and keeps the C result.
 */
int avm_gpu_hook_deblock(uint16_t *p, int across_pitch, int step, int filt_neg,
                         int filt_pos, int q_thresh, int side_thresh, int bd,
                         int ll_neg, int ll_pos, int vertical);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AVM_AVM_GPU_VULKAN_HOOK_DEBLOCK_H_
