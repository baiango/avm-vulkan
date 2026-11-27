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

#ifndef AVM_AVM_GPU_VULKAN_HOOK_CDEF_H_
#define AVM_AVM_GPU_VULKAN_HOOK_CDEF_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * CDEF seam for one filter block, the work av2_cdef_filter_fb() hands to
 * av2_cdef_filter_16_{0,1,2,3}(): `in` is the CDEF source buffer (its rows are
 * packed with CDEF_BSTRIDE), i0/j0 place the block inside it, dst/dstride are
 * the destination block, and the rest are the C filter's own arguments
 * (enable_pri/enable_sec stand in for the wrapper variant). Only the block plus
 * the filter's 2/8-sample border is uploaded.
 *
 * Returns 1 when the GPU filtered the block, 0 to let the C run. Inert unless
 * AV2_GPU_CDEF=1; AV2_GPU_CDEF_STATS=1 prints the block count at exit,
 * AV2_GPU_CDEF_VERIFY=1 runs the C filter on the same input, reports every
 * differing sample and keeps the C result.
 */
int avm_gpu_hook_cdef(const uint16_t *in, int in_rows, int stride, int i0,
                      int j0, int w, int h, int dir, int pri_strength,
                      int sec_strength, int pri_damping, int sec_damping,
                      int coeff_shift, int enable_pri, int enable_sec,
                      uint16_t *dst, int dstride);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AVM_AVM_GPU_VULKAN_HOOK_CDEF_H_
