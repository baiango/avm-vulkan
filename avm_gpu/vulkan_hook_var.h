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

#ifndef AVM_AVM_GPU_VULKAN_HOOK_VAR_H_
#define AVM_AVM_GPU_VULKAN_HOOK_VAR_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Encoder integration seam for the variance/SSE kernel.
 *
 * Called from avm_dsp/variance.c::highbd_variance64() when the library is
 * built with AVM_GPU_HOOKS. Returns 1 and fills *sse / *sum when the Vulkan
 * kernel produced the raw accumulators, 0 when the caller must run the CPU
 * loop. Inert unless AV2_GPU_VAR=1; AV2_GPU_VAR_STATS=1 prints the block count.
 */
int avm_gpu_hook_variance64(const uint16_t *a, int a_stride, const uint16_t *b,
                            int b_stride, int w, int h, uint64_t *sse,
                            int64_t *sum);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AVM_AVM_GPU_VULKAN_HOOK_VAR_H_
