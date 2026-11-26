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

#ifndef AVM_AVM_GPU_VULKAN_HOOK_H_
#define AVM_AVM_GPU_VULKAN_HOOK_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Largest candidate batch the batched entry point accepts. */
#define AVM_GPU_HOOK_MAX_BATCH 16

/*
 * Encoder integration seam for the C SAD path.
 *
 * Called from avm_dsp/sad.c::highbd_sad() when the library is built with
 * AVM_GPU_HOOKS. Returns 1 and writes *sad when the Vulkan path priced the
 * block, 0 when the caller must use the CPU path.
 *
 * Disabled unless the environment variable AV2_GPU_SAD is set to 1, so a
 * default build keeps the CPU path untouched. Set AV2_GPU_SAD_STATS=1 to print
 * the number of blocks that went through the GPU at exit.
 */
/* AV2_GPU_SAD=1 (or "batch"): batched entry only; "single": per-call entry
 * only; "all": both; unset/0: the hook is inert. */
int avm_gpu_hook_sad(const uint16_t *src, int src_stride, const uint16_t *ref,
                     int ref_stride, int width, int height,
                     unsigned int *sad);

/*
 * Batched form used by motion search: prices n candidates against the same
 * source block in one dispatch. Entries with a NULL reference pointer are
 * skipped and their sad[] slot is left untouched. Returns 1 when handled.
 */
int avm_gpu_hook_sad_multi(const uint16_t *src, int src_stride,
                           const uint16_t *const *ref_ptrs, int ref_stride,
                           int width, int height, int n, unsigned int *sad);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AVM_AVM_GPU_VULKAN_HOOK_H_
