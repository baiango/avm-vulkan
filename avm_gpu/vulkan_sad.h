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

#ifndef AVM_AVM_GPU_VULKAN_SAD_H_
#define AVM_AVM_GPU_VULKAN_SAD_H_

#include <stdint.h>

#include "avm_gpu/vulkan_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* One candidate: where the block starts in each plane, in samples. */
typedef struct avm_gpu_sad_pos {
  uint32_t src_off;
  uint32_t ref_off;
} avm_gpu_sad_pos;

typedef struct avm_gpu_sad avm_gpu_sad;

int avm_gpu_sad_create(avm_gpu_ctx *ctx, avm_gpu_sad **sad,
                       char err[AVM_GPU_ERR_LEN]);
void avm_gpu_sad_destroy(avm_gpu_ctx *ctx, avm_gpu_sad *sad);

/*
 * Uploads the source and reference planes once; they stay resident until the
 * next call. Planes are 16-bit samples with an explicit stride in samples.
 */
int avm_gpu_sad_set_planes(avm_gpu_ctx *ctx, avm_gpu_sad *sad,
                           const uint16_t *src, int src_stride, int src_rows,
                           const uint16_t *ref, int ref_stride, int ref_rows,
                           char err[AVM_GPU_ERR_LEN]);

/*
 * Batched high-bitdepth SAD: every entry of blocks is priced at the same
 * block size, matching how avm_highbd_sadWxH_c is called per candidate in
 * motion search. Planes must have been set with avm_gpu_sad_set_planes().
 *
 * cost[n] receives the per-block SAD, bit-identical to the C reference.
 * Returns 0 on success, -1 with err filled in otherwise.
 */
int avm_gpu_sad_run(avm_gpu_ctx *ctx, avm_gpu_sad *sad,
                    const avm_gpu_sad_pos *blocks, int n, int bw, int bh,
                    uint32_t *cost, char err[AVM_GPU_ERR_LEN]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AVM_AVM_GPU_VULKAN_SAD_H_
