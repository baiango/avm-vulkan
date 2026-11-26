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

#ifndef AVM_AVM_GPU_VULKAN_VAR_H_
#define AVM_AVM_GPU_VULKAN_VAR_H_

#include <stdint.h>

#include "avm_gpu/vulkan_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* One candidate: where the block starts in each plane, in samples. */
typedef struct avm_gpu_var_pos {
  uint32_t src_off;
  uint32_t ref_off;
} avm_gpu_var_pos;

typedef struct avm_gpu_var avm_gpu_var;

int avm_gpu_var_create(avm_gpu_ctx *ctx, avm_gpu_var **var,
                       char err[AVM_GPU_ERR_LEN]);
void avm_gpu_var_destroy(avm_gpu_ctx *ctx, avm_gpu_var *var);

/* Uploads the source and reference planes once; they stay resident. */
int avm_gpu_var_set_planes(avm_gpu_ctx *ctx, avm_gpu_var *var,
                           const uint16_t *src, int src_stride, int src_rows,
                           const uint16_t *ref, int ref_stride, int ref_rows,
                           char err[AVM_GPU_ERR_LEN]);

/*
 * Batched variance/SSE for one block size and bitdepth (8, 10 or 12), matching
 * avm_highbd_{8,10,12}_varianceWxH_c:
 *
 *   sse_out[i]      == the reference's *sse (already bitdepth-scaled)
 *   variance_out[i] == the reference's return value
 *
 * Planes must have been set with avm_gpu_var_set_planes().
 */
int avm_gpu_var_run(avm_gpu_ctx *ctx, avm_gpu_var *var,
                    const avm_gpu_var_pos *blocks, int n, int bw, int bh,
                    int bitdepth, uint32_t *variance_out, uint32_t *sse_out,
                    char err[AVM_GPU_ERR_LEN]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AVM_AVM_GPU_VULKAN_VAR_H_
