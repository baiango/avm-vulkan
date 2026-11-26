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

#ifndef AVM_AVM_GPU_VULKAN_PIPELINE_H_
#define AVM_AVM_GPU_VULKAN_PIPELINE_H_

#include <stddef.h>
#include <stdint.h>

#include "avm_gpu/vulkan_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct avm_gpu_pipeline avm_gpu_pipeline;

/*
 * Shared compute-pipeline setup for the avm_gpu kernels: SPIR-V module,
 * descriptor set layout with n_bindings storage buffers, push-constant range,
 * pipeline, descriptor pool/set and one reusable command buffer.
 *
 * spv_path_default is baked in at build time; setting the environment variable
 * named by spv_env (when non-NULL) overrides it.
 */
int avm_gpu_pipeline_create(avm_gpu_ctx *ctx, const char *spv_path_default,
                            const char *spv_env, int n_bindings,
                            size_t push_constants_size,
                            avm_gpu_pipeline **out,
                            char err[AVM_GPU_ERR_LEN]);
void avm_gpu_pipeline_destroy(avm_gpu_ctx *ctx, avm_gpu_pipeline *pipe);

/* Binds bufs[0..n_bindings-1] in order, pushes pc and dispatches groups.x1. */
int avm_gpu_pipeline_dispatch(avm_gpu_ctx *ctx, avm_gpu_pipeline *pipe,
                              const avm_gpu_buf *bufs, int n_bindings,
                              const void *pc, size_t pc_size, uint32_t groups,
                              char err[AVM_GPU_ERR_LEN]);

/* Same, with an explicit 2D workgroup grid. */
int avm_gpu_pipeline_dispatch_2d(avm_gpu_ctx *ctx, avm_gpu_pipeline *pipe,
                                 const avm_gpu_buf *bufs, int n_bindings,
                                 const void *pc, size_t pc_size, uint32_t gx,
                                 uint32_t gy, char err[AVM_GPU_ERR_LEN]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AVM_AVM_GPU_VULKAN_PIPELINE_H_
