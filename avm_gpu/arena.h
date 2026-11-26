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

#ifndef AVM_AVM_GPU_VULKAN_ARENA_H_
#define AVM_AVM_GPU_VULKAN_ARENA_H_

#include <stddef.h>
#include <stdint.h>

#include "avm_gpu/vulkan_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Device-local arena: one VkBuffer suballocated in 64-byte aligned blocks,
 * with a host-visible staging buffer for uploads and readbacks. Used by the
 * SAD and variance modules when AV2_GPU_ARENA=1; the default is the simpler
 * host-visible path.
 */
typedef struct avm_gpu_arena avm_gpu_arena;

int avm_gpu_arena_create(avm_gpu_ctx *ctx, VkDeviceSize bytes,
                         avm_gpu_arena **out, char err[AVM_GPU_ERR_LEN]);
void avm_gpu_arena_destroy(avm_gpu_ctx *ctx, avm_gpu_arena *arena);

VkBuffer avm_gpu_arena_buffer(const avm_gpu_arena *arena);

/* First-fit suballocation; returns 0 and the byte offset on success. */
int avm_gpu_arena_alloc(avm_gpu_ctx *ctx, avm_gpu_arena *arena,
                        VkDeviceSize size, VkDeviceSize *offset,
                        char err[AVM_GPU_ERR_LEN]);
void avm_gpu_arena_free(avm_gpu_arena *arena, VkDeviceSize offset,
                        VkDeviceSize size);

/* Staging copy in and out of the arena. */
int avm_gpu_arena_upload(avm_gpu_ctx *ctx, avm_gpu_arena *arena,
                         VkDeviceSize offset, const void *data,
                         VkDeviceSize size, char err[AVM_GPU_ERR_LEN]);
int avm_gpu_arena_download(avm_gpu_ctx *ctx, avm_gpu_arena *arena,
                           VkDeviceSize offset, void *data, VkDeviceSize size,
                           char err[AVM_GPU_ERR_LEN]);

/*
 * Buffer slot helpers. A slot is either an ordinary host-visible buffer
 * (arena == NULL) or a suballocation of the arena. The two are told apart by
 * buf->mapped: arena slots have none.
 */
int avm_gpu_slot_reserve(avm_gpu_ctx *ctx, avm_gpu_arena *arena,
                         avm_gpu_buf *buf, VkDeviceSize *cap, VkDeviceSize need,
                         char err[AVM_GPU_ERR_LEN]);
int avm_gpu_slot_upload(avm_gpu_ctx *ctx, avm_gpu_arena *arena,
                        const avm_gpu_buf *buf, const void *data,
                        VkDeviceSize bytes, char err[AVM_GPU_ERR_LEN]);
int avm_gpu_slot_download(avm_gpu_ctx *ctx, avm_gpu_arena *arena,
                          const avm_gpu_buf *buf, void *data, VkDeviceSize bytes,
                          char err[AVM_GPU_ERR_LEN]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AVM_AVM_GPU_VULKAN_ARENA_H_
