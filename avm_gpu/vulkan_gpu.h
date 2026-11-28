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

#ifndef AVM_AVM_GPU_VULKAN_GPU_H_
#define AVM_AVM_GPU_VULKAN_GPU_H_

#include <stddef.h>
#include <stdint.h>

#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Largest error string avm_gpu_* fills in. */
#define AVM_GPU_ERR_LEN 256

/* Owns the instance, the compute device, one queue and the command pool. */
typedef struct avm_gpu_ctx avm_gpu_ctx;

/*
 * A host-visible, coherent storage-buffer block. M0 keeps every plane in this
 * memory; a later milestone stages through device-local memory + a copy.
 */
typedef struct avm_gpu_buf {
  VkBuffer buffer;
  VkDeviceMemory memory;
  void *mapped;
  VkDeviceSize size;
  /* Suballocation inside a shared buffer (arena); 0/0 = whole buffer. */
  VkDeviceSize offset;
} avm_gpu_buf;

/* Returns 0 on success and -1 on failure, with err filled in. */
int avm_gpu_ctx_create(avm_gpu_ctx **ctx, char err[AVM_GPU_ERR_LEN]);
void avm_gpu_ctx_destroy(avm_gpu_ctx *ctx);

/* Device name and driver info, for logs and reports. */
const char *avm_gpu_device_name(const avm_gpu_ctx *ctx);

/* Buffer helpers; size is in bytes. */
int avm_gpu_buf_create(avm_gpu_ctx *ctx, VkDeviceSize size, avm_gpu_buf *buf,
                       char err[AVM_GPU_ERR_LEN]);
void avm_gpu_buf_destroy(avm_gpu_ctx *ctx, avm_gpu_buf *buf);

/* Synchronous submit of one recorded command buffer. */
int avm_gpu_submit_wait(avm_gpu_ctx *ctx, VkCommandBuffer cb,
                        char err[AVM_GPU_ERR_LEN]);

/* Exposed for the kernels' host side; do not use outside avm_gpu. */
VkDevice avm_gpu_device(const avm_gpu_ctx *ctx);
/* minStorageBufferOffsetAlignment of the device; arena suballocations and any
 * buffer offset handed to a descriptor or a copy respect it. */
VkDeviceSize avm_gpu_storage_offset_alignment(const avm_gpu_ctx *ctx);
VkPhysicalDevice avm_gpu_physical_device(const avm_gpu_ctx *ctx);
VkCommandPool avm_gpu_command_pool(const avm_gpu_ctx *ctx);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  // AVM_AVM_GPU_VULKAN_GPU_H_
