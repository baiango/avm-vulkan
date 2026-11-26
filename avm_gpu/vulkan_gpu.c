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

#include "avm_gpu/vulkan_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct avm_gpu_ctx {
  VkInstance instance;
  VkPhysicalDevice physical;
  VkDevice device;
  VkQueue queue;
  VkCommandPool pool;
  uint32_t queue_family;
  VkFence fence;
  char name[256];
};

static void set_err(char err[AVM_GPU_ERR_LEN], const char *msg, VkResult r) {
  if (!err) return;
  snprintf(err, AVM_GPU_ERR_LEN, "%s (VkResult %d)", msg, (int)r);
}

static int has_instance_ext(const char *name) {
  uint32_t n = 0;
  if (vkEnumerateInstanceExtensionProperties(NULL, &n, NULL) != VK_SUCCESS) {
    return 0;
  }
  VkExtensionProperties *ext = calloc(n ? n : 1, sizeof(*ext));
  if (!ext) return 0;
  int found = 0;
  if (vkEnumerateInstanceExtensionProperties(NULL, &n, ext) == VK_SUCCESS) {
    for (uint32_t i = 0; i < n; i++) {
      if (!strcmp(ext[i].extensionName, name)) found = 1;
    }
  }
  free(ext);
  return found;
}

static uint32_t pick_memory_type(VkPhysicalDevice physical, uint32_t bits,
                                 VkMemoryPropertyFlags want) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(physical, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
    if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) {
      return i;
    }
  }
  return UINT32_MAX;
}

int avm_gpu_ctx_create(avm_gpu_ctx **out, char err[AVM_GPU_ERR_LEN]) {
  *out = NULL;
  if (err) err[0] = '\0';

  VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
  app.pApplicationName = "avm_gpu";
  app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
  app.pEngineName = "avm";
  app.apiVersion = VK_API_VERSION_1_1;

  const char *ext[1];
  uint32_t ext_count = 0;
  VkInstanceCreateFlags flags = 0;
  /* MoltenVK is a portability driver: without these, vkCreateInstance fails
   * with VK_ERROR_INCOMPATIBLE_DRIVER. */
  if (has_instance_ext(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
    ext[ext_count++] = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
    flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
  }

  VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
  ici.flags = flags;
  ici.pApplicationInfo = &app;
  ici.enabledExtensionCount = ext_count;
  ici.ppEnabledExtensionNames = ext_count ? ext : NULL;

  avm_gpu_ctx *ctx = calloc(1, sizeof(*ctx));
  if (!ctx) {
    set_err(err, "out of memory", VK_ERROR_OUT_OF_HOST_MEMORY);
    return -1;
  }

  VkResult r = vkCreateInstance(&ici, NULL, &ctx->instance);
  if (r != VK_SUCCESS) {
    set_err(err, "vkCreateInstance", r);
    free(ctx);
    return -1;
  }

  uint32_t ndev = 0;
  r = vkEnumeratePhysicalDevices(ctx->instance, &ndev, NULL);
  if (r != VK_SUCCESS || ndev == 0) {
    set_err(err, "no Vulkan physical device", r);
    avm_gpu_ctx_destroy(ctx);
    return -1;
  }
  VkPhysicalDevice *devs = calloc(ndev, sizeof(*devs));
  if (!devs) {
    set_err(err, "out of memory", VK_ERROR_OUT_OF_HOST_MEMORY);
    avm_gpu_ctx_destroy(ctx);
    return -1;
  }
  vkEnumeratePhysicalDevices(ctx->instance, &ndev, devs);

  /* First device with a compute queue wins; single-GPU hosts are the M0 case. */
  int chosen = -1;
  for (uint32_t d = 0; d < ndev && chosen < 0; d++) {
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(devs[d], &nq, NULL);
    VkQueueFamilyProperties *q = calloc(nq ? nq : 1, sizeof(*q));
    if (!q) break;
    vkGetPhysicalDeviceQueueFamilyProperties(devs[d], &nq, q);
    for (uint32_t i = 0; i < nq; i++) {
      if (q[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        ctx->queue_family = i;
        chosen = (int)d;
        break;
      }
    }
    free(q);
  }
  if (chosen < 0) {
    set_err(err, "no compute queue", VK_ERROR_FEATURE_NOT_PRESENT);
    free(devs);
    avm_gpu_ctx_destroy(ctx);
    return -1;
  }
  ctx->physical = devs[chosen];
  free(devs);

  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(ctx->physical, &props);
  snprintf(ctx->name, sizeof(ctx->name), "%s (Vulkan %u.%u.%u)", props.deviceName,
           VK_API_VERSION_MAJOR(props.apiVersion),
           VK_API_VERSION_MINOR(props.apiVersion),
           VK_API_VERSION_PATCH(props.apiVersion));

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
  qci.queueFamilyIndex = ctx->queue_family;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  r = vkCreateDevice(ctx->physical, &dci, NULL, &ctx->device);
  if (r != VK_SUCCESS) {
    set_err(err, "vkCreateDevice", r);
    avm_gpu_ctx_destroy(ctx);
    return -1;
  }
  vkGetDeviceQueue(ctx->device, ctx->queue_family, 0, &ctx->queue);

  VkCommandPoolCreateInfo cpi = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
  cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  cpi.queueFamilyIndex = ctx->queue_family;
  r = vkCreateCommandPool(ctx->device, &cpi, NULL, &ctx->pool);
  if (r != VK_SUCCESS) {
    set_err(err, "vkCreateCommandPool", r);
    avm_gpu_ctx_destroy(ctx);
    return -1;
  }

  VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
  r = vkCreateFence(ctx->device, &fci, NULL, &ctx->fence);
  if (r != VK_SUCCESS) {
    set_err(err, "vkCreateFence", r);
    avm_gpu_ctx_destroy(ctx);
    return -1;
  }

  *out = ctx;
  return 0;
}

void avm_gpu_ctx_destroy(avm_gpu_ctx *ctx) {
  if (!ctx) return;
  if (ctx->device) {
    vkDeviceWaitIdle(ctx->device);
    if (ctx->fence) vkDestroyFence(ctx->device, ctx->fence, NULL);
    if (ctx->pool) vkDestroyCommandPool(ctx->device, ctx->pool, NULL);
    vkDestroyDevice(ctx->device, NULL);
  }
  if (ctx->instance) vkDestroyInstance(ctx->instance, NULL);
  free(ctx);
}

const char *avm_gpu_device_name(const avm_gpu_ctx *ctx) { return ctx->name; }

int avm_gpu_buf_create(avm_gpu_ctx *ctx, VkDeviceSize size, avm_gpu_buf *buf,
                       char err[AVM_GPU_ERR_LEN]) {
  memset(buf, 0, sizeof(*buf));
  buf->size = size ? size : 4;

  VkBufferCreateInfo bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
  bi.size = buf->size;
  bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VkResult r = vkCreateBuffer(ctx->device, &bi, NULL, &buf->buffer);
  if (r != VK_SUCCESS) {
    set_err(err, "vkCreateBuffer", r);
    return -1;
  }

  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(ctx->device, buf->buffer, &req);
  const uint32_t mt = pick_memory_type(ctx->physical, req.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (mt == UINT32_MAX) {
    set_err(err, "no host-visible memory type", VK_ERROR_FEATURE_NOT_PRESENT);
    avm_gpu_buf_destroy(ctx, buf);
    return -1;
  }
  VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = mt;
  r = vkAllocateMemory(ctx->device, &ai, NULL, &buf->memory);
  if (r != VK_SUCCESS) {
    set_err(err, "vkAllocateMemory", r);
    avm_gpu_buf_destroy(ctx, buf);
    return -1;
  }
  r = vkBindBufferMemory(ctx->device, buf->buffer, buf->memory, 0);
  if (r != VK_SUCCESS) {
    set_err(err, "vkBindBufferMemory", r);
    avm_gpu_buf_destroy(ctx, buf);
    return -1;
  }
  r = vkMapMemory(ctx->device, buf->memory, 0, buf->size, 0, &buf->mapped);
  if (r != VK_SUCCESS) {
    set_err(err, "vkMapMemory", r);
    avm_gpu_buf_destroy(ctx, buf);
    return -1;
  }
  return 0;
}

void avm_gpu_buf_destroy(avm_gpu_ctx *ctx, avm_gpu_buf *buf) {
  if (!buf) return;
  /* Suballocations of a shared arena own nothing: memory == NULL. */
  if (buf->buffer && !buf->memory) {
    memset(buf, 0, sizeof(*buf));
    return;
  }
  if (buf->mapped && buf->memory) {
    vkUnmapMemory(ctx->device, buf->memory);
    buf->mapped = NULL;
  }
  if (buf->buffer) vkDestroyBuffer(ctx->device, buf->buffer, NULL);
  if (buf->memory) vkFreeMemory(ctx->device, buf->memory, NULL);
  buf->buffer = VK_NULL_HANDLE;
  buf->memory = VK_NULL_HANDLE;
  buf->size = 0;
}

int avm_gpu_submit_wait(avm_gpu_ctx *ctx, VkCommandBuffer cb,
                        char err[AVM_GPU_ERR_LEN]) {
  VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cb;
  VkResult r = vkResetFences(ctx->device, 1, &ctx->fence);
  if (r == VK_SUCCESS) r = vkQueueSubmit(ctx->queue, 1, &si, ctx->fence);
  if (r == VK_SUCCESS) {
    r = vkWaitForFences(ctx->device, 1, &ctx->fence, VK_TRUE, UINT64_MAX);
  }
  if (r != VK_SUCCESS) set_err(err, "queue submit/wait", r);
  return r == VK_SUCCESS ? 0 : -1;
}

VkDevice avm_gpu_device(const avm_gpu_ctx *ctx) { return ctx->device; }
VkPhysicalDevice avm_gpu_physical_device(const avm_gpu_ctx *ctx) {
  return ctx->physical;
}
VkCommandPool avm_gpu_command_pool(const avm_gpu_ctx *ctx) { return ctx->pool; }
