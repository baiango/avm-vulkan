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

#include "avm_gpu/arena.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARENA_ALIGN 64
#define ARENA_STAGING_MIN (1u << 20)

typedef struct arena_block {
  VkDeviceSize offset;
  VkDeviceSize size;
  int free;
} arena_block;

struct avm_gpu_arena {
  VkBuffer buffer;
  VkDeviceMemory memory;
  VkDeviceSize size;
  VkBuffer staging;
  VkDeviceMemory staging_memory;
  void *staging_map;
  VkDeviceSize staging_size;
  VkDeviceSize align;
  arena_block *blocks;
  int n_blocks;
  int cap_blocks;
  VkCommandBuffer cb;
};

/* Suballocations must satisfy the device's descriptor-offset alignment, which
 * is usually larger than a cache line (256 on many drivers). */
static VkDeviceSize align_up(VkDeviceSize v, VkDeviceSize align) {
  return (v + (align - 1)) & ~(align - 1);
}

static uint32_t find_memory_type(avm_gpu_ctx *ctx, uint32_t bits,
                                 VkMemoryPropertyFlags want) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(avm_gpu_physical_device(ctx), &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
    if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) {
      return i;
    }
  }
  return UINT32_MAX;
}

static int make_buffer(avm_gpu_ctx *ctx, VkDeviceSize size,
                       VkMemoryPropertyFlags props, VkBuffer *buffer,
                       VkDeviceMemory *memory, void **map,
                       char err[AVM_GPU_ERR_LEN]) {
  VkBufferCreateInfo bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
  bi.size = size;
  bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
             VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
             VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VkResult r = vkCreateBuffer(avm_gpu_device(ctx), &bi, NULL, buffer);
  if (r != VK_SUCCESS) {
    snprintf(err, AVM_GPU_ERR_LEN, "vkCreateBuffer (%d)", (int)r);
    return -1;
  }
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(avm_gpu_device(ctx), *buffer, &req);
  const uint32_t mt = find_memory_type(ctx, req.memoryTypeBits, props);
  if (mt == UINT32_MAX) {
    snprintf(err, AVM_GPU_ERR_LEN, "no suitable memory type");
    vkDestroyBuffer(avm_gpu_device(ctx), *buffer, NULL);
    return -1;
  }
  VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = mt;
  r = vkAllocateMemory(avm_gpu_device(ctx), &ai, NULL, memory);
  if (r != VK_SUCCESS) {
    snprintf(err, AVM_GPU_ERR_LEN, "vkAllocateMemory (%d)", (int)r);
    vkDestroyBuffer(avm_gpu_device(ctx), *buffer, NULL);
    return -1;
  }
  vkBindBufferMemory(avm_gpu_device(ctx), *buffer, *memory, 0);
  if (map) {
    r = vkMapMemory(avm_gpu_device(ctx), *memory, 0, size, 0, map);
    if (r != VK_SUCCESS) {
      snprintf(err, AVM_GPU_ERR_LEN, "vkMapMemory (%d)", (int)r);
      return -1;
    }
  }
  return 0;
}

int avm_gpu_arena_create(avm_gpu_ctx *ctx, VkDeviceSize bytes,
                         avm_gpu_arena **out, char err[AVM_GPU_ERR_LEN]) {
  *out = NULL;
  avm_gpu_arena *a = calloc(1, sizeof(*a));
  if (!a) {
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    return -1;
  }
  a->align = ARENA_ALIGN;
  {
    const VkDeviceSize dev = avm_gpu_storage_offset_alignment(ctx);
    if (dev > a->align) a->align = dev;
  }
  if (getenv("AV2_GPU_ARENA_VERBOSE")) {
    fprintf(stderr,
            "avm_gpu: arena %llu MiB, suballocation offset alignment %llu\n",
            (unsigned long long)(bytes / (1024u * 1024u)),
            (unsigned long long)a->align);
  }
  a->size = align_up(bytes, a->align);
  if (make_buffer(ctx, a->size, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &a->buffer,
                  &a->memory, NULL, err) != 0) {
    free(a);
    return -1;
  }
  a->staging_size = ARENA_STAGING_MIN;
  if (make_buffer(ctx, a->staging_size,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  &a->staging, &a->staging_memory, &a->staging_map, err) != 0) {
    vkDestroyBuffer(avm_gpu_device(ctx), a->buffer, NULL);
    vkFreeMemory(avm_gpu_device(ctx), a->memory, NULL);
    free(a);
    return -1;
  }

  VkCommandBufferAllocateInfo cbai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
  cbai.commandPool = avm_gpu_command_pool(ctx);
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkResult r = vkAllocateCommandBuffers(avm_gpu_device(ctx), &cbai, &a->cb);
  if (r != VK_SUCCESS) {
    snprintf(err, AVM_GPU_ERR_LEN, "vkAllocateCommandBuffers (%d)", (int)r);
    avm_gpu_arena_destroy(ctx, a);
    return -1;
  }

  a->cap_blocks = 8;
  a->blocks = calloc((size_t)a->cap_blocks, sizeof(*a->blocks));
  if (!a->blocks) {
    avm_gpu_arena_destroy(ctx, a);
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    return -1;
  }
  a->blocks[0] = (arena_block){ 0, a->size, 1 };
  a->n_blocks = 1;
  *out = a;
  return 0;
}

void avm_gpu_arena_destroy(avm_gpu_ctx *ctx, avm_gpu_arena *a) {
  if (!a) return;
  if (a->cb) vkFreeCommandBuffers(avm_gpu_device(ctx), avm_gpu_command_pool(ctx),
                                  1, &a->cb);
  if (a->staging_map) vkUnmapMemory(avm_gpu_device(ctx), a->staging_memory);
  if (a->staging) vkDestroyBuffer(avm_gpu_device(ctx), a->staging, NULL);
  if (a->staging_memory) vkFreeMemory(avm_gpu_device(ctx), a->staging_memory, NULL);
  if (a->buffer) vkDestroyBuffer(avm_gpu_device(ctx), a->buffer, NULL);
  if (a->memory) vkFreeMemory(avm_gpu_device(ctx), a->memory, NULL);
  free(a->blocks);
  free(a);
}

VkBuffer avm_gpu_arena_buffer(const avm_gpu_arena *a) { return a->buffer; }

int avm_gpu_arena_alloc(avm_gpu_ctx *ctx, avm_gpu_arena *a, VkDeviceSize size,
                        VkDeviceSize *offset, char err[AVM_GPU_ERR_LEN]) {
  (void)ctx;
  const VkDeviceSize need = align_up(size ? size : 4, a->align);
  for (int i = 0; i < a->n_blocks; i++) {
    if (!a->blocks[i].free || a->blocks[i].size < need) continue;
    if (a->blocks[i].size == need) {
      a->blocks[i].free = 0;
      *offset = a->blocks[i].offset;
      return 0;
    }
    /* Split. */
    if (a->n_blocks == a->cap_blocks) {
      const int cap = a->cap_blocks * 2;
      arena_block *nb = realloc(a->blocks, (size_t)cap * sizeof(*nb));
      if (!nb) {
        snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
        return -1;
      }
      a->blocks = nb;
      a->cap_blocks = cap;
    }
    memmove(&a->blocks[i + 2], &a->blocks[i + 1],
            (size_t)(a->n_blocks - i - 1) * sizeof(*a->blocks));
    a->n_blocks++;
    /* Re-read the block after the possible realloc above: a pointer taken
     * before it would dangle and corrupt the table (which produced
     * overlapping allocations). */
    arena_block *b = &a->blocks[i];
    arena_block *tail = &a->blocks[i + 1];
    tail->offset = b->offset + need;
    tail->size = b->size - need;
    tail->free = 1;
    b->size = need;
    b->free = 0;
    *offset = b->offset;
    return 0;
  }
  snprintf(err, AVM_GPU_ERR_LEN, "arena full (%llu bytes requested)",
           (unsigned long long)size);
  return -1;
}

void avm_gpu_arena_free(avm_gpu_arena *a, VkDeviceSize offset,
                        VkDeviceSize size) {
  if (!a || !size) return;
  for (int i = 0; i < a->n_blocks; i++) {
    if (a->blocks[i].offset == offset && !a->blocks[i].free) {
      a->blocks[i].free = 1;
      /* Coalesce with the following block. */
      if (i + 1 < a->n_blocks && a->blocks[i + 1].free) {
        a->blocks[i].size += a->blocks[i + 1].size;
        memmove(&a->blocks[i + 1], &a->blocks[i + 2],
                (size_t)(a->n_blocks - i - 2) * sizeof(*a->blocks));
        a->n_blocks--;
      }
      /* Coalesce with the preceding block. */
      if (i > 0 && a->blocks[i - 1].free) {
        a->blocks[i - 1].size += a->blocks[i].size;
        memmove(&a->blocks[i], &a->blocks[i + 1],
                (size_t)(a->n_blocks - i - 1) * sizeof(*a->blocks));
        a->n_blocks--;
      }
      return;
    }
  }
}

static int ensure_staging(avm_gpu_ctx *ctx, avm_gpu_arena *a, VkDeviceSize need,
                          char err[AVM_GPU_ERR_LEN]) {
  if (a->staging_size >= need) return 0;
  VkDeviceSize size = a->staging_size;
  while (size < need) size *= 2;
  if (a->staging_map) vkUnmapMemory(avm_gpu_device(ctx), a->staging_memory);
  vkDestroyBuffer(avm_gpu_device(ctx), a->staging, NULL);
  vkFreeMemory(avm_gpu_device(ctx), a->staging_memory, NULL);
  a->staging_map = NULL;
  if (make_buffer(ctx, size,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  &a->staging, &a->staging_memory, &a->staging_map,
                  err) != 0) {
    return -1;
  }
  a->staging_size = size;
  return 0;
}

static int copy_range(avm_gpu_ctx *ctx, avm_gpu_arena *a, VkBuffer src,
                      VkBuffer dst, VkDeviceSize src_off, VkDeviceSize dst_off,
                      VkDeviceSize size, char err[AVM_GPU_ERR_LEN]) {
  VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VkResult r = vkBeginCommandBuffer(a->cb, &bi);
  if (r != VK_SUCCESS) {
    snprintf(err, AVM_GPU_ERR_LEN, "vkBeginCommandBuffer (%d)", (int)r);
    return -1;
  }
  VkBufferCopy region = { src_off, dst_off, size };
  vkCmdCopyBuffer(a->cb, src, dst, 1, &region);
  vkEndCommandBuffer(a->cb);
  const int rc = avm_gpu_submit_wait(ctx, a->cb, err);
  vkResetCommandBuffer(a->cb, 0);
  return rc;
}

int avm_gpu_arena_upload(avm_gpu_ctx *ctx, avm_gpu_arena *a,
                         VkDeviceSize offset, const void *data,
                         VkDeviceSize size, char err[AVM_GPU_ERR_LEN]) {
  if (ensure_staging(ctx, a, size, err) != 0) return -1;
  memcpy(a->staging_map, data, (size_t)size);
  return copy_range(ctx, a, a->staging, a->buffer, 0, offset, size, err);
}

int avm_gpu_arena_download(avm_gpu_ctx *ctx, avm_gpu_arena *a,
                           VkDeviceSize offset, void *data, VkDeviceSize size,
                           char err[AVM_GPU_ERR_LEN]) {
  if (ensure_staging(ctx, a, size, err) != 0) return -1;
  if (copy_range(ctx, a, a->buffer, a->staging, offset, 0, size, err) != 0) {
    return -1;
  }
  memcpy(data, a->staging_map, (size_t)size);
  return 0;
}

/* ------------------------------------------------------------------ slots */

int avm_gpu_slot_reserve(avm_gpu_ctx *ctx, avm_gpu_arena *arena,
                         avm_gpu_buf *buf, VkDeviceSize *cap, VkDeviceSize need,
                         char err[AVM_GPU_ERR_LEN]) {
  if (need == 0) need = 4;
  if (arena && buf->buffer == avm_gpu_arena_buffer(arena) && *cap >= need) {
    return 0;
  }
  if (arena) {
    char aerr[AVM_GPU_ERR_LEN];
    VkDeviceSize offset = 0;
    if (avm_gpu_arena_alloc(ctx, arena, need, &offset, aerr) == 0) {
      if (buf->buffer == avm_gpu_arena_buffer(arena)) {
        avm_gpu_arena_free(arena, buf->offset, *cap);
      } else {
        avm_gpu_buf_destroy(ctx, buf);
      }
      memset(buf, 0, sizeof(*buf));
      buf->buffer = avm_gpu_arena_buffer(arena);
      buf->offset = offset;
      buf->size = need;
      *cap = need;
      return 0;
    }
    /* Arena exhausted or fragmented: fall back to a host-visible buffer
     * rather than failing the encode. */
    if (getenv("AV2_GPU_ARENA_VERBOSE")) {
      fprintf(stderr, "avm_gpu: arena slot fallback (%s)\n", aerr);
    }
  }
  if (*cap >= need && buf->buffer && buf->mapped) return 0;
  avm_gpu_buf_destroy(ctx, buf);
  if (avm_gpu_buf_create(ctx, need, buf, err) != 0) {
    *cap = 0;
    return -1;
  }
  *cap = buf->size;
  return 0;
}

int avm_gpu_slot_upload(avm_gpu_ctx *ctx, avm_gpu_arena *arena,
                        const avm_gpu_buf *buf, const void *data,
                        VkDeviceSize bytes, char err[AVM_GPU_ERR_LEN]) {
  if (arena && buf->buffer == avm_gpu_arena_buffer(arena)) {
    return avm_gpu_arena_upload(ctx, arena, buf->offset, data, bytes, err);
  }
  memcpy(buf->mapped, data, (size_t)bytes);
  return 0;
}

int avm_gpu_slot_download(avm_gpu_ctx *ctx, avm_gpu_arena *arena,
                          const avm_gpu_buf *buf, void *data, VkDeviceSize bytes,
                          char err[AVM_GPU_ERR_LEN]) {
  if (arena && buf->buffer == avm_gpu_arena_buffer(arena)) {
    return avm_gpu_arena_download(ctx, arena, buf->offset, data, bytes, err);
  }
  memcpy(data, buf->mapped, (size_t)bytes);
  return 0;
}

int avm_gpu_shadow_reserve(uint32_t **host, size_t *cap, VkDeviceSize bytes,
                           char err[AVM_GPU_ERR_LEN]) {
  if ((size_t)bytes <= *cap) return 0;
  free(*host);
  *host = malloc((size_t)bytes);
  *cap = *host ? (size_t)bytes : 0;
  if (!*host) {
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    return -1;
  }
  /* A fresh host-visible VkBuffer reads as zero; keep the shadow identical. */
  memset(*host, 0, (size_t)bytes);
  return 0;
}

int avm_gpu_shadow_push(avm_gpu_ctx *ctx, avm_gpu_arena *arena,
                        const avm_gpu_buf *buf, const uint32_t *host,
                        VkDeviceSize bytes, char err[AVM_GPU_ERR_LEN]) {
  if (buf->mapped) return 0;
  return avm_gpu_slot_upload(ctx, arena, buf, host, bytes, err);
}

int avm_gpu_shadow_pull(avm_gpu_ctx *ctx, avm_gpu_arena *arena,
                        const avm_gpu_buf *buf, uint32_t *host,
                        VkDeviceSize bytes, char err[AVM_GPU_ERR_LEN]) {
  if (buf->mapped) return 0;
  return avm_gpu_slot_download(ctx, arena, buf, host, bytes, err);
}
