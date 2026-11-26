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

#include "avm_gpu/vulkan_sad.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "avm_gpu/arena.h"
#include "avm_gpu/pipeline.h"

#ifndef AVM_GPU_SAD_SPV_PATH
#define AVM_GPU_SAD_SPV_PATH "sad.spv"
#endif

struct avm_gpu_sad {
  avm_gpu_pipeline *pipe;
  avm_gpu_buf src, ref, pos, cost;
  VkDeviceSize src_cap, ref_cap, pos_cap, cost_cap;
  uint32_t *widen; /* scratch for the M0 u16 -> u32 expansion */
  size_t widen_cap;
  int src_stride, ref_stride;
  int packed; /* M1: two samples per word, no expansion on upload */
  avm_gpu_arena *arena; /* device-local arena (AV2_GPU_ARENA=1) */
};

/* Arena size in MiB, overridable with AV2_GPU_ARENA_MB. */
static VkDeviceSize arena_bytes_from_env(void) {
  const char *mb = getenv("AV2_GPU_ARENA_MB");
  const long v = mb && *mb ? strtol(mb, NULL, 10) : 64;
  return (VkDeviceSize)(v > 0 ? v : 64) << 20;
}

static avm_gpu_arena *maybe_create_arena(avm_gpu_ctx *ctx) {
  const char *env = getenv("AV2_GPU_ARENA");
  if (!env || !*env || !strcmp(env, "0")) return NULL;
  char err[AVM_GPU_ERR_LEN];
  avm_gpu_arena *arena = NULL;
  if (avm_gpu_arena_create(ctx, arena_bytes_from_env(), &arena, err) != 0) {
    fprintf(stderr, "avm_gpu: device-local arena disabled (%s)\n", err);
    return NULL;
  }
  if (getenv("AV2_GPU_ARENA_VERBOSE")) {
    fprintf(stderr, "avm_gpu: SAD arena %llu MiB\n",
            (unsigned long long)(arena_bytes_from_env() >> 20));
  }
  return arena;
}

typedef struct push_constants {
  uint32_t bw, bh, src_stride, ref_stride, count, packed;
} push_constants;

int avm_gpu_sad_create(avm_gpu_ctx *ctx, avm_gpu_sad **out,
                       char err[AVM_GPU_ERR_LEN]) {
  *out = NULL;
  avm_gpu_sad *sad = calloc(1, sizeof(*sad));
  if (!sad) {
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    return -1;
  }
  if (avm_gpu_pipeline_create(ctx, AVM_GPU_SAD_SPV_PATH, "AVM_GPU_SAD_SPV", 4,
                              sizeof(push_constants), &sad->pipe, err) != 0) {
    free(sad);
    return -1;
  }
  sad->arena = maybe_create_arena(ctx);
  *out = sad;
  return 0;
}

void avm_gpu_sad_destroy(avm_gpu_ctx *ctx, avm_gpu_sad *sad) {
  if (!sad) return;
  avm_gpu_pipeline_destroy(ctx, sad->pipe);
  avm_gpu_buf_destroy(ctx, &sad->src);
  avm_gpu_buf_destroy(ctx, &sad->ref);
  avm_gpu_buf_destroy(ctx, &sad->pos);
  avm_gpu_buf_destroy(ctx, &sad->cost);
  avm_gpu_arena_destroy(ctx, sad->arena);
  free(sad->widen);
  free(sad);
}

/* M0 uploads one 16-bit sample per 32-bit word; see avm_gpu/README.md.
 * The expansion buffer is reused across calls: motion search calls this
 * millions of times per encode. */
static int widen_into(avm_gpu_sad *sad, const uint16_t *src, size_t src_n,
                      const uint16_t *ref, size_t ref_n) {
  if (src_n + ref_n > sad->widen_cap) {
    free(sad->widen);
    sad->widen = malloc((src_n + ref_n) * sizeof(*sad->widen));
    sad->widen_cap = sad->widen ? src_n + ref_n : 0;
    if (!sad->widen) return -1;
  }
  uint32_t *sw = sad->widen;
  uint32_t *rw = sad->widen + src_n;
  for (size_t i = 0; i < src_n; i++) sw[i] = src[i];
  for (size_t i = 0; i < ref_n; i++) rw[i] = ref[i];
  return 0;
}

int avm_gpu_sad_set_planes(avm_gpu_ctx *ctx, avm_gpu_sad *sad,
                           const uint16_t *src, int src_stride, int src_rows,
                           const uint16_t *ref, int ref_stride, int ref_rows,
                           char err[AVM_GPU_ERR_LEN]) {
  if (!src || !ref || src_stride <= 0 || ref_stride <= 0 || src_rows <= 0 ||
      ref_rows <= 0) {
    snprintf(err, AVM_GPU_ERR_LEN, "invalid planes");
    return -1;
  }
  const size_t src_n = (size_t)src_stride * (size_t)src_rows;
  const size_t ref_n = (size_t)ref_stride * (size_t)ref_rows;

  /* M1 packed layout needs even strides so a row maps onto whole words. */
  const char *pack_env = getenv("AVM_GPU_PACK16");
  sad->packed = pack_env && *pack_env && *pack_env != '0' &&
                (src_stride % 2) == 0 && (ref_stride % 2) == 0;

  const VkDeviceSize src_bytes =
      sad->packed ? (VkDeviceSize)(src_n * 2) : (VkDeviceSize)(src_n * 4);
  const VkDeviceSize ref_bytes =
      sad->packed ? (VkDeviceSize)(ref_n * 2) : (VkDeviceSize)(ref_n * 4);
  if (avm_gpu_slot_reserve(ctx, sad->arena, &sad->src, &sad->src_cap, src_bytes,
                           err) ||
      avm_gpu_slot_reserve(ctx, sad->arena, &sad->ref, &sad->ref_cap, ref_bytes,
                           err)) {
    return -1;
  }
  const void *src_up = src;
  const void *ref_up = ref;
  if (!sad->packed) {
    if (widen_into(sad, src, src_n, ref, ref_n) != 0) {
      snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
      return -1;
    }
    src_up = sad->widen;
    ref_up = sad->widen + src_n;
  }
  /* Packed: two adjacent 16-bit samples already are one little-endian word. */
  if (avm_gpu_slot_upload(ctx, sad->arena, &sad->src, src_up, src_bytes, err) ||
      avm_gpu_slot_upload(ctx, sad->arena, &sad->ref, ref_up, ref_bytes, err)) {
    return -1;
  }
  sad->src_stride = src_stride;
  sad->ref_stride = ref_stride;
  return 0;
}

int avm_gpu_sad_run(avm_gpu_ctx *ctx, avm_gpu_sad *sad,
                    const avm_gpu_sad_pos *blocks, int n, int bw, int bh,
                    uint32_t *cost, char err[AVM_GPU_ERR_LEN]) {
  if (n <= 0) return 0;
  if (sad->src_stride <= 0 || sad->ref_stride <= 0) {
    snprintf(err, AVM_GPU_ERR_LEN, "planes not set");
    return -1;
  }
  const VkDeviceSize pos_bytes = (VkDeviceSize)n * sizeof(*blocks);
  const VkDeviceSize cost_bytes = (VkDeviceSize)n * sizeof(*cost);
  if (avm_gpu_slot_reserve(ctx, sad->arena, &sad->pos, &sad->pos_cap,
                           pos_bytes, err) ||
      avm_gpu_slot_reserve(ctx, sad->arena, &sad->cost, &sad->cost_cap,
                           cost_bytes, err)) {
    return -1;
  }
  if (avm_gpu_slot_upload(ctx, sad->arena, &sad->pos, blocks, pos_bytes, err)) {
    return -1;
  }

  /* One workgroup per candidate: the whole batch rides one dispatch. */
  const avm_gpu_buf bufs[4] = { sad->src, sad->ref, sad->pos, sad->cost };
  push_constants pc = { (uint32_t)bw,   (uint32_t)bh,
                        (uint32_t)sad->src_stride, (uint32_t)sad->ref_stride,
                        (uint32_t)n,    (uint32_t)sad->packed };
  int rc = avm_gpu_pipeline_dispatch(ctx, sad->pipe, bufs, 4, &pc,
                                           sizeof(pc), (uint32_t)n, err);
  if (rc == 0) {
    rc = avm_gpu_slot_download(ctx, sad->arena, &sad->cost, cost, cost_bytes,
                               err);
  }
  return rc;
}
