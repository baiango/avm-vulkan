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
  int src_stride, ref_stride;
  int src_packed, ref_packed; /* M1: two samples per word, no expansion */
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
  free(sad);
}

/* M0 uploads one 16-bit sample per 32-bit word; see avm_gpu/README.md. */
static uint32_t *widen(const uint16_t *plane, int stride, int rows) {
  if (!plane || stride <= 0 || rows <= 0) return NULL;
  const size_t n = (size_t)stride * (size_t)rows;
  uint32_t *w = malloc(n * sizeof(*w));
  if (!w) return NULL;
  for (size_t i = 0; i < n; i++) w[i] = plane[i];
  return w;
}

/* M1 packed layout needs an even stride so a row maps onto whole words. */
static int packed_for(int stride) {
  const char *env = getenv("AVM_GPU_PACK16");
  return env && *env && *env != '0' && (stride % 2) == 0;
}

static int upload_plane(avm_gpu_ctx *ctx, avm_gpu_sad *sad,
                        const uint16_t *plane, int stride, int rows,
                        avm_gpu_buf *buf, VkDeviceSize *cap, int *buf_packed,
                        char err[AVM_GPU_ERR_LEN]) {
  if (!plane || stride <= 0 || rows <= 0) {
    snprintf(err, AVM_GPU_ERR_LEN, "invalid plane");
    return -1;
  }
  const int packed = packed_for(stride);
  const size_t n = (size_t)stride * (size_t)rows;
  const VkDeviceSize bytes = (VkDeviceSize)(packed ? n * 2 : n * 4);
  if (avm_gpu_slot_reserve(ctx, sad->arena, buf, cap, bytes, err) != 0) {
    return -1;
  }
  int rc;
  if (packed) {
    /* Two adjacent 16-bit samples already are one little-endian word. */
    rc = avm_gpu_slot_upload(ctx, sad->arena, buf, plane, bytes, err);
  } else {
    uint32_t *w = widen(plane, stride, rows);
    if (!w) {
      snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
      return -1;
    }
    rc = avm_gpu_slot_upload(ctx, sad->arena, buf, w, bytes, err);
    free(w);
  }
  if (rc == 0) *buf_packed = packed;
  return rc;
}

int avm_gpu_sad_set_src(avm_gpu_ctx *ctx, avm_gpu_sad *sad,
                        const uint16_t *src, int src_stride, int src_rows,
                        char err[AVM_GPU_ERR_LEN]) {
  if (upload_plane(ctx, sad, src, src_stride, src_rows, &sad->src,
                   &sad->src_cap, &sad->src_packed, err) != 0) {
    return -1;
  }
  sad->src_stride = src_stride;
  return 0;
}

int avm_gpu_sad_set_ref(avm_gpu_ctx *ctx, avm_gpu_sad *sad,
                        const uint16_t *ref, int ref_stride, int ref_rows,
                        char err[AVM_GPU_ERR_LEN]) {
  if (upload_plane(ctx, sad, ref, ref_stride, ref_rows, &sad->ref,
                   &sad->ref_cap, &sad->ref_packed, err) != 0) {
    return -1;
  }
  sad->ref_stride = ref_stride;
  return 0;
}

int avm_gpu_sad_set_planes(avm_gpu_ctx *ctx, avm_gpu_sad *sad,
                           const uint16_t *src, int src_stride, int src_rows,
                           const uint16_t *ref, int ref_stride, int ref_rows,
                           char err[AVM_GPU_ERR_LEN]) {
  if (avm_gpu_sad_set_src(ctx, sad, src, src_stride, src_rows, err) != 0) {
    return -1;
  }
  return avm_gpu_sad_set_ref(ctx, sad, ref, ref_stride, ref_rows, err);
}

int avm_gpu_sad_run(avm_gpu_ctx *ctx, avm_gpu_sad *sad,
                    const avm_gpu_sad_pos *blocks, int n, int bw, int bh,
                    uint32_t *cost, char err[AVM_GPU_ERR_LEN]) {
  if (n <= 0) return 0;
  if (sad->src_stride <= 0 || sad->ref_stride <= 0) {
    snprintf(err, AVM_GPU_ERR_LEN, "planes not set");
    return -1;
  }
  if (sad->src_packed != sad->ref_packed) {
    snprintf(err, AVM_GPU_ERR_LEN, "mixed upload layouts");
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
                        (uint32_t)n,    (uint32_t)sad->src_packed };
  int rc = avm_gpu_pipeline_dispatch(ctx, sad->pipe, bufs, 4, &pc,
                                           sizeof(pc), (uint32_t)n, err);
  if (rc == 0) {
    rc = avm_gpu_slot_download(ctx, sad->arena, &sad->cost, cost, cost_bytes,
                               err);
  }
  return rc;
}
