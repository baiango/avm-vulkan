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
};

typedef struct push_constants {
  uint32_t bw, bh, src_stride, ref_stride, count;
} push_constants;

static int ensure_buf(avm_gpu_ctx *ctx, avm_gpu_buf *buf, VkDeviceSize *cap,
                      VkDeviceSize need, char err[AVM_GPU_ERR_LEN]) {
  if (*cap >= need) return 0;
  avm_gpu_buf_destroy(ctx, buf);
  if (avm_gpu_buf_create(ctx, need, buf, err) != 0) return -1;
  *cap = buf->size;
  return 0;
}

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
  if (widen_into(sad, src, src_n, ref, ref_n) != 0) {
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    return -1;
  }
  const VkDeviceSize src_bytes = (VkDeviceSize)src_n * 4;
  const VkDeviceSize ref_bytes = (VkDeviceSize)ref_n * 4;
  if (ensure_buf(ctx, &sad->src, &sad->src_cap, src_bytes, err) ||
      ensure_buf(ctx, &sad->ref, &sad->ref_cap, ref_bytes, err)) {
    return -1;
  }
  memcpy(sad->src.mapped, sad->widen, (size_t)src_bytes);
  memcpy(sad->ref.mapped, sad->widen + src_n, (size_t)ref_bytes);
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
  if (ensure_buf(ctx, &sad->pos, &sad->pos_cap, pos_bytes, err) ||
      ensure_buf(ctx, &sad->cost, &sad->cost_cap, cost_bytes, err)) {
    return -1;
  }
  memcpy(sad->pos.mapped, blocks, (size_t)pos_bytes);

  /* One workgroup per candidate: the whole batch rides one dispatch. */
  const avm_gpu_buf bufs[4] = { sad->src, sad->ref, sad->pos, sad->cost };
  push_constants pc = { (uint32_t)bw, (uint32_t)bh, (uint32_t)sad->src_stride,
                        (uint32_t)sad->ref_stride, (uint32_t)n };
  const int rc = avm_gpu_pipeline_dispatch(ctx, sad->pipe, bufs, 4, &pc,
                                           sizeof(pc), (uint32_t)n, err);
  if (rc == 0) memcpy(cost, sad->cost.mapped, (size_t)cost_bytes);
  return rc;
}
