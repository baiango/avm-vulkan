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

#include "avm_gpu/vulkan_txfm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "avm_gpu/pipeline.h"

#ifndef AVM_GPU_TXFM_SPV_PATH
#define AVM_GPU_TXFM_SPV_PATH "txfm.spv"
#endif

#define TXFM_WORKGROUP 64

struct avm_gpu_txfm {
  avm_gpu_pipeline *pipe;
  avm_gpu_buf src, pos, out;
  VkDeviceSize src_cap, pos_cap, out_cap;
  int32_t *expand;
  size_t expand_cap;
  int have_src;
  int stride;
};

typedef struct push_constants {
  uint32_t stride, size, count, pad;
} push_constants;

static int ensure_buf(avm_gpu_ctx *ctx, avm_gpu_buf *buf, VkDeviceSize *cap,
                      VkDeviceSize need, char err[AVM_GPU_ERR_LEN]) {
  if (*cap >= need) return 0;
  avm_gpu_buf_destroy(ctx, buf);
  if (avm_gpu_buf_create(ctx, need, buf, err) != 0) return -1;
  *cap = buf->size;
  return 0;
}

int avm_gpu_txfm_create(avm_gpu_ctx *ctx, avm_gpu_txfm **out,
                        char err[AVM_GPU_ERR_LEN]) {
  *out = NULL;
  avm_gpu_txfm *t = calloc(1, sizeof(*t));
  if (!t) {
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    return -1;
  }
  if (avm_gpu_pipeline_create(ctx, AVM_GPU_TXFM_SPV_PATH, "AVM_GPU_TXFM_SPV",
                              3, sizeof(push_constants), &t->pipe,
                              err) != 0) {
    free(t);
    return -1;
  }
  *out = t;
  return 0;
}

void avm_gpu_txfm_destroy(avm_gpu_ctx *ctx, avm_gpu_txfm *t) {
  if (!t) return;
  avm_gpu_pipeline_destroy(ctx, t->pipe);
  avm_gpu_buf_destroy(ctx, &t->src);
  avm_gpu_buf_destroy(ctx, &t->pos);
  avm_gpu_buf_destroy(ctx, &t->out);
  free(t->expand);
  free(t);
}

int avm_gpu_txfm_set_source(avm_gpu_ctx *ctx, avm_gpu_txfm *t,
                            const int16_t *src, int stride, int rows,
                            char err[AVM_GPU_ERR_LEN]) {
  if (!src || stride <= 0 || rows <= 0) {
    snprintf(err, AVM_GPU_ERR_LEN, "invalid source");
    return -1;
  }
  const size_t n = (size_t)stride * (size_t)rows;
  if (n > t->expand_cap) {
    free(t->expand);
    t->expand = malloc(n * sizeof(*t->expand));
    t->expand_cap = t->expand ? n : 0;
    if (!t->expand) {
      snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
      return -1;
    }
  }
  for (size_t i = 0; i < n; i++) t->expand[i] = src[i];

  if (ensure_buf(ctx, &t->src, &t->src_cap, (VkDeviceSize)n * 4, err)) return -1;
  memcpy(t->src.mapped, t->expand, n * 4);
  t->stride = stride;
  t->have_src = 1;
  return 0;
}

int avm_gpu_txfm_fdct(avm_gpu_ctx *ctx, avm_gpu_txfm *t,
                      const uint32_t *offsets, int n, int size,
                      int32_t *coeffs, char err[AVM_GPU_ERR_LEN]) {
  if (n <= 0) return 0;
  if (!t->have_src) {
    snprintf(err, AVM_GPU_ERR_LEN, "source not set");
    return -1;
  }
  if (size != 4 && size != 8) {
    snprintf(err, AVM_GPU_ERR_LEN, "unsupported transform size %d", size);
    return -1;
  }

  const VkDeviceSize pos_bytes = (VkDeviceSize)n * sizeof(*offsets);
  const VkDeviceSize out_bytes =
      (VkDeviceSize)n * (VkDeviceSize)size * (VkDeviceSize)size * 4;
  if (ensure_buf(ctx, &t->pos, &t->pos_cap, pos_bytes, err) ||
      ensure_buf(ctx, &t->out, &t->out_cap, out_bytes, err)) {
    return -1;
  }
  memcpy(t->pos.mapped, offsets, (size_t)pos_bytes);

  /* One invocation per block; the pipeline dispatches whole workgroups, so
   * round up and let the shader discard the tail. */
  const avm_gpu_buf bufs[3] = { t->src, t->pos, t->out };
  push_constants pc = { (uint32_t)t->stride, (uint32_t)size, (uint32_t)n, 0 };
  const uint32_t groups = (uint32_t)((n + TXFM_WORKGROUP - 1) / TXFM_WORKGROUP);
  const int rc =
      avm_gpu_pipeline_dispatch(ctx, t->pipe, bufs, 3, &pc, sizeof(pc), groups,
                                err);
  if (rc == 0) memcpy(coeffs, t->out.mapped, (size_t)out_bytes);
  return rc;
}
