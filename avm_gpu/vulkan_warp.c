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

#include "avm_gpu/vulkan_warp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "avm_gpu/pipeline.h"

#ifndef AVM_GPU_WARP_SPV_PATH
#define AVM_GPU_WARP_SPV_PATH "warp.spv"
#endif

struct avm_gpu_warp {
  avm_gpu_pipeline *pipe;
  avm_gpu_buf ref, pred, filt;
  VkDeviceSize ref_cap, pred_cap;
  uint32_t *expand;
  size_t expand_cap;
};

typedef struct push_constants {
  uint32_t width, height, stride, p_col, p_row, p_width, p_height, p_stride,
      bd, round_0;
  int32_t m0, m1, m2, m3, m4, m5;
  int32_t alpha, beta, gamma, delta;
  uint32_t pad0, pad1;
} push_constants;

int avm_gpu_warp_create(avm_gpu_ctx *ctx, avm_gpu_warp **out,
                        char err[AVM_GPU_ERR_LEN]) {
  *out = NULL;
  avm_gpu_warp *w = calloc(1, sizeof(*w));
  if (!w) {
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    return -1;
  }
  if (avm_gpu_pipeline_create(ctx, AVM_GPU_WARP_SPV_PATH, "AVM_GPU_WARP_SPV",
                              3, sizeof(push_constants), &w->pipe,
                              err) != 0) {
    free(w);
    return -1;
  }
  *out = w;
  return 0;
}

void avm_gpu_warp_destroy(avm_gpu_ctx *ctx, avm_gpu_warp *w) {
  if (!w) return;
  avm_gpu_pipeline_destroy(ctx, w->pipe);
  avm_gpu_buf_destroy(ctx, &w->ref);
  avm_gpu_buf_destroy(ctx, &w->pred);
  avm_gpu_buf_destroy(ctx, &w->filt);
  free(w->expand);
  free(w);
}

int avm_gpu_warp_affine(avm_gpu_ctx *ctx, avm_gpu_warp *w,
                        const int32_t mat[6], const uint16_t *ref, int width,
                        int height, int stride, uint16_t *pred, int p_col,
                        int p_row, int p_width, int p_height, int p_stride,
                        int bd, int round_0, int alpha, int beta, int gamma,
                        int delta, const int16_t *filter,
                        char err[AVM_GPU_ERR_LEN]) {
  if (!mat || !ref || !pred || !filter) {
    snprintf(err, AVM_GPU_ERR_LEN, "invalid arguments");
    return -1;
  }
  if (p_width <= 0 || p_height <= 0 || (p_width & 7) || (p_height & 7)) {
    snprintf(err, AVM_GPU_ERR_LEN, "unsupported warp size %dx%d", p_width,
             p_height);
    return -1;
  }
  if (stride < width || p_stride < p_width) {
    snprintf(err, AVM_GPU_ERR_LEN, "stride too small");
    return -1;
  }

  const size_t ref_n = (size_t)stride * height;
  const size_t pred_n = (size_t)p_stride * (p_row + p_height);
  if (ref_n > w->expand_cap) {
    free(w->expand);
    w->expand = malloc(ref_n * sizeof(*w->expand));
    w->expand_cap = w->expand ? ref_n : 0;
    if (!w->expand) {
      snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
      return -1;
    }
  }

  if (w->ref_cap < (VkDeviceSize)ref_n * 4) {
    avm_gpu_buf_destroy(ctx, &w->ref);
    if (avm_gpu_buf_create(ctx, (VkDeviceSize)ref_n * 4, &w->ref, err) != 0) {
      return -1;
    }
    w->ref_cap = w->ref.size;
  }
  if (w->pred_cap < (VkDeviceSize)pred_n * 4) {
    avm_gpu_buf_destroy(ctx, &w->pred);
    if (avm_gpu_buf_create(ctx, (VkDeviceSize)pred_n * 4, &w->pred, err) != 0) {
      return -1;
    }
    w->pred_cap = w->pred.size;
  }
  if (!w->filt.buffer) {
    if (avm_gpu_buf_create(ctx, AVM_GPU_WARP_FILTER_ROWS * 8 * 4, &w->filt,
                           err) != 0) {
      return -1;
    }
  }

  for (size_t i = 0; i < ref_n; i++) w->expand[i] = ref[i];
  memcpy(w->ref.mapped, w->expand, ref_n * 4);
  uint32_t *pp = w->pred.mapped;
  for (size_t i = 0; i < pred_n; i++) pp[i] = pred[i];
  int32_t *fp = w->filt.mapped;
  for (int i = 0; i < AVM_GPU_WARP_FILTER_ROWS * 8; i++) fp[i] = filter[i];

  const avm_gpu_buf bufs[3] = { w->ref, w->pred, w->filt };
  push_constants pc = { (uint32_t)width,  (uint32_t)height, (uint32_t)stride,
                        (uint32_t)p_col,  (uint32_t)p_row,  (uint32_t)p_width,
                        (uint32_t)p_height, (uint32_t)p_stride,
                        (uint32_t)bd,     (uint32_t)round_0,
                        mat[0],           mat[1],
                        mat[2],           mat[3],
                        mat[4],           mat[5],
                        alpha,            beta,
                        gamma,            delta,
                        0,                0 };
  const uint32_t gx = (uint32_t)(p_width / 8);
  const uint32_t gy = (uint32_t)(p_height / 8);
  const int rc = avm_gpu_pipeline_dispatch_2d(ctx, w->pipe, bufs, 3, &pc,
                                              sizeof(pc), gx, gy, err);
  if (rc != 0) return rc;

  for (size_t i = 0; i < pred_n; i++) pred[i] = (uint16_t)pp[i];
  return 0;
}
