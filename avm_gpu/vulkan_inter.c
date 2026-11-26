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

#include "avm_gpu/vulkan_inter.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "avm_gpu/pipeline.h"

#ifndef AVM_GPU_INTER_SPV_PATH
#define AVM_GPU_INTER_SPV_PATH "inter.spv"
#endif

#define INTER_FILTER_ENTRIES (16 * 8)

struct avm_gpu_inter {
  avm_gpu_pipeline *pipe;
  avm_gpu_buf src, filt, dst;
  VkDeviceSize src_cap, filt_cap, dst_cap;
  uint32_t *expand;
  size_t expand_cap;
};

typedef struct push_constants {
  uint32_t src_stride, dst_stride, sx, sy, w, h, bd, x0, y0;
} push_constants;

static int ensure_buf(avm_gpu_ctx *ctx, avm_gpu_buf *buf, VkDeviceSize *cap,
                      VkDeviceSize need, char err[AVM_GPU_ERR_LEN]) {
  if (*cap >= need) return 0;
  avm_gpu_buf_destroy(ctx, buf);
  if (avm_gpu_buf_create(ctx, need, buf, err) != 0) return -1;
  *cap = buf->size;
  return 0;
}

int avm_gpu_inter_create(avm_gpu_ctx *ctx, avm_gpu_inter **out,
                         char err[AVM_GPU_ERR_LEN]) {
  *out = NULL;
  avm_gpu_inter *in = calloc(1, sizeof(*in));
  if (!in) {
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    return -1;
  }
  if (avm_gpu_pipeline_create(ctx, AVM_GPU_INTER_SPV_PATH, "AVM_GPU_INTER_SPV",
                              3, sizeof(push_constants), &in->pipe,
                              err) != 0) {
    free(in);
    return -1;
  }
  *out = in;
  return 0;
}

void avm_gpu_inter_destroy(avm_gpu_ctx *ctx, avm_gpu_inter *in) {
  if (!in) return;
  avm_gpu_pipeline_destroy(ctx, in->pipe);
  avm_gpu_buf_destroy(ctx, &in->src);
  avm_gpu_buf_destroy(ctx, &in->filt);
  avm_gpu_buf_destroy(ctx, &in->dst);
  free(in->expand);
  free(in);
}

int avm_gpu_inter_convolve(avm_gpu_ctx *ctx, avm_gpu_inter *in,
                           const uint16_t *src, int src_stride, int src_rows,
                           int sx, int sy, const int16_t *filter, int x0_q4,
                           int y0_q4, int w, int h, int bd, uint16_t *dst,
                           int dst_stride, char err[AVM_GPU_ERR_LEN]) {
  if (!src || !filter || !dst) {
    snprintf(err, AVM_GPU_ERR_LEN, "invalid arguments");
    return -1;
  }
  if (w <= 0 || h <= 0 || w > 64 || h > 64) {
    snprintf(err, AVM_GPU_ERR_LEN, "unsupported block size %dx%d", w, h);
    return -1;
  }
  if (x0_q4 < 0 || x0_q4 > 15 || y0_q4 < 0 || y0_q4 > 15) {
    snprintf(err, AVM_GPU_ERR_LEN, "subpel position out of range");
    return -1;
  }
  if (sx < 3 || sy < 3 || sx + w + 4 > src_stride || sy + h + 4 > src_rows) {
    snprintf(err, AVM_GPU_ERR_LEN, "block overlaps the filter border");
    return -1;
  }

  const size_t pixels = (size_t)src_stride * (size_t)src_rows;
  if (pixels > in->expand_cap) {
    free(in->expand);
    in->expand = malloc(pixels * sizeof(*in->expand));
    in->expand_cap = in->expand ? pixels : 0;
    if (!in->expand) {
      snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
      return -1;
    }
  }
  for (size_t i = 0; i < pixels; i++) in->expand[i] = src[i];

  if (ensure_buf(ctx, &in->src, &in->src_cap, (VkDeviceSize)pixels * 4, err) ||
      ensure_buf(ctx, &in->filt, &in->filt_cap,
                 (VkDeviceSize)INTER_FILTER_ENTRIES * 4, err) ||
      ensure_buf(ctx, &in->dst, &in->dst_cap,
                 (VkDeviceSize)dst_stride * h * 4, err)) {
    return -1;
  }
  memcpy(in->src.mapped, in->expand, (size_t)pixels * 4);
  uint32_t *fp = in->filt.mapped;
  for (int i = 0; i < INTER_FILTER_ENTRIES; i++) fp[i] = (uint32_t)filter[i];
  memset(in->dst.mapped, 0, (size_t)dst_stride * h * 4);

  const avm_gpu_buf bufs[3] = { in->src, in->filt, in->dst };
  push_constants pc = { (uint32_t)src_stride,
                        (uint32_t)dst_stride,
                        (uint32_t)sx,
                        (uint32_t)sy,
                        (uint32_t)w,
                        (uint32_t)h,
                        (uint32_t)bd,
                        (uint32_t)x0_q4,
                        (uint32_t)y0_q4 };
  const uint32_t gx = (uint32_t)((w + 7) / 8);
  const uint32_t gy = (uint32_t)((h + 7) / 8);
  const int rc = avm_gpu_pipeline_dispatch_2d(ctx, in->pipe, bufs, 3, &pc,
                                              sizeof(pc), gx, gy, err);
  if (rc != 0) return rc;

  const uint32_t *res = in->dst.mapped;
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      dst[(size_t)y * dst_stride + x] = (uint16_t)res[y * dst_stride + x];
    }
  }
  return 0;
}
