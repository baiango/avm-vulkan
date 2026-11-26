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

#include "avm_gpu/vulkan_intra.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "avm_gpu/pipeline.h"

#ifndef AVM_GPU_INTRA_SPV_PATH
#define AVM_GPU_INTRA_SPV_PATH "intra.spv"
#endif

#define INTRA_MAX_DIM 64
#define INTRA_WORKGROUP 64

struct avm_gpu_intra {
  avm_gpu_pipeline *pipe;
  avm_gpu_buf above, left, dst;
  VkDeviceSize above_cap, left_cap, dst_cap;
  uint32_t *expand;
  size_t expand_cap;
};

typedef struct push_constants {
  uint32_t bw, bh, mode, bd, stride, pad;
} push_constants;

static int ensure_buf(avm_gpu_ctx *ctx, avm_gpu_buf *buf, VkDeviceSize *cap,
                      VkDeviceSize need, char err[AVM_GPU_ERR_LEN]) {
  if (*cap >= need) return 0;
  avm_gpu_buf_destroy(ctx, buf);
  if (avm_gpu_buf_create(ctx, need, buf, err) != 0) return -1;
  *cap = buf->size;
  return 0;
}

int avm_gpu_intra_create(avm_gpu_ctx *ctx, avm_gpu_intra **out,
                         char err[AVM_GPU_ERR_LEN]) {
  *out = NULL;
  avm_gpu_intra *in = calloc(1, sizeof(*in));
  if (!in) {
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    return -1;
  }
  if (avm_gpu_pipeline_create(ctx, AVM_GPU_INTRA_SPV_PATH, "AVM_GPU_INTRA_SPV",
                              3, sizeof(push_constants), &in->pipe,
                              err) != 0) {
    free(in);
    return -1;
  }
  *out = in;
  return 0;
}

void avm_gpu_intra_destroy(avm_gpu_ctx *ctx, avm_gpu_intra *in) {
  if (!in) return;
  avm_gpu_pipeline_destroy(ctx, in->pipe);
  avm_gpu_buf_destroy(ctx, &in->above);
  avm_gpu_buf_destroy(ctx, &in->left);
  avm_gpu_buf_destroy(ctx, &in->dst);
  free(in->expand);
  free(in);
}

int avm_gpu_intra_auto_mode(int has_above, int has_left) {
  if (!has_above && !has_left) return AVM_GPU_INTRA_DC_128;
  if (!has_above) return AVM_GPU_INTRA_DC_LEFT;
  if (!has_left) return AVM_GPU_INTRA_DC_TOP;
  return AVM_GPU_INTRA_DC;
}

void avm_gpu_intra_edges(const uint16_t *plane, int stride, int x, int y,
                         int bw, int bh, int frame_w, int frame_h, int bd,
                         int has_above, int has_left, uint16_t *above,
                         uint16_t *left) {
  const uint16_t base = (uint16_t)(1 << (bd - 1));
  for (int i = 0; i < bw; i++) {
    if (!has_above) {
      above[i] = base;
    } else {
      int px = x + i;
      if (px > frame_w - 1) px = frame_w - 1;
      above[i] = plane[(y - 1) * stride + px];
    }
  }
  for (int j = 0; j < bh; j++) {
    if (!has_left) {
      left[j] = base;
    } else {
      int py = y + j;
      if (py > frame_h - 1) py = frame_h - 1;
      left[j] = plane[py * stride + (x - 1)];
    }
  }
}

int avm_gpu_intra_predict(avm_gpu_ctx *ctx, avm_gpu_intra *in,
                          const uint16_t *above, const uint16_t *left, int bw,
                          int bh, int mode, int bd, uint16_t *dst, int stride,
                          char err[AVM_GPU_ERR_LEN]) {
  if (bw <= 0 || bh <= 0 || bw > INTRA_MAX_DIM || bh > INTRA_MAX_DIM) {
    snprintf(err, AVM_GPU_ERR_LEN, "unsupported block size %dx%d", bw, bh);
    return -1;
  }
  if (mode < AVM_GPU_INTRA_DC || mode > AVM_GPU_INTRA_DC_128) {
    snprintf(err, AVM_GPU_ERR_LEN, "unsupported intra mode %d", mode);
    return -1;
  }
  /* V needs the above row, H needs the left column; the DC variants only need
   * whichever edge the host prepared (missing ones are filled with base). */
  const uint16_t dummy = (uint16_t)(1 << (bd - 1));
  const int need_above = mode != AVM_GPU_INTRA_H && mode != AVM_GPU_INTRA_DC_LEFT &&
                         mode != AVM_GPU_INTRA_DC_128;
  const int need_left = mode != AVM_GPU_INTRA_V && mode != AVM_GPU_INTRA_DC_TOP &&
                        mode != AVM_GPU_INTRA_DC_128;

  const size_t pixels = (size_t)stride * bh;
  if (pixels * 4 > in->expand_cap) {
    free(in->expand);
    in->expand = malloc(pixels * sizeof(*in->expand));
    in->expand_cap = in->expand ? pixels : 0;
    if (!in->expand) {
      snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
      return -1;
    }
  }

  if (ensure_buf(ctx, &in->above, &in->above_cap, (VkDeviceSize)bw * 4, err) ||
      ensure_buf(ctx, &in->left, &in->left_cap, (VkDeviceSize)bh * 4, err) ||
      ensure_buf(ctx, &in->dst, &in->dst_cap, (VkDeviceSize)pixels * 4, err)) {
    return -1;
  }
  uint32_t *ab = in->above.mapped;
  uint32_t *lf = in->left.mapped;
  for (int i = 0; i < bw; i++) ab[i] = need_above ? above[i] : dummy;
  for (int i = 0; i < bh; i++) lf[i] = need_left ? left[i] : dummy;
  memset(in->dst.mapped, 0, (size_t)pixels * 4);

  const avm_gpu_buf bufs[3] = { in->above, in->left, in->dst };
  push_constants pc = { (uint32_t)bw, (uint32_t)bh, (uint32_t)mode,
                        (uint32_t)bd, (uint32_t)stride, 0 };
  const int rc = avm_gpu_pipeline_dispatch(ctx, in->pipe, bufs, 3, &pc,
                                           sizeof(pc), 1, err);
  if (rc != 0) return rc;

  const uint32_t *res = in->dst.mapped;
  for (int y = 0; y < bh; y++) {
    for (int x = 0; x < bw; x++) dst[(size_t)y * stride + x] = res[y * stride + x];
  }
  return 0;
}
