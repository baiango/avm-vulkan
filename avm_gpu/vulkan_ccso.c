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

#include "avm_gpu/vulkan_ccso.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "avm_gpu/pipeline.h"

#ifndef AVM_GPU_CCSO_SPV_PATH
#define AVM_GPU_CCSO_SPV_PATH "ccso.spv"
#endif

struct avm_gpu_ccso {
  avm_gpu_pipeline *pipe;
  avm_gpu_buf src, dst, lut;
  VkDeviceSize src_cap, dst_cap;
  uint32_t *expand;
  size_t expand_cap;
};

typedef struct push_constants {
  uint32_t src_stride, dst_stride, x0, y0, blk_w, blk_h, pic_w, pic_h, hscale,
      vscale;
  int32_t thr, neg_thr, off0, off1;
  uint32_t max_val, single_band, shift_bits, edge_clf, bo_only, pad;
} push_constants;

int avm_gpu_ccso_create(avm_gpu_ctx *ctx, avm_gpu_ccso **out,
                        char err[AVM_GPU_ERR_LEN]) {
  *out = NULL;
  avm_gpu_ccso *cc = calloc(1, sizeof(*cc));
  if (!cc) {
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    return -1;
  }
  if (avm_gpu_pipeline_create(ctx, AVM_GPU_CCSO_SPV_PATH, "AVM_GPU_CCSO_SPV",
                              3, sizeof(push_constants), &cc->pipe,
                              err) != 0) {
    free(cc);
    return -1;
  }
  *out = cc;
  return 0;
}

void avm_gpu_ccso_destroy(avm_gpu_ctx *ctx, avm_gpu_ccso *cc) {
  if (!cc) return;
  avm_gpu_pipeline_destroy(ctx, cc->pipe);
  avm_gpu_buf_destroy(ctx, &cc->src);
  avm_gpu_buf_destroy(ctx, &cc->dst);
  avm_gpu_buf_destroy(ctx, &cc->lut);
  free(cc->expand);
  free(cc);
}

int avm_gpu_ccso_block(avm_gpu_ctx *ctx, avm_gpu_ccso *cc,
                       const uint16_t *src_y, int src_stride,
                       const uint16_t *dst, int dst_stride, int pic_w,
                       int pic_h, int x, int y, const int8_t *offset_buf,
                       int src_loc_0, int src_loc_1, int y_uv_hscale,
                       int y_uv_vscale, int thr, int neg_thr, int max_val,
                       int blk_w, int blk_h, int single_band, int shift_bits,
                       int edge_clf, int bo_only, uint16_t *out,
                       char err[AVM_GPU_ERR_LEN]) {
  if (!src_y || !dst || !offset_buf || !out) {
    snprintf(err, AVM_GPU_ERR_LEN, "invalid arguments");
    return -1;
  }
  const size_t src_pixels = (size_t)src_stride * (size_t)pic_h;
  const size_t dst_pixels = (size_t)dst_stride * (size_t)pic_h;

  if (src_pixels > cc->expand_cap) {
    free(cc->expand);
    cc->expand = malloc(src_pixels * sizeof(*cc->expand));
    cc->expand_cap = cc->expand ? src_pixels : 0;
    if (!cc->expand) {
      snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
      return -1;
    }
  }

  if (cc->src_cap < (VkDeviceSize)src_pixels * 4) {
    avm_gpu_buf_destroy(ctx, &cc->src);
    if (avm_gpu_buf_create(ctx, (VkDeviceSize)src_pixels * 4, &cc->src,
                           err) != 0) {
      return -1;
    }
    cc->src_cap = cc->src.size;
  }
  if (cc->dst_cap < (VkDeviceSize)dst_pixels * 4) {
    avm_gpu_buf_destroy(ctx, &cc->dst);
    if (avm_gpu_buf_create(ctx, (VkDeviceSize)dst_pixels * 4, &cc->dst,
                           err) != 0) {
      return -1;
    }
    cc->dst_cap = cc->dst.size;
  }
  if (!cc->lut.buffer) {
    if (avm_gpu_buf_create(ctx, AVM_GPU_CCSO_LUT_ENTRIES * 4, &cc->lut,
                           err) != 0) {
      return -1;
    }
  }

  for (size_t i = 0; i < src_pixels; i++) cc->expand[i] = src_y[i];
  memcpy(cc->src.mapped, cc->expand, src_pixels * 4);
  uint32_t *dp = cc->dst.mapped;
  for (size_t i = 0; i < dst_pixels; i++) dp[i] = dst[i];
  uint32_t *lp = cc->lut.mapped;
  for (int i = 0; i < AVM_GPU_CCSO_LUT_ENTRIES; i++) lp[i] = (uint32_t)offset_buf[i];

  const avm_gpu_buf bufs[3] = { cc->src, cc->dst, cc->lut };
  push_constants pc = { (uint32_t)src_stride,
                        (uint32_t)dst_stride,
                        (uint32_t)x,
                        (uint32_t)y,
                        (uint32_t)blk_w,
                        (uint32_t)blk_h,
                        (uint32_t)pic_w,
                        (uint32_t)pic_h,
                        (uint32_t)y_uv_hscale,
                        (uint32_t)y_uv_vscale,
                        (int32_t)thr,
                        (int32_t)neg_thr,
                        (int32_t)src_loc_0,
                        (int32_t)src_loc_1,
                        (uint32_t)max_val,
                        (uint32_t)(single_band ? 1 : 0),
                        (uint32_t)shift_bits,
                        (uint32_t)edge_clf,
                        (uint32_t)(bo_only ? 1 : 0),
                        0 };
  const uint32_t gx = (uint32_t)((blk_w + 7) / 8);
  const uint32_t gy = (uint32_t)((blk_h + 7) / 8);
  const int rc = avm_gpu_pipeline_dispatch_2d(ctx, cc->pipe, bufs, 3, &pc,
                                              sizeof(pc), gx, gy, err);
  if (rc != 0) return rc;

  for (size_t i = 0; i < dst_pixels; i++) out[i] = (uint16_t)dp[i];
  return 0;
}
