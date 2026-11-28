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

#include "avm_gpu/arena.h"
#include "avm_gpu/pipeline.h"

#ifndef AVM_GPU_CCSO_SPV_PATH
#define AVM_GPU_CCSO_SPV_PATH "ccso.spv"
#endif

struct avm_gpu_ccso {
  avm_gpu_arena *arena; /* device-local arena when AV2_GPU_ARENA=1 */
  uint32_t *src_host, *dst_host, *lut_host;
  size_t src_host_cap, dst_host_cap, lut_host_cap;
  avm_gpu_pipeline *pipe;
  avm_gpu_buf src, dst, lut;
  VkDeviceSize src_cap, dst_cap, lut_cap;
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
  {
    if (avm_gpu_arena_wanted("ccso")) {
      const char *mb = getenv("AV2_GPU_ARENA_MB");
      const VkDeviceSize bytes =
          (VkDeviceSize)(mb && *mb ? atoi(mb) : 64) * 1024u * 1024u;
      char aerr[AVM_GPU_ERR_LEN];
      if (avm_gpu_arena_create(ctx, bytes, &cc->arena, aerr) != 0) {
        cc->arena = NULL;
      }
    }
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
  /* the host pointers alias the mapped buffers on the host-visible path */
  if (cc->src_host != cc->src.mapped) free(cc->src_host);
  if (cc->dst_host != cc->dst.mapped) free(cc->dst_host);
  if (cc->lut_host != cc->lut.mapped) free(cc->lut_host);
  avm_gpu_buf_destroy(ctx, &cc->src);
  avm_gpu_buf_destroy(ctx, &cc->dst);
  avm_gpu_buf_destroy(ctx, &cc->lut);
  avm_gpu_arena_destroy(ctx, cc->arena);
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

  const VkDeviceSize src_bytes = (VkDeviceSize)src_pixels * 4;
  const VkDeviceSize dst_bytes = (VkDeviceSize)dst_pixels * 4;
  const VkDeviceSize lut_bytes = AVM_GPU_CCSO_LUT_ENTRIES * 4;
  if (avm_gpu_slot_reserve(ctx, cc->arena, &cc->src, &cc->src_cap, src_bytes,
                           err) != 0 ||
      avm_gpu_slot_reserve(ctx, cc->arena, &cc->dst, &cc->dst_cap, dst_bytes,
                           err) != 0 ||
      avm_gpu_slot_reserve(ctx, cc->arena, &cc->lut, &cc->lut_cap, lut_bytes,
                           err) != 0) {
    return -1;
  }
  if (cc->src.mapped) {
    cc->src_host = cc->src.mapped;
  } else if (avm_gpu_shadow_reserve(&cc->src_host, &cc->src_host_cap, src_bytes,
                                    err) != 0) {
    return -1;
  }
  if (cc->dst.mapped) {
    cc->dst_host = cc->dst.mapped;
  } else if (avm_gpu_shadow_reserve(&cc->dst_host, &cc->dst_host_cap, dst_bytes,
                                    err) != 0) {
    return -1;
  }
  if (cc->lut.mapped) {
    cc->lut_host = cc->lut.mapped;
  } else if (avm_gpu_shadow_reserve(&cc->lut_host, &cc->lut_host_cap, lut_bytes,
                                    err) != 0) {
    return -1;
  }

  for (size_t i = 0; i < src_pixels; i++) cc->expand[i] = src_y[i];
  memcpy(cc->src_host, cc->expand, src_pixels * 4);
  uint32_t *dp = cc->dst_host;
  for (size_t i = 0; i < dst_pixels; i++) dp[i] = dst[i];
  uint32_t *lp = cc->lut_host;
  for (int i = 0; i < AVM_GPU_CCSO_LUT_ENTRIES; i++) lp[i] = (uint32_t)offset_buf[i];
  if (avm_gpu_shadow_push(ctx, cc->arena, &cc->src, cc->src_host, src_bytes,
                          err) != 0 ||
      avm_gpu_shadow_push(ctx, cc->arena, &cc->dst, cc->dst_host, dst_bytes,
                          err) != 0 ||
      avm_gpu_shadow_push(ctx, cc->arena, &cc->lut, cc->lut_host, lut_bytes,
                          err) != 0) {
    return -1;
  }

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

  if (avm_gpu_shadow_pull(ctx, cc->arena, &cc->dst, cc->dst_host, dst_bytes,
                          err) != 0) {
    return -1;
  }
  for (size_t i = 0; i < dst_pixels; i++) out[i] = (uint16_t)dp[i];
  return 0;
}

int avm_gpu_ccso_block_window(avm_gpu_ctx *ctx, avm_gpu_ccso *cc,
                              const uint16_t *src_y, int src_stride,
                              uint16_t *dst, int dst_stride, int pic_w,
                              int pic_h, int x, int y,
                              const int8_t *offset_buf, int src_loc_0,
                              int src_loc_1, int y_uv_hscale, int y_uv_vscale,
                              int thr, int neg_thr, int max_val, int blk_w,
                              int blk_h, int single_band, int shift_bits,
                              int edge_clf, int bo_only,
                              char err[AVM_GPU_ERR_LEN]) {
  const int dh = 2, dv = 1;
  if (!src_y || !dst || x < dh || y < dv || src_stride <= 0 ||
      dst_stride <= 0) {
    snprintf(err, AVM_GPU_ERR_LEN, "no pad on the left or top edge");
    return -1;
  }
  const int x_room = pic_w - x, y_room = pic_h - y;
  const int x_end = x_room < blk_w ? x_room : blk_w;
  const int y_end = y_room < blk_h ? y_room : blk_h;
  if (x_end <= 0 || y_end <= 0) {
    snprintf(err, AVM_GPU_ERR_LEN, "empty block");
    return -1;
  }
  const int win_w = x_end + 2 * dh;
  const int win_h = y_end + 2 * dv;
  /* avm_gpu_ccso_block() copies pic_h rows of dst_stride words in and back out,
   * so the staged destination must be that tall and that wide: win_h is larger
   * than the block (the filter reads one row below it) and win_w can exceed the
   * plane's stride at the right edge. */
  const int stage_stride = win_w > dst_stride ? win_w : dst_stride;

  /* Luma: the reference rows/columns line up with the plane, so the window is
   * just a pointer into it. Chroma reads a subsampled luma reference, whose
   * rows/columns are vscale/hscale apart, so the window is staged. */
  const uint16_t *src_win = NULL;
  uint16_t *src_stage = NULL;
  int src_pic_w = win_w, src_pic_h = win_h;
  if (y_uv_hscale != 0 || y_uv_vscale != 0) {
    const int src_cols = (((dh + x_end - 1) << y_uv_hscale)) + 3;
    const int src_rows = (((dv + y_end - 1) << y_uv_vscale)) + 2;
    src_stage = calloc((size_t)src_rows * (size_t)src_stride,
                       sizeof(*src_stage));
    if (!src_stage) {
      snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
      return -1;
    }
    /* The block's first reference sample sits at (x << hscale) of the block
     * row, so the window origin is scaled the same way. */
    const uint16_t *start = src_y - (size_t)(dv << y_uv_vscale) * src_stride +
                            ((x - dh) << y_uv_hscale);
    for (int j = 0; j < src_rows; j++) {
      memcpy(src_stage + (size_t)j * src_stride,
             start + (size_t)j * src_stride,
             (size_t)src_cols * sizeof(*src_stage));
    }
    src_win = src_stage;
    src_pic_w = src_cols;
    src_pic_h = src_rows > win_h ? src_rows : win_h;
  } else {
    src_win = src_y - (size_t)dv * src_stride + (x - dh);
  }

  const int dst_rows = src_pic_h;
  uint16_t *dst_stage =
      calloc((size_t)dst_rows * (size_t)stage_stride, sizeof(*dst_stage));
  if (!dst_stage) {
    free(src_stage);
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    return -1;
  }
  for (int j = 0; j < dst_rows; j++) {
    const int srow = j - dv;
    const uint16_t *src_row = dst + (size_t)srow * dst_stride;
    uint16_t *drow = dst_stage + (size_t)j * stage_stride;
    for (int i = 0; i < win_w; i++) {
      const int scol = x - dh + i;
      /* Past the plane's stride the row is padded with its last sample rather
       * than read out of the buffer. */
      const int c = (srow < 0 || srow >= pic_h)
                        ? 0
                        : (scol < 0 ? 0
                                    : (scol >= dst_stride ? dst_stride - 1 : scol));
      drow[i] = (srow < 0 || srow >= pic_h) ? 0 : src_row[c];
    }
  }

  const int rc = avm_gpu_ccso_block(
      ctx, cc, src_win, src_stride, dst_stage, stage_stride, src_pic_w,
      src_pic_h, dh, dv, offset_buf, src_loc_0, src_loc_1, y_uv_hscale,
      y_uv_vscale, thr, neg_thr, max_val, x_end, y_end, single_band,
      shift_bits, edge_clf, bo_only, dst_stage, err);
  if (rc == 0) {
    for (int j = 0; j < y_end; j++) {
      memcpy(dst + (size_t)j * dst_stride + x,
             dst_stage + (size_t)(j + dv) * stage_stride + dh,
             (size_t)x_end * sizeof(*dst));
    }
  }
  free(src_stage);
  free(dst_stage);
  return rc;
}
