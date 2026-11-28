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

#include "avm_gpu/arena.h"
#include "avm_gpu/pipeline.h"

#ifndef AVM_GPU_INTRA_SPV_PATH
#define AVM_GPU_INTRA_SPV_PATH "intra.spv"
#endif
#ifndef AVM_GPU_INTRA_DIR_SPV_PATH
#define AVM_GPU_INTRA_DIR_SPV_PATH "intra_dir.spv"
#endif

#define INTRA_MAX_DIM 64
/* Guard samples before the top-left, mirroring AVM's extended edge rows. */
#define INTRA_EDGE_PAD 4
#define INTRA_WORKGROUP 64

struct avm_gpu_intra {
  avm_gpu_arena *arena; /* device-local arena when AV2_GPU_ARENA=1 */
  uint32_t *h_above, *h_left, *h_dst, *h_d_above, *h_d_left, *h_d_dst;
  size_t h_above_cap, h_left_cap, h_dst_cap, h_d_above_cap, h_d_left_cap,
      h_d_dst_cap;
  avm_gpu_pipeline *pipe;
  avm_gpu_pipeline *dir_pipe;
  avm_gpu_buf above, left, dst;
  avm_gpu_buf d_above, d_left, d_dst;
  VkDeviceSize above_cap, left_cap, dst_cap;
  VkDeviceSize d_above_cap, d_left_cap, d_dst_cap;
  uint32_t *expand;
  size_t expand_cap;
};

typedef struct dir_push_constants {
  uint32_t bw, bh, dx, dy, mrl, stride, mode, pad;
} dir_push_constants;

typedef struct push_constants {
  uint32_t bw, bh, mode, bd, stride, tl;
} push_constants;

static int ensure_buf(avm_gpu_ctx *ctx, avm_gpu_arena *arena, avm_gpu_buf *buf,
                      VkDeviceSize *cap, VkDeviceSize need,
                      char err[AVM_GPU_ERR_LEN]) {
  if (*cap >= need) return 0;
  avm_gpu_buf_destroy(ctx, buf);
  return avm_gpu_slot_reserve(ctx, arena, buf, cap, need, err);
}

int avm_gpu_intra_create(avm_gpu_ctx *ctx, avm_gpu_intra **out,
                         char err[AVM_GPU_ERR_LEN]) {
  *out = NULL;
  avm_gpu_intra *in = calloc(1, sizeof(*in));
  if (!in) {
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    return -1;
  }
  if (avm_gpu_arena_wanted("intra")) {
    const char *mb = getenv("AV2_GPU_ARENA_MB");
    const VkDeviceSize bytes =
        (VkDeviceSize)(mb && *mb ? atoi(mb) : 64) * 1024u * 1024u;
    char aerr[AVM_GPU_ERR_LEN];
    if (avm_gpu_arena_create(ctx, bytes, &in->arena, aerr) != 0) in->arena = NULL;
  }
  if (avm_gpu_pipeline_create(ctx, AVM_GPU_INTRA_SPV_PATH, "AVM_GPU_INTRA_SPV",
                              3, sizeof(push_constants), &in->pipe,
                              err) != 0) {
    free(in);
    return -1;
  }
  if (avm_gpu_pipeline_create(ctx, AVM_GPU_INTRA_DIR_SPV_PATH,
                              "AVM_GPU_INTRA_DIR_SPV", 3,
                              sizeof(dir_push_constants), &in->dir_pipe,
                              err) != 0) {
    avm_gpu_intra_destroy(ctx, in);
    return -1;
  }
  *out = in;
  return 0;
}

void avm_gpu_intra_destroy(avm_gpu_ctx *ctx, avm_gpu_intra *in) {
  if (!in) return;
  avm_gpu_pipeline_destroy(ctx, in->pipe);
  avm_gpu_pipeline_destroy(ctx, in->dir_pipe);
  avm_gpu_buf_destroy(ctx, &in->above);
  avm_gpu_buf_destroy(ctx, &in->left);
  avm_gpu_buf_destroy(ctx, &in->dst);
  avm_gpu_buf_destroy(ctx, &in->d_above);
  avm_gpu_buf_destroy(ctx, &in->d_left);
  if (in->h_above != in->above.mapped) free(in->h_above);
  if (in->h_left != in->left.mapped) free(in->h_left);
  if (in->h_dst != in->dst.mapped) free(in->h_dst);
  if (in->h_d_above != in->d_above.mapped) free(in->h_d_above);
  if (in->h_d_left != in->d_left.mapped) free(in->h_d_left);
  if (in->h_d_dst != in->d_dst.mapped) free(in->h_d_dst);
  avm_gpu_buf_destroy(ctx, &in->d_dst);
  avm_gpu_arena_destroy(ctx, in->arena);
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

int avm_gpu_intra_predict_tl(avm_gpu_ctx *ctx, avm_gpu_intra *in,
                             const uint16_t *above, const uint16_t *left,
                             int top_left, int bw, int bh, int mode, int bd,
                             uint16_t *dst, int stride,
                             char err[AVM_GPU_ERR_LEN]) {
  if (bw <= 0 || bh <= 0 || bw > INTRA_MAX_DIM || bh > INTRA_MAX_DIM) {
    snprintf(err, AVM_GPU_ERR_LEN, "unsupported block size %dx%d", bw, bh);
    return -1;
  }
  if (mode < AVM_GPU_INTRA_DC || mode > AVM_GPU_INTRA_PAETH) {
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

  /* SMOOTH* also reads above[bw] and left[bh] (top-right / bottom-left). */
  const VkDeviceSize ab_bytes = (VkDeviceSize)(bw + 1) * 4;
  const VkDeviceSize lf_bytes = (VkDeviceSize)(bh + 1) * 4;
  const VkDeviceSize dst_bytes = (VkDeviceSize)pixels * 4;
  if (ensure_buf(ctx, in->arena, &in->above, &in->above_cap, ab_bytes, err) ||
      ensure_buf(ctx, in->arena, &in->left, &in->left_cap, lf_bytes, err) ||
      ensure_buf(ctx, in->arena, &in->dst, &in->dst_cap, dst_bytes, err)) {
    return -1;
  }
  if ((!in->above.mapped &&
       avm_gpu_shadow_reserve(&in->h_above, &in->h_above_cap, ab_bytes, err)) ||
      (!in->left.mapped &&
       avm_gpu_shadow_reserve(&in->h_left, &in->h_left_cap, lf_bytes, err)) ||
      (!in->dst.mapped &&
       avm_gpu_shadow_reserve(&in->h_dst, &in->h_dst_cap, dst_bytes, err))) {
    return -1;
  }
  uint32_t *ab = in->above.mapped ? in->above.mapped : in->h_above;
  uint32_t *lf = in->left.mapped ? in->left.mapped : in->h_left;
  uint32_t *ds = in->dst.mapped ? in->dst.mapped : in->h_dst;
  for (int i = 0; i <= bw; i++) ab[i] = need_above ? above[i] : dummy;
  for (int i = 0; i <= bh; i++) lf[i] = need_left ? left[i] : dummy;
  memset(ds, 0, (size_t)dst_bytes);
  if (avm_gpu_shadow_push(ctx, in->arena, &in->above, ab, ab_bytes, err) ||
      avm_gpu_shadow_push(ctx, in->arena, &in->left, lf, lf_bytes, err) ||
      avm_gpu_shadow_push(ctx, in->arena, &in->dst, ds, dst_bytes, err)) {
    return -1;
  }

  const avm_gpu_buf bufs[3] = { in->above, in->left, in->dst };
  push_constants pc = { (uint32_t)bw,      (uint32_t)bh, (uint32_t)mode,
                        (uint32_t)bd,      (uint32_t)stride,
                        (uint32_t)top_left };
  const int rc = avm_gpu_pipeline_dispatch(ctx, in->pipe, bufs, 3, &pc,
                                           sizeof(pc), 1, err);
  if (rc != 0) return rc;

  if (avm_gpu_shadow_pull(ctx, in->arena, &in->dst, ds, dst_bytes, err)) {
    return -1;
  }
  const uint32_t *res = ds;
  for (int y = 0; y < bh; y++) {
    for (int x = 0; x < bw; x++) dst[(size_t)y * stride + x] = res[y * stride + x];
  }
  return 0;
}

int avm_gpu_intra_dr_predict(avm_gpu_ctx *ctx, avm_gpu_intra *in,
                             const uint16_t *above, const uint16_t *left,
                             int bw, int bh, int zone, int dx, int dy,
                             int mrl_index, uint16_t *dst, int stride,
                             char err[AVM_GPU_ERR_LEN]) {
  if (bw <= 0 || bh <= 0 || bw > INTRA_MAX_DIM || bh > INTRA_MAX_DIM) {
    snprintf(err, AVM_GPU_ERR_LEN, "unsupported block size %dx%d", bw, bh);
    return -1;
  }
  if (zone < 1 || zone > 3) {
    snprintf(err, AVM_GPU_ERR_LEN, "unsupported directional zone %d", zone);
    return -1;
  }
  if (mrl_index < 0 || mrl_index > 3) {
    snprintf(err, AVM_GPU_ERR_LEN, "unsupported mrl index %d", mrl_index);
    return -1;
  }
  const int edge_len = bw + bh + 8 + INTRA_EDGE_PAD;

  const VkDeviceSize edge_bytes = (VkDeviceSize)edge_len * 4;
  const VkDeviceSize ddst_bytes = (VkDeviceSize)stride * bh * 4;
  if (ensure_buf(ctx, in->arena, &in->d_above, &in->d_above_cap, edge_bytes,
                 err) ||
      ensure_buf(ctx, in->arena, &in->d_left, &in->d_left_cap, edge_bytes, err) ||
      ensure_buf(ctx, in->arena, &in->d_dst, &in->d_dst_cap, ddst_bytes, err)) {
    return -1;
  }
  if ((!in->d_above.mapped && avm_gpu_shadow_reserve(
          &in->h_d_above, &in->h_d_above_cap, edge_bytes, err)) ||
      (!in->d_left.mapped && avm_gpu_shadow_reserve(
          &in->h_d_left, &in->h_d_left_cap, edge_bytes, err)) ||
      (!in->d_dst.mapped && avm_gpu_shadow_reserve(
          &in->h_d_dst, &in->h_d_dst_cap, ddst_bytes, err))) {
    return -1;
  }

  /* above/left point at the top-left sample; copy the guard window too. */
  uint32_t *ab = in->d_above.mapped ? in->d_above.mapped : in->h_d_above;
  uint32_t *lf = in->d_left.mapped ? in->d_left.mapped : in->h_d_left;
  uint32_t *ds = in->d_dst.mapped ? in->d_dst.mapped : in->h_d_dst;
  for (int i = 0; i < edge_len; i++) {
    ab[i] = above[i - INTRA_EDGE_PAD];
    lf[i] = left[i - INTRA_EDGE_PAD];
  }
  if (avm_gpu_shadow_push(ctx, in->arena, &in->d_above, ab, edge_bytes, err) ||
      avm_gpu_shadow_push(ctx, in->arena, &in->d_left, lf, edge_bytes, err) ||
      avm_gpu_shadow_push(ctx, in->arena, &in->d_dst, ds, ddst_bytes, err)) {
    return -1;
  }

  const avm_gpu_buf bufs[3] = { in->d_above, in->d_left, in->d_dst };
  dir_push_constants pc = { (uint32_t)bw,     (uint32_t)bh,   (uint32_t)dx,
                            (uint32_t)dy,     (uint32_t)mrl_index,
                            (uint32_t)stride, (uint32_t)zone, 0 };
  const uint32_t gx = (uint32_t)((bw + 7) / 8);
  const uint32_t gy = (uint32_t)((bh + 7) / 8);
  const int rc = avm_gpu_pipeline_dispatch_2d(ctx, in->dir_pipe, bufs, 3, &pc,
                                              sizeof(pc), gx, gy, err);
  if (rc != 0) return rc;

  if (avm_gpu_shadow_pull(ctx, in->arena, &in->d_dst, ds, ddst_bytes, err)) {
    return -1;
  }
  const uint32_t *res = ds;
  for (int y = 0; y < bh; y++) {
    for (int x = 0; x < bw; x++) {
      dst[(size_t)y * stride + x] = res[y * stride + x];
    }
  }
  return 0;
}

int avm_gpu_intra_predict(avm_gpu_ctx *ctx, avm_gpu_intra *in,
                          const uint16_t *above, const uint16_t *left, int bw,
                          int bh, int mode, int bd, uint16_t *dst, int stride,
                          char err[AVM_GPU_ERR_LEN]) {
  return avm_gpu_intra_predict_tl(ctx, in, above, left, 0, bw, bh, mode, bd,
                                   dst, stride, err);
}
