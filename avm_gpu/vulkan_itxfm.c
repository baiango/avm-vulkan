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

#include "avm_gpu/vulkan_itxfm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "avm_gpu/pipeline.h"

#ifndef AVM_GPU_ITXFM_SPV_PATH
#define AVM_GPU_ITXFM_SPV_PATH "itxfm.spv"
#endif

#define ITXFM_WORKGROUP 32

struct avm_gpu_itxfm {
  avm_gpu_pipeline *pipe;
  avm_gpu_buf coef, pos, dst, mat;
  VkDeviceSize coef_cap, pos_cap, dst_cap;
  uint32_t *plane;
  size_t plane_cap;
};

typedef struct push_constants {
  uint32_t w, h, bd, stride, count, shift1, shift2, sqrt2;
  uint32_t out_w, out_h;
} push_constants;

typedef struct block_pos {
  uint32_t src_off, dst_off;
} block_pos;

static int ensure_buf(avm_gpu_ctx *ctx, avm_gpu_buf *buf, VkDeviceSize *cap,
                      VkDeviceSize need, char err[AVM_GPU_ERR_LEN]) {
  if (*cap >= need) return 0;
  avm_gpu_buf_destroy(ctx, buf);
  if (avm_gpu_buf_create(ctx, need, buf, err) != 0) return -1;
  *cap = buf->size;
  return 0;
}

int avm_gpu_itxfm_create(avm_gpu_ctx *ctx, avm_gpu_itxfm **out,
                         char err[AVM_GPU_ERR_LEN]) {
  *out = NULL;
  avm_gpu_itxfm *it = calloc(1, sizeof(*it));
  if (!it) {
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    return -1;
  }
  if (avm_gpu_pipeline_create(ctx, AVM_GPU_ITXFM_SPV_PATH, "AVM_GPU_ITXFM_SPV",
                              4, sizeof(push_constants), &it->pipe,
                              err) != 0) {
    free(it);
    return -1;
  }
  *out = it;
  return 0;
}

void avm_gpu_itxfm_destroy(avm_gpu_ctx *ctx, avm_gpu_itxfm *it) {
  if (!it) return;
  avm_gpu_pipeline_destroy(ctx, it->pipe);
  avm_gpu_buf_destroy(ctx, &it->coef);
  avm_gpu_buf_destroy(ctx, &it->pos);
  avm_gpu_buf_destroy(ctx, &it->dst);
  avm_gpu_buf_destroy(ctx, &it->mat);
  free(it->plane);
  free(it);
}

/* tx_kernel_dct2_size4/8[INV_TXFM][0], used when the caller passes NULL. */
static const int kMat4[16] = { 64, 64, 64, 64,   83, 35, -35, -83,
                               64, -64, -64, 64,  35, -83, 83, -35 };
static const int kMat8[64] = {
  64, 64, 64, 64, 64, 64, 64, 64,   89, 75, 50, 18, -18, -50, -75, -89,
  83, 35, -35, -83, -83, -35, 35, 83, 75, -18, -89, -50, 50, 89, 18, -75,
  64, -64, -64, 64, 64, -64, -64, 64, 50, -89, 18, 75, -75, -18, 89, -50,
  35, -83, 83, -35, -35, 83, -83, 35, 18, -50, 75, -89, 89, -75, 50, -18
};

static int log2_of(int v) {
  int r = 0;
  while (v > 1) {
    v >>= 1;
    ++r;
  }
  return r;
}

/* inv_tx_shift[] from av2/common/common_data.h for the supported sizes. */
static void shift_pair(int w, int h, uint32_t *s1, uint32_t *s2) {
  static const struct {
    int w, h, s1, s2;
  } kTab[] = {
    { 4, 4, 7, 10 },   { 8, 8, 7, 11 },   { 16, 16, 6, 13 },
    { 32, 32, 6, 13 }, { 4, 8, 7, 10 },   { 8, 4, 7, 10 },
    { 8, 16, 7, 11 },  { 16, 8, 7, 11 },  { 16, 32, 6, 12 },
    { 32, 16, 6, 12 }, { 4, 16, 6, 12 },  { 16, 4, 6, 12 },
    { 8, 32, 6, 13 },  { 32, 8, 6, 13 },  { 4, 32, 7, 11 },
    { 32, 4, 7, 11 },
    /* 64-wide/tall sizes: inv_tx_shift[TX_*] for the 64 row. */
    { 64, 64, 6, 13 }, { 64, 32, 6, 12 }, { 32, 64, 6, 12 },
    { 16, 64, 6, 13 }, { 64, 16, 6, 13 }, { 8, 64, 6, 12 },
    { 64, 8, 6, 12 },  { 4, 64, 6, 13 },  { 64, 4, 6, 13 },
  };
  *s1 = 7;
  *s2 = 10;
  for (size_t i = 0; i < sizeof(kTab) / sizeof(kTab[0]); i++) {
    if (kTab[i].w == w && kTab[i].h == h) {
      *s1 = (uint32_t)kTab[i].s1;
      *s2 = (uint32_t)kTab[i].s2;
      return;
    }
  }
}

/* tx_w/tx_h are the block's transform size (4..64 in steps of 4); the 1D pass
 * runs at w = min(32, tx_w), h = min(32, tx_h) and the kernel replicates the
 * result, exactly as inv_txfm_c() does. */
static int add_tx(avm_gpu_ctx *ctx, avm_gpu_itxfm *it, const int32_t *coeffs,
                  const uint32_t *dst_off, int n, int tx_w, int tx_h, int bd,
                  const int *mat_w, const int *mat_h, uint16_t *dest,
                  int stride, int rows, char err[AVM_GPU_ERR_LEN]) {
  if (n <= 0) return 0;
  if (tx_w < 4 || tx_h < 4 || tx_w > 64 || tx_h > 64 || (tx_w & 3) ||
      (tx_h & 3)) {
    snprintf(err, AVM_GPU_ERR_LEN, "unsupported transform size %dx%d", tx_w,
             tx_h);
    return -1;
  }
  if (!coeffs || !dst_off || !dest || stride < tx_w || rows < tx_h) {
    snprintf(err, AVM_GPU_ERR_LEN, "invalid arguments");
    return -1;
  }
  const int w = tx_w > 32 ? 32 : tx_w;
  const int h = tx_h > 32 ? 32 : tx_h;
  const size_t coef_per_block = (size_t)tx_w * (size_t)tx_h;

  const size_t pixels = (size_t)stride * rows;
  if (pixels > it->plane_cap) {
    free(it->plane);
    it->plane = malloc(pixels * sizeof(*it->plane));
    it->plane_cap = it->plane ? pixels : 0;
    if (!it->plane) {
      snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
      return -1;
    }
  }
  for (size_t i = 0; i < pixels; i++) it->plane[i] = dest[i];

  block_pos *pos = malloc((size_t)n * sizeof(*pos));
  if (!pos) {
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    return -1;
  }
  for (int i = 0; i < n; i++) {
    pos[i].src_off = (uint32_t)((size_t)i * coef_per_block);
    pos[i].dst_off = dst_off[i];
  }

  const VkDeviceSize coef_bytes = (VkDeviceSize)n * coef_per_block * 4;
  const VkDeviceSize pos_bytes = (VkDeviceSize)n * sizeof(*pos);
  const VkDeviceSize dst_bytes = (VkDeviceSize)pixels * 4;
  const VkDeviceSize mat_bytes = (VkDeviceSize)(w * w + h * h) * 4;
  int rc = -1;
  if (ensure_buf(ctx, &it->coef, &it->coef_cap, coef_bytes, err) ||
      ensure_buf(ctx, &it->pos, &it->pos_cap, pos_bytes, err) ||
      ensure_buf(ctx, &it->dst, &it->dst_cap, dst_bytes, err)) {
    free(pos);
    return -1;
  }
  if (it->mat.size < mat_bytes) {
    avm_gpu_buf_destroy(ctx, &it->mat);
    if (avm_gpu_buf_create(ctx, mat_bytes, &it->mat, err) != 0) {
      free(pos);
      return -1;
    }
  }
  memcpy(it->coef.mapped, coeffs, (size_t)coef_bytes);
  memcpy(it->pos.mapped, pos, (size_t)pos_bytes);
  memcpy(it->dst.mapped, it->plane, (size_t)dst_bytes);
  int32_t *mp = it->mat.mapped;
  memcpy(mp, mat_w, (size_t)w * w * 4);
  memcpy(mp + w * w, mat_h, (size_t)h * h * 4);
  free(pos);

  const avm_gpu_buf bufs[4] = { it->coef, it->pos, it->dst, it->mat };
  uint32_t s1 = 7, s2 = 10;
  shift_pair(tx_w, tx_h, &s1, &s2);
  const int sqrt2 = ((log2_of(tx_w) + log2_of(tx_h)) & 1) ? 1 : 0;
  push_constants pc = { (uint32_t)w,      (uint32_t)h,    (uint32_t)bd,
                        (uint32_t)stride, (uint32_t)n,    s1,
                        s2,               (uint32_t)sqrt2,
                        (uint32_t)tx_w,   (uint32_t)tx_h };
  const uint32_t groups = (uint32_t)((n + ITXFM_WORKGROUP - 1) / ITXFM_WORKGROUP);
  rc = avm_gpu_pipeline_dispatch(ctx, it->pipe, bufs, 4, &pc, sizeof(pc),
                                 groups, err);
  if (rc != 0) return rc;

  const uint32_t *res = it->dst.mapped;
  for (size_t i = 0; i < pixels; i++) dest[i] = (uint16_t)res[i];
  return 0;
}

int avm_gpu_itxfm_add_rect(avm_gpu_ctx *ctx, avm_gpu_itxfm *it,
                           const int32_t *coeffs, const uint32_t *dst_off,
                           int n, int w, int h, int bd, const int *mat_w,
                           const int *mat_h, uint16_t *dest, int stride,
                           int rows, char err[AVM_GPU_ERR_LEN]) {
  return add_tx(ctx, it, coeffs, dst_off, n, w, h, bd, mat_w, mat_h, dest,
                stride, rows, err);
}

int avm_gpu_itxfm_add_tx(avm_gpu_ctx *ctx, avm_gpu_itxfm *it,
                         const int32_t *coeffs, const uint32_t *dst_off, int n,
                         int tx_w, int tx_h, int bd, const int *mat_w,
                         const int *mat_h, uint16_t *dest, int stride,
                         int rows, char err[AVM_GPU_ERR_LEN]) {
  return add_tx(ctx, it, coeffs, dst_off, n, tx_w, tx_h, bd, mat_w, mat_h,
                dest, stride, rows, err);
}

int avm_gpu_itxfm_add_m(avm_gpu_ctx *ctx, avm_gpu_itxfm *it,
                        const int32_t *coeffs, const uint32_t *dst_off, int n,
                        int size, int bd, const int *matrix, uint16_t *dest,
                        int stride, int rows, char err[AVM_GPU_ERR_LEN]) {
  const int *mat = matrix;
  if (!mat) mat = size == 4 ? kMat4 : (size == 8 ? kMat8 : NULL);
  if (!mat) {
    snprintf(err, AVM_GPU_ERR_LEN, "a matrix is required for size %d", size);
    return -1;
  }
  return avm_gpu_itxfm_add_rect(ctx, it, coeffs, dst_off, n, size, size, bd,
                                mat, mat, dest, stride, rows, err);
}

int avm_gpu_itxfm_add(avm_gpu_ctx *ctx, avm_gpu_itxfm *it,
                      const int32_t *coeffs, const uint32_t *dst_off, int n,
                      int size, int bd, uint16_t *dest, int stride, int rows,
                      char err[AVM_GPU_ERR_LEN]) {
  return avm_gpu_itxfm_add_m(ctx, it, coeffs, dst_off, n, size, bd, NULL, dest,
                             stride, rows, err);
}
