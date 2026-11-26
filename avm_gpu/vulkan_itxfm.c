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
  uint32_t size, bd, stride, count, shift1, shift2, pad0, pad1;
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

int avm_gpu_itxfm_add_m(avm_gpu_ctx *ctx, avm_gpu_itxfm *it,
                        const int32_t *coeffs, const uint32_t *dst_off, int n,
                        int size, int bd, const int *matrix, uint16_t *dest,
                        int stride, int rows, char err[AVM_GPU_ERR_LEN]) {
  if (n <= 0) return 0;
  if (size != 4 && size != 8 && size != 16 && size != 32) {
    snprintf(err, AVM_GPU_ERR_LEN, "unsupported transform size %d", size);
    return -1;
  }
  if (!coeffs || !dst_off || !dest || stride < size || rows < size) {
    snprintf(err, AVM_GPU_ERR_LEN, "invalid arguments");
    return -1;
  }

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
    pos[i].src_off = (uint32_t)i * (uint32_t)(size * size);
    pos[i].dst_off = dst_off[i];
  }

  const VkDeviceSize coef_bytes =
      (VkDeviceSize)n * (VkDeviceSize)size * size * 4;
  const VkDeviceSize pos_bytes = (VkDeviceSize)n * sizeof(*pos);
  const VkDeviceSize dst_bytes = (VkDeviceSize)pixels * 4;
  int rc = -1;
  if (ensure_buf(ctx, &it->coef, &it->coef_cap, coef_bytes, err) ||
      ensure_buf(ctx, &it->pos, &it->pos_cap, pos_bytes, err) ||
      ensure_buf(ctx, &it->dst, &it->dst_cap, dst_bytes, err)) {
    free(pos);
    return -1;
  }
  memcpy(it->coef.mapped, coeffs, (size_t)coef_bytes);
  memcpy(it->pos.mapped, pos, (size_t)pos_bytes);
  memcpy(it->dst.mapped, it->plane, (size_t)dst_bytes);
  free(pos);

  const int *mat = matrix;
  if (!mat) mat = size == 4 ? kMat4 : (size == 8 ? kMat8 : NULL);
  if (!mat) {
    snprintf(err, AVM_GPU_ERR_LEN, "a matrix is required for size %d", size);
    return -1;
  }
  const VkDeviceSize mat_bytes = (VkDeviceSize)size * size * 4;
  if (it->mat.size < mat_bytes) {
    avm_gpu_buf_destroy(ctx, &it->mat);
    if (avm_gpu_buf_create(ctx, mat_bytes, &it->mat, err) != 0) return -1;
  }
  memcpy(it->mat.mapped, mat, (size_t)mat_bytes);

  const avm_gpu_buf bufs[4] = { it->coef, it->pos, it->dst, it->mat };
  push_constants pc = { (uint32_t)size,
                        (uint32_t)bd,
                        (uint32_t)stride,
                        (uint32_t)n,
                        size <= 8 ? 7u : 6u,
                        size == 4 ? 10u : (size == 8 ? 11u : 13u),
                        0,
                        0 };
  const uint32_t groups = (uint32_t)((n + ITXFM_WORKGROUP - 1) / ITXFM_WORKGROUP);
  rc = avm_gpu_pipeline_dispatch(ctx, it->pipe, bufs, 4, &pc, sizeof(pc),
                                 groups, err);
  if (rc != 0) return rc;

  const uint32_t *res = it->dst.mapped;
  for (size_t i = 0; i < pixels; i++) dest[i] = (uint16_t)res[i];
  return 0;
}

int avm_gpu_itxfm_add(avm_gpu_ctx *ctx, avm_gpu_itxfm *it,
                      const int32_t *coeffs, const uint32_t *dst_off, int n,
                      int size, int bd, uint16_t *dest, int stride, int rows,
                      char err[AVM_GPU_ERR_LEN]) {
  return avm_gpu_itxfm_add_m(ctx, it, coeffs, dst_off, n, size, bd, NULL, dest,
                             stride, rows, err);
}
