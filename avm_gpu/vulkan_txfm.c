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

#ifndef AVM_GPU_FWD_TXFM_SPV_PATH
#define AVM_GPU_FWD_TXFM_SPV_PATH "fwd_txfm.spv"
#endif

typedef struct fwd_constants {
  uint32_t w, h, diff_stride, ds_x, ds_y, shift1, shift2, sqrt2, count;
  uint32_t mat_h_off;
  uint32_t kind_w, kind_h;
  int32_t scale_w, scale_h;
  int32_t flip_w, flip_h;
} fwd_constants;

struct avm_gpu_txfm {
  avm_gpu_pipeline *fwd_pipe;
  avm_gpu_buf f_resi, f_pos, f_coef, f_mat;
  VkDeviceSize f_resi_cap, f_pos_cap, f_coef_cap, f_mat_cap;
  uint32_t *f_expand;
  size_t f_expand_cap;
};

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
  if (avm_gpu_pipeline_create(ctx, AVM_GPU_FWD_TXFM_SPV_PATH,
                              "AVM_GPU_FWD_TXFM_SPV", 4, sizeof(fwd_constants),
                              &t->fwd_pipe, err) != 0) {
    free(t);
    return -1;
  }
  *out = t;
  return 0;
}

void avm_gpu_txfm_destroy(avm_gpu_ctx *ctx, avm_gpu_txfm *t) {
  if (!t) return;
  avm_gpu_pipeline_destroy(ctx, t->fwd_pipe);
  avm_gpu_buf_destroy(ctx, &t->f_resi);
  avm_gpu_buf_destroy(ctx, &t->f_pos);
  avm_gpu_buf_destroy(ctx, &t->f_coef);
  avm_gpu_buf_destroy(ctx, &t->f_mat);
  free(t->f_expand);
  free(t);
}

int avm_gpu_txfm_fwd(avm_gpu_ctx *ctx, avm_gpu_txfm *t, const int16_t *resi,
                     int diff_stride, const avm_gpu_txfm_pos *pos, int n,
                     int tx_w, int tx_h, int shift1, int shift2, int kind_w,
                     int kind_h, int scale_w, int scale_h, int flip_w,
                     int flip_h, const int *mat_w, const int *mat_h,
                     int32_t *coeff, char err[AVM_GPU_ERR_LEN]) {
  if (n <= 0) return 0;
  if (!resi || !pos || !coeff || diff_stride <= 0) {
    snprintf(err, AVM_GPU_ERR_LEN, "invalid arguments");
    return -1;
  }
  if ((kind_w == 2 && !mat_w) || (kind_h == 2 && !mat_h)) {
    snprintf(err, AVM_GPU_ERR_LEN, "matrix kind needs a kernel");
    return -1;
  }
  if (kind_w < 0 || kind_w > 2 || kind_h < 0 || kind_h > 2) {
    snprintf(err, AVM_GPU_ERR_LEN, "unsupported 1D kind");
    return -1;
  }
  if (tx_w < 4 || tx_h < 4 || tx_w > 64 || tx_h > 64 || (tx_w & 3) ||
      (tx_h & 3)) {
    snprintf(err, AVM_GPU_ERR_LEN, "unsupported transform size %dx%d", tx_w,
             tx_h);
    return -1;
  }
  const int w = tx_w > 32 ? 32 : tx_w;
  const int h = tx_h > 32 ? 32 : tx_h;
  const int ds_x = tx_w > 32;
  const int ds_y = tx_h > 32;

  size_t max_off = 0;
  for (int i = 0; i < n; i++) {
    /* The downsample reads row (y << ds_y) and, when ds_y, the line below it,
     * and column (x << ds_x) plus one when ds_x. */
    const size_t rows = (size_t)(((h - 1) << ds_y) + (ds_y ? 1 : 0));
    const size_t cols = (size_t)(((w - 1) << ds_x) + (ds_x ? 1 : 0));
    const size_t last = (size_t)pos[i].resi_off + rows * diff_stride + cols + 1;
    if (last > max_off) max_off = last;
  }
  if (max_off > t->f_expand_cap) {
    free(t->f_expand);
    t->f_expand = malloc(max_off * sizeof(*t->f_expand));
    t->f_expand_cap = t->f_expand ? max_off : 0;
    if (!t->f_expand) {
      snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
      return -1;
    }
  }
  for (size_t i = 0; i < max_off; i++) t->f_expand[i] = (uint32_t)(int32_t)resi[i];

  const VkDeviceSize resi_bytes = (VkDeviceSize)max_off * 4;
  const VkDeviceSize coef_bytes = (VkDeviceSize)n * (VkDeviceSize)w * h * 4;
  const VkDeviceSize mat_bytes = (VkDeviceSize)(w * w + h * h) * 4;
  if (ensure_buf(ctx, &t->f_resi, &t->f_resi_cap, resi_bytes, err) ||
      ensure_buf(ctx, &t->f_pos, &t->f_pos_cap,
                 (VkDeviceSize)n * sizeof(*pos), err) ||
      ensure_buf(ctx, &t->f_coef, &t->f_coef_cap, coef_bytes, err)) {
    return -1;
  }
  if (t->f_mat.size < mat_bytes) {
    avm_gpu_buf_destroy(ctx, &t->f_mat);
    if (avm_gpu_buf_create(ctx, mat_bytes, &t->f_mat, err) != 0) return -1;
  }
  memcpy(t->f_resi.mapped, t->f_expand, (size_t)resi_bytes);
  memcpy(t->f_pos.mapped, pos, (size_t)n * sizeof(*pos));
  int32_t *mp = t->f_mat.mapped;
  memset(mp, 0, (size_t)(w * w + h * h) * 4);
  if (mat_w) memcpy(mp, mat_w, (size_t)w * w * 4);
  if (mat_h) memcpy(mp + w * w, mat_h, (size_t)h * h * 4);

  const avm_gpu_buf bufs[4] = { t->f_resi, t->f_coef, t->f_pos, t->f_mat };
  const int log2w = tx_w == 64 ? 6 : (tx_w == 32 ? 5 : (tx_w == 16 ? 4 : (tx_w == 8 ? 3 : 2)));
  const int log2h = tx_h == 64 ? 6 : (tx_h == 32 ? 5 : (tx_h == 16 ? 4 : (tx_h == 8 ? 3 : 2)));
  fwd_constants pc = { (uint32_t)w,        (uint32_t)h,
                       (uint32_t)diff_stride, (uint32_t)ds_x,
                       (uint32_t)ds_y,     (uint32_t)shift1,
                       (uint32_t)shift2,   (uint32_t)((log2w + log2h) & 1),
                       (uint32_t)n,        (uint32_t)(w * w),
                       (uint32_t)kind_w,   (uint32_t)kind_h,
                       scale_w,            scale_h,
                       flip_w,             flip_h };
  const uint32_t groups = (uint32_t)((n + 31) / 32);
  const int rc = avm_gpu_pipeline_dispatch(ctx, t->fwd_pipe, bufs, 4, &pc,
                                           sizeof(pc), groups, err);
  if (rc != 0) return rc;

  int32_t *out = t->f_coef.mapped;
  for (int i = 0; i < n; i++) {
    memcpy(coeff + pos[i].coef_off, out + (size_t)i * w * h,
           (size_t)w * h * 4);
  }
  return 0;
}
