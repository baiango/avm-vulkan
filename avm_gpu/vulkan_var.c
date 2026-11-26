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

#include "avm_gpu/vulkan_var.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "avm_gpu/pipeline.h"

#ifndef AVM_GPU_VAR_SPV_PATH
#define AVM_GPU_VAR_SPV_PATH "variance.spv"
#endif

struct avm_gpu_var {
  avm_gpu_pipeline *pipe;
  avm_gpu_buf src, ref, pos, res;
  VkDeviceSize src_cap, ref_cap, pos_cap, res_cap;
  int src_stride, ref_stride;
  int packed;
};

typedef struct push_constants {
  uint32_t bw, bh, src_stride, ref_stride, count, packed;
} push_constants;

/* Shader output: (sse_hi, sse_lo, sum, 0). */
typedef struct block_result {
  uint32_t sse_hi, sse_lo, sum, pad;
} block_result;

static int ensure_buf(avm_gpu_ctx *ctx, avm_gpu_buf *buf, VkDeviceSize *cap,
                      VkDeviceSize need, char err[AVM_GPU_ERR_LEN]) {
  if (*cap >= need) return 0;
  avm_gpu_buf_destroy(ctx, buf);
  if (avm_gpu_buf_create(ctx, need, buf, err) != 0) return -1;
  *cap = buf->size;
  return 0;
}

int avm_gpu_var_create(avm_gpu_ctx *ctx, avm_gpu_var **out,
                       char err[AVM_GPU_ERR_LEN]) {
  *out = NULL;
  avm_gpu_var *v = calloc(1, sizeof(*v));
  if (!v) {
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    return -1;
  }
  if (avm_gpu_pipeline_create(ctx, AVM_GPU_VAR_SPV_PATH, "AVM_GPU_VAR_SPV", 4,
                              sizeof(push_constants), &v->pipe, err) != 0) {
    free(v);
    return -1;
  }
  *out = v;
  return 0;
}

void avm_gpu_var_destroy(avm_gpu_ctx *ctx, avm_gpu_var *v) {
  if (!v) return;
  avm_gpu_pipeline_destroy(ctx, v->pipe);
  avm_gpu_buf_destroy(ctx, &v->src);
  avm_gpu_buf_destroy(ctx, &v->ref);
  avm_gpu_buf_destroy(ctx, &v->pos);
  avm_gpu_buf_destroy(ctx, &v->res);
  free(v);
}

/* M0 uploads one 16-bit sample per 32-bit word; see avm_gpu/README.md. */
static uint32_t *widen(const uint16_t *plane, int stride, int rows) {
  if (!plane || stride <= 0 || rows <= 0) return NULL;
  const size_t n = (size_t)stride * (size_t)rows;
  uint32_t *w = malloc(n * sizeof(*w));
  if (!w) return NULL;
  for (size_t i = 0; i < n; i++) w[i] = plane[i];
  return w;
}

int avm_gpu_var_set_planes(avm_gpu_ctx *ctx, avm_gpu_var *v,
                           const uint16_t *src, int src_stride, int src_rows,
                           const uint16_t *ref, int ref_stride, int ref_rows,
                           char err[AVM_GPU_ERR_LEN]) {
  /* M1 packed layout needs even strides so a row maps onto whole words. */
  const char *pack_env = getenv("AVM_GPU_PACK16");
  v->packed = pack_env && *pack_env && *pack_env != '0' &&
              (src_stride % 2) == 0 && (ref_stride % 2) == 0;

  const size_t src_n = (size_t)src_stride * (size_t)src_rows;
  const size_t ref_n = (size_t)ref_stride * (size_t)ref_rows;
  const VkDeviceSize src_bytes =
      v->packed ? (VkDeviceSize)(src_n * 2) : (VkDeviceSize)(src_n * 4);
  const VkDeviceSize ref_bytes =
      v->packed ? (VkDeviceSize)(ref_n * 2) : (VkDeviceSize)(ref_n * 4);

  if (v->packed) {
    if (ensure_buf(ctx, &v->src, &v->src_cap, src_bytes, err) ||
        ensure_buf(ctx, &v->ref, &v->ref_cap, ref_bytes, err)) {
      return -1;
    }
    /* Two adjacent 16-bit samples already are one little-endian word. */
    memcpy(v->src.mapped, src, (size_t)src_bytes);
    memcpy(v->ref.mapped, ref, (size_t)ref_bytes);
    v->src_stride = src_stride;
    v->ref_stride = ref_stride;
    return 0;
  }

  uint32_t *sw = widen(src, src_stride, src_rows);
  uint32_t *rw = widen(ref, ref_stride, ref_rows);
  if (!sw || !rw) {
    free(sw);
    free(rw);
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    return -1;
  }
  if (ensure_buf(ctx, &v->src, &v->src_cap, src_bytes, err) ||
      ensure_buf(ctx, &v->ref, &v->ref_cap, ref_bytes, err)) {
    free(sw);
    free(rw);
    return -1;
  }
  memcpy(v->src.mapped, sw, (size_t)src_bytes);
  memcpy(v->ref.mapped, rw, (size_t)ref_bytes);
  free(sw);
  free(rw);
  v->src_stride = src_stride;
  v->ref_stride = ref_stride;
  return 0;
}

/* avm_dsp/variance.c semantics, verbatim. */
static uint64_t round_pow2_64(uint64_t value, int n) {
  return (value + ((((uint64_t)1 << n) >> 1))) >> n;
}

static int64_t round_pow2_signed(int64_t value, int n) {
  return (value + (((int64_t)1 << n) >> 1)) >> n;
}

int avm_gpu_var_run(avm_gpu_ctx *ctx, avm_gpu_var *v,
                    const avm_gpu_var_pos *blocks, int n, int bw, int bh,
                    int bitdepth, uint32_t *variance_out, uint32_t *sse_out,
                    char err[AVM_GPU_ERR_LEN]) {
  if (n <= 0) return 0;
  if (v->src_stride <= 0 || v->ref_stride <= 0) {
    snprintf(err, AVM_GPU_ERR_LEN, "planes not set");
    return -1;
  }
  if (bitdepth != 8 && bitdepth != 10 && bitdepth != 12) {
    snprintf(err, AVM_GPU_ERR_LEN, "unsupported bitdepth %d", bitdepth);
    return -1;
  }
  const VkDeviceSize pos_bytes = (VkDeviceSize)n * sizeof(*blocks);
  const VkDeviceSize res_bytes = (VkDeviceSize)n * sizeof(block_result);
  if (ensure_buf(ctx, &v->pos, &v->pos_cap, pos_bytes, err) ||
      ensure_buf(ctx, &v->res, &v->res_cap, res_bytes, err)) {
    return -1;
  }
  memcpy(v->pos.mapped, blocks, (size_t)pos_bytes);

  const avm_gpu_buf bufs[4] = { v->src, v->ref, v->pos, v->res };
  push_constants pc = { (uint32_t)bw,   (uint32_t)bh,
                        (uint32_t)v->src_stride, (uint32_t)v->ref_stride,
                        (uint32_t)n,    (uint32_t)v->packed };
  const int rc = avm_gpu_pipeline_dispatch(ctx, v->pipe, bufs, 4, &pc,
                                           sizeof(pc), (uint32_t)n, err);
  if (rc != 0) return rc;

  const block_result *res = (const block_result *)v->res.mapped;
  const int sse_shift = bitdepth == 8 ? 0 : (bitdepth == 10 ? 4 : 8);
  const int sum_shift = sse_shift / 2;
  for (int i = 0; i < n; i++) {
    const uint64_t sse_raw = ((uint64_t)res[i].sse_hi << 32) | res[i].sse_lo;
    const int64_t sum_raw = (int32_t)res[i].sum;
    const uint64_t sse = sse_shift ? round_pow2_64(sse_raw, sse_shift)
                                   : sse_raw;
    const int64_t sum = sum_shift ? round_pow2_signed(sum_raw, sum_shift)
                                  : sum_raw;
    const uint32_t sse32 = (uint32_t)sse;
    if (sse_out) sse_out[i] = sse32;
    variance_out[i] = sse32 - (uint32_t)(((int64_t)sum * sum) / (bw * bh));
  }
  return 0;
}

int avm_gpu_var_run_raw(avm_gpu_ctx *ctx, avm_gpu_var *v,
                        const avm_gpu_var_pos *blocks, int n, int bw, int bh,
                        uint64_t *sse_out, int64_t *sum_out,
                        char err[AVM_GPU_ERR_LEN]) {
  if (n <= 0) return 0;
  if (v->src_stride <= 0 || v->ref_stride <= 0) {
    snprintf(err, AVM_GPU_ERR_LEN, "planes not set");
    return -1;
  }
  const VkDeviceSize pos_bytes = (VkDeviceSize)n * sizeof(*blocks);
  const VkDeviceSize res_bytes = (VkDeviceSize)n * sizeof(block_result);
  if (ensure_buf(ctx, &v->pos, &v->pos_cap, pos_bytes, err) ||
      ensure_buf(ctx, &v->res, &v->res_cap, res_bytes, err)) {
    return -1;
  }
  memcpy(v->pos.mapped, blocks, (size_t)pos_bytes);

  const avm_gpu_buf bufs[4] = { v->src, v->ref, v->pos, v->res };
  push_constants pc = { (uint32_t)bw,   (uint32_t)bh,
                        (uint32_t)v->src_stride, (uint32_t)v->ref_stride,
                        (uint32_t)n,    (uint32_t)v->packed };
  const int rc = avm_gpu_pipeline_dispatch(ctx, v->pipe, bufs, 4, &pc,
                                           sizeof(pc), (uint32_t)n, err);
  if (rc != 0) return rc;

  const block_result *res = (const block_result *)v->res.mapped;
  for (int i = 0; i < n; i++) {
    if (sse_out) {
      sse_out[i] = ((uint64_t)res[i].sse_hi << 32) | res[i].sse_lo;
    }
    if (sum_out) sum_out[i] = (int32_t)res[i].sum;
  }
  return 0;
}
