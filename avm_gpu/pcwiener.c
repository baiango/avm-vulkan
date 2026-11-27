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

#include "avm_gpu/pcwiener.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "avm_gpu/pipeline.h"

#ifndef AVM_GPU_PCWIENER_SPV_PATH
#define AVM_GPU_PCWIENER_SPV_PATH "pcwiener.spv"
#endif

#define PCWIENER_WORKGROUP 64

struct avm_gpu_pcwiener {
  avm_gpu_pipeline *pipe;
  avm_gpu_buf img, cid, sel, flt, geo, dst;
  VkDeviceSize img_cap, cid_cap, sel_cap, flt_cap, geo_cap, dst_cap;
};

typedef struct push_constants {
  uint32_t width, height, stride, dst_stride, lead_rows, lead_cols;
  uint32_t bd, cid_stride, num_sym, singleton_index, prec_bits, num_taps;
  uint32_t count;
} push_constants;

typedef struct tap_geo {
  int32_t diff, pos;
} tap_geo;

static int ensure_buf(avm_gpu_ctx *ctx, avm_gpu_buf *buf, VkDeviceSize *cap,
                      VkDeviceSize need, char err[AVM_GPU_ERR_LEN]) {
  if (*cap >= need) return 0;
  avm_gpu_buf_destroy(ctx, buf);
  if (avm_gpu_buf_create(ctx, need, buf, err) != 0) return -1;
  *cap = buf->size;
  return 0;
}

int avm_gpu_pcwiener_create(avm_gpu_ctx *ctx, avm_gpu_pcwiener **out,
                            char err[AVM_GPU_ERR_LEN]) {
  *out = NULL;
  avm_gpu_pcwiener *pc = calloc(1, sizeof(*pc));
  if (!pc) {
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    return -1;
  }
  if (avm_gpu_pipeline_create(ctx, AVM_GPU_PCWIENER_SPV_PATH,
                              "AVM_GPU_PCWIENER_SPV", 6, sizeof(push_constants),
                              &pc->pipe, err) != 0) {
    free(pc);
    return -1;
  }
  *out = pc;
  return 0;
}

void avm_gpu_pcwiener_destroy(avm_gpu_ctx *ctx, avm_gpu_pcwiener *pc) {
  if (!pc) return;
  avm_gpu_pipeline_destroy(ctx, pc->pipe);
  avm_gpu_buf_destroy(ctx, &pc->img);
  avm_gpu_buf_destroy(ctx, &pc->cid);
  avm_gpu_buf_destroy(ctx, &pc->sel);
  avm_gpu_buf_destroy(ctx, &pc->flt);
  avm_gpu_buf_destroy(ctx, &pc->geo);
  avm_gpu_buf_destroy(ctx, &pc->dst);
  free(pc);
}

int avm_gpu_pcwiener_filter(avm_gpu_ctx *ctx, avm_gpu_pcwiener *pc,
                            const avm_gpu_pcwiener_params *params,
                            const uint16_t *dgd, const uint8_t *class_id,
                            const uint8_t *filter_selector,
                            const int16_t *filters, uint16_t *dst,
                            char err[AVM_GPU_ERR_LEN]) {
  if (!params || !dgd || !class_id || !filter_selector || !filters || !dst) {
    snprintf(err, AVM_GPU_ERR_LEN, "invalid arguments");
    return -1;
  }
  const int w = params->width, h = params->height;
  if (w <= 0 || h <= 0 || params->stride < w + params->lead_cols ||
      params->dst_stride < w + params->lead_cols ||
      params->rows < h + 2 * params->lead_rows || params->num_sym <= 0 ||
      params->num_sym > 24 || !params->config) {
    snprintf(err, AVM_GPU_ERR_LEN, "invalid geometry");
    return -1;
  }

  /* Tap geometry exactly as av2_convolve_symmetric_highbd_c() builds it: the
   * even config entries are the symmetric pairs, the last entry is the centre
   * tap's filter position. */
  tap_geo geo[24];
  for (int k = 0; k < params->num_sym; k++) {
    const int r = params->config[2 * k][0];
    const int c = params->config[2 * k][1];
    geo[k].diff = r * params->stride + c;
    geo[k].pos = params->config[2 * k][2];
  }

  const size_t img_n = (size_t)params->rows * (size_t)params->stride;
  const size_t cid_n = (size_t)((h + 3) / 4) * (size_t)params->class_id_stride;
  const size_t filter_words =
      (size_t)params->num_taps * (size_t)256; /* 256 classes at most */

  if (ensure_buf(ctx, &pc->img, &pc->img_cap, img_n * 4, err) ||
      ensure_buf(ctx, &pc->cid, &pc->cid_cap, cid_n * 4, err) ||
      ensure_buf(ctx, &pc->sel, &pc->sel_cap, 256 * 4, err) ||
      ensure_buf(ctx, &pc->flt, &pc->flt_cap, filter_words * 4, err) ||
      ensure_buf(ctx, &pc->geo, &pc->geo_cap, sizeof(geo), err) ||
      ensure_buf(ctx, &pc->dst, &pc->dst_cap, img_n * 4, err)) {
    return -1;
  }

  /* The plane is uploaded as one 16-bit sample per 32-bit word (M0). */
  const uint16_t *plane = dgd - (size_t)params->lead_rows * params->stride -
                          params->lead_cols;
  uint32_t *img = pc->img.mapped;
  for (size_t i = 0; i < img_n; i++) img[i] = plane[i];
  uint32_t *cid = pc->cid.mapped;
  for (size_t i = 0; i < cid_n; i++) cid[i] = class_id[i];
  uint32_t *sel = pc->sel.mapped;
  for (int i = 0; i < 256; i++) sel[i] = filter_selector[i];
  int32_t *flt = pc->flt.mapped;
  size_t taps = 0;
  for (int f = 0; f < 256; f++) {
    for (int t = 0; t < params->num_taps; t++) {
      flt[taps++] = filters[(size_t)f * params->num_taps + t];
    }
  }
  memcpy(pc->geo.mapped, geo, sizeof(geo));

  const uint16_t *dst_plane = dst - (size_t)params->lead_rows * params->dst_stride -
                              params->lead_cols;
  uint32_t *d32 = pc->dst.mapped;
  for (size_t i = 0; i < img_n; i++) d32[i] = dst_plane[i];

  const avm_gpu_buf bufs[6] = { pc->img, pc->cid, pc->sel,
                                pc->flt, pc->geo, pc->dst };
  push_constants p = { (uint32_t)w,
                       (uint32_t)h,
                       (uint32_t)params->stride,
                       (uint32_t)params->dst_stride,
                       (uint32_t)params->lead_rows,
                       (uint32_t)params->lead_cols,
                       (uint32_t)params->bd,
                       (uint32_t)params->class_id_stride,
                       (uint32_t)params->num_sym,
                       (uint32_t)params->singleton_index,
                       (uint32_t)params->prec_bits,
                       (uint32_t)params->num_taps,
                       (uint32_t)(w * h) };
  const uint32_t groups =
      (uint32_t)((w * h + PCWIENER_WORKGROUP - 1) / PCWIENER_WORKGROUP);
  const int rc = avm_gpu_pipeline_dispatch(ctx, pc->pipe, bufs, 6, &p,
                                           sizeof(p), groups, err);
  if (rc != 0) return rc;

  const uint32_t *res = pc->dst.mapped;
  for (size_t i = 0; i < img_n; i++) ((uint16_t *)dst_plane)[i] = (uint16_t)res[i];
  return 0;
}
