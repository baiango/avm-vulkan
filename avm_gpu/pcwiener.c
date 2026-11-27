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
#ifndef AVM_GPU_PCWCLS_SPV_PATH
#define AVM_GPU_PCWCLS_SPV_PATH "pcwcls.spv"
#endif

#define PCWIENER_WORKGROUP 64

struct avm_gpu_pcwiener {
  avm_gpu_pipeline *pipe;
  avm_gpu_buf img, cid, sel, flt, geo, dst;
  VkDeviceSize img_cap, cid_cap, sel_cap, flt_cap, geo_cap, dst_cap;
  /* Classification stage. */
  avm_gpu_pipeline *cls_pipe;
  avm_gpu_buf c_img, c_tsk, c_lut, c_norm, c_thr, c_lutcls, c_out;
  VkDeviceSize c_img_cap, c_tsk_cap, c_lut_cap, c_norm_cap, c_thr_cap;
  VkDeviceSize c_lutcls_cap, c_out_cap;
};

typedef struct push_constants {
  uint32_t width, height, stride, dst_stride, lead_rows, lead_cols;
  uint32_t bd, cid_stride, num_sym, singleton_index, prec_bits, num_taps;
  uint32_t count;
} push_constants;

typedef struct tap_geo {
  int32_t diff, pos;
} tap_geo;

typedef struct cls_constants {
  uint32_t width, height, stride, lead_rows, lead_cols, tskip_stride;
  uint32_t use_tskip, bd, cell_stride, cell_count, prec_feature, thr_shift;
} cls_constants;

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
  if (avm_gpu_pipeline_create(ctx, AVM_GPU_PCWCLS_SPV_PATH,
                              "AVM_GPU_PCWCLS_SPV", 7, sizeof(cls_constants),
                              &pc->cls_pipe, err) != 0) {
    avm_gpu_pipeline_destroy(ctx, pc->pipe);
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
  avm_gpu_pipeline_destroy(ctx, pc->cls_pipe);
  avm_gpu_buf_destroy(ctx, &pc->c_img);
  avm_gpu_buf_destroy(ctx, &pc->c_tsk);
  avm_gpu_buf_destroy(ctx, &pc->c_lut);
  avm_gpu_buf_destroy(ctx, &pc->c_norm);
  avm_gpu_buf_destroy(ctx, &pc->c_thr);
  avm_gpu_buf_destroy(ctx, &pc->c_lutcls);
  avm_gpu_buf_destroy(ctx, &pc->c_out);
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

int avm_gpu_pcwiener_classify(
    avm_gpu_ctx *ctx, avm_gpu_pcwiener *pc,
    const avm_gpu_pcwiener_classify_params *params, const uint16_t *dgd,
    const uint8_t *tskip, uint8_t *class_id, char err[AVM_GPU_ERR_LEN]) {
  if (!params || !dgd || !class_id || !params->qval_lut ||
      !params->normalizers || !params->thresholds || !params->lut_to_class) {
    snprintf(err, AVM_GPU_ERR_LEN, "invalid arguments");
    return -1;
  }
  const int w = params->width, h = params->height;
  if (w < 4 || h < 4 || params->stride < w + 2 + params->lead_cols ||
      params->rows < h + 2 + params->lead_rows || params->lead_rows < 1 ||
      params->cell_stride * 4 < w || params->cell_rows * 4 < h) {
    snprintf(err, AVM_GPU_ERR_LEN, "invalid geometry");
    return -1;
  }
  const int use_tskip = params->tskip_stride > 0 && tskip != NULL;
  const size_t img_n = (size_t)params->rows * (size_t)params->stride;
  const size_t tsk_n =
      use_tskip ? (size_t)params->cell_rows * params->tskip_stride : 4;
  const size_t cell_n = (size_t)params->cell_stride * params->cell_rows;

  if (ensure_buf(ctx, &pc->c_img, &pc->c_img_cap, img_n * 4, err) ||
      ensure_buf(ctx, &pc->c_tsk, &pc->c_tsk_cap, tsk_n * 4, err) ||
      ensure_buf(ctx, &pc->c_lut, &pc->c_lut_cap, 256 * 4 * 4, err) ||
      ensure_buf(ctx, &pc->c_norm, &pc->c_norm_cap, 5 * 4, err) ||
      ensure_buf(ctx, &pc->c_thr, &pc->c_thr_cap, 4 * 4, err) ||
      ensure_buf(ctx, &pc->c_lutcls, &pc->c_lutcls_cap, 4096 * 4, err) ||
      ensure_buf(ctx, &pc->c_out, &pc->c_out_cap, cell_n * 4, err)) {
    return -1;
  }

  const uint16_t *plane = dgd - (size_t)params->lead_rows * params->stride -
                          params->lead_cols;
  uint32_t *img = pc->c_img.mapped;
  for (size_t i = 0; i < img_n; i++) img[i] = plane[i];
  uint32_t *tsk = pc->c_tsk.mapped;
  for (size_t i = 0; i < tsk_n; i++) tsk[i] = use_tskip ? tskip[i] : 0u;
  int32_t *lut = pc->c_lut.mapped;
  for (int i = 0; i < 256 * 4; i++) lut[i] = params->qval_lut[i];
  int32_t *norm = pc->c_norm.mapped;
  for (int i = 0; i < 5; i++) norm[i] = params->normalizers[i];
  int32_t *thr = pc->c_thr.mapped;
  for (int i = 0; i < 4; i++) thr[i] = params->thresholds[i];
  uint32_t *lutcls = pc->c_lutcls.mapped;
  for (int i = 0; i < 4096; i++) lutcls[i] = params->lut_to_class[i];

  const avm_gpu_buf bufs[7] = { pc->c_img,    pc->c_tsk,     pc->c_lut,
                                pc->c_norm,   pc->c_thr,     pc->c_lutcls,
                                pc->c_out };
  cls_constants p = { (uint32_t)w,
                      (uint32_t)h,
                      (uint32_t)params->stride,
                      (uint32_t)params->lead_rows,
                      (uint32_t)params->lead_cols,
                      (uint32_t)(use_tskip ? params->tskip_stride : 0),
                      (uint32_t)use_tskip,
                      (uint32_t)params->bd,
                      (uint32_t)params->cell_stride,
                      (uint32_t)cell_n,
                      (uint32_t)params->prec_feature,
                      (uint32_t)params->thr_shift };
  const uint32_t groups = (uint32_t)((cell_n + 16 - 1) / 16);
  const int rc = avm_gpu_pipeline_dispatch(ctx, pc->cls_pipe, bufs, 7, &p,
                                           sizeof(p), groups, err);
  if (rc != 0) return rc;

  const uint32_t *res = pc->c_out.mapped;
  for (size_t i = 0; i < cell_n; i++) class_id[i] = (uint8_t)res[i];
  return 0;
}
