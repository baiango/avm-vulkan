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

#include "avm_gpu/arena.h"
#include "avm_gpu/pipeline.h"

#ifndef AVM_GPU_PCWIENER_SPV_PATH
#define AVM_GPU_PCWIENER_SPV_PATH "pcwiener.spv"
#endif
#ifndef AVM_GPU_PCWCLS_SPV_PATH
#define AVM_GPU_PCWCLS_SPV_PATH "pcwcls.spv"
#endif

#define PCWIENER_WORKGROUP 64

struct avm_gpu_pcwiener {
  avm_gpu_arena *arena; /* device-local arena when AV2_GPU_ARENA=1 */
  uint32_t *h_img, *h_cid, *h_sel, *h_flt, *h_geo, *h_dst, *h_skip;
  uint32_t *h_cimg, *h_ctsk, *h_clut, *h_cnorm, *h_cthr, *h_clutcls, *h_cout;
  size_t h_img_cap, h_cid_cap, h_sel_cap, h_flt_cap, h_geo_cap, h_dst_cap,
      h_skip_cap;
  size_t h_cimg_cap, h_ctsk_cap, h_clut_cap, h_cnorm_cap, h_cthr_cap,
      h_clutcls_cap, h_cout_cap;
  avm_gpu_pipeline *pipe;
  avm_gpu_buf img, cid, sel, flt, geo, dst, skip;
  VkDeviceSize img_cap, cid_cap, sel_cap, flt_cap, geo_cap, dst_cap, skip_cap;
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
  uint32_t use_skip;
} push_constants;

typedef struct tap_geo {
  int32_t diff, pos;
} tap_geo;

typedef struct cls_constants {
  uint32_t width, height, stride, lead_rows, lead_cols, tskip_stride;
  uint32_t use_tskip, bd, cell_stride, cell_count, prec_feature, thr_shift;
} cls_constants;

static int ensure_buf(avm_gpu_ctx *ctx, avm_gpu_arena *arena, avm_gpu_buf *buf,
                      VkDeviceSize *cap, VkDeviceSize need,
                      char err[AVM_GPU_ERR_LEN]) {
  if (*cap >= need) return 0;
  avm_gpu_buf_destroy(ctx, buf);
  return avm_gpu_slot_reserve(ctx, arena, buf, cap, need, err);
}

int avm_gpu_pcwiener_create(avm_gpu_ctx *ctx, avm_gpu_pcwiener **out,
                            char err[AVM_GPU_ERR_LEN]) {
  *out = NULL;
  avm_gpu_pcwiener *pc = calloc(1, sizeof(*pc));
  if (!pc) {
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    return -1;
  }
  if (avm_gpu_arena_wanted("pcwiener")) {
    const char *mb = getenv("AV2_GPU_ARENA_MB");
    const VkDeviceSize bytes =
        (VkDeviceSize)(mb && *mb ? atoi(mb) : 64) * 1024u * 1024u;
    char aerr[AVM_GPU_ERR_LEN];
    if (avm_gpu_arena_create(ctx, bytes, &pc->arena, aerr) != 0) {
      pc->arena = NULL;
    }
  }
  if (avm_gpu_pipeline_create(ctx, AVM_GPU_PCWIENER_SPV_PATH,
                              "AVM_GPU_PCWIENER_SPV", 7, sizeof(push_constants),
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
  avm_gpu_buf_destroy(ctx, &pc->skip);
  avm_gpu_pipeline_destroy(ctx, pc->cls_pipe);
  avm_gpu_buf_destroy(ctx, &pc->c_img);
  avm_gpu_buf_destroy(ctx, &pc->c_tsk);
  avm_gpu_buf_destroy(ctx, &pc->c_lut);
  avm_gpu_buf_destroy(ctx, &pc->c_norm);
  avm_gpu_buf_destroy(ctx, &pc->c_thr);
  avm_gpu_buf_destroy(ctx, &pc->c_lutcls);
  if (pc->h_img != pc->img.mapped) free(pc->h_img);
  if (pc->h_cid != pc->cid.mapped) free(pc->h_cid);
  if (pc->h_sel != pc->sel.mapped) free(pc->h_sel);
  if (pc->h_flt != pc->flt.mapped) free(pc->h_flt);
  if (pc->h_geo != pc->geo.mapped) free(pc->h_geo);
  if (pc->h_dst != pc->dst.mapped) free(pc->h_dst);
  if (pc->h_skip != pc->skip.mapped) free(pc->h_skip);
  if (pc->h_cimg != pc->c_img.mapped) free(pc->h_cimg);
  if (pc->h_ctsk != pc->c_tsk.mapped) free(pc->h_ctsk);
  if (pc->h_clut != pc->c_lut.mapped) free(pc->h_clut);
  if (pc->h_cnorm != pc->c_norm.mapped) free(pc->h_cnorm);
  if (pc->h_cthr != pc->c_thr.mapped) free(pc->h_cthr);
  if (pc->h_clutcls != pc->c_lutcls.mapped) free(pc->h_clutcls);
  if (pc->h_cout != pc->c_out.mapped) free(pc->h_cout);
  avm_gpu_buf_destroy(ctx, &pc->c_out);
  avm_gpu_arena_destroy(ctx, pc->arena);
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

  const VkDeviceSize f_img_bytes = img_n * 4, f_cid_bytes = cid_n * 4;
  const VkDeviceSize f_sel_bytes = 256 * 4, f_flt_bytes = filter_words * 4;
  const VkDeviceSize f_geo_bytes = sizeof(geo), f_dst_bytes = img_n * 4;
  const VkDeviceSize f_skip_bytes = (VkDeviceSize)cid_n * 4;
  if (ensure_buf(ctx, pc->arena, &pc->img, &pc->img_cap, f_img_bytes, err) ||
      ensure_buf(ctx, pc->arena, &pc->cid, &pc->cid_cap, f_cid_bytes, err) ||
      ensure_buf(ctx, pc->arena, &pc->sel, &pc->sel_cap, f_sel_bytes, err) ||
      ensure_buf(ctx, pc->arena, &pc->flt, &pc->flt_cap, f_flt_bytes, err) ||
      ensure_buf(ctx, pc->arena, &pc->geo, &pc->geo_cap, f_geo_bytes, err) ||
      ensure_buf(ctx, pc->arena, &pc->dst, &pc->dst_cap, f_dst_bytes, err) ||
      ensure_buf(ctx, pc->arena, &pc->skip, &pc->skip_cap, f_skip_bytes, err)) {
    return -1;
  }
  if ((!pc->img.mapped &&
       avm_gpu_shadow_reserve(&pc->h_img, &pc->h_img_cap, f_img_bytes, err)) ||
      (!pc->cid.mapped &&
       avm_gpu_shadow_reserve(&pc->h_cid, &pc->h_cid_cap, f_cid_bytes, err)) ||
      (!pc->sel.mapped &&
       avm_gpu_shadow_reserve(&pc->h_sel, &pc->h_sel_cap, f_sel_bytes, err)) ||
      (!pc->flt.mapped &&
       avm_gpu_shadow_reserve(&pc->h_flt, &pc->h_flt_cap, f_flt_bytes, err)) ||
      (!pc->geo.mapped &&
       avm_gpu_shadow_reserve(&pc->h_geo, &pc->h_geo_cap, f_geo_bytes, err)) ||
      (!pc->dst.mapped &&
       avm_gpu_shadow_reserve(&pc->h_dst, &pc->h_dst_cap, f_dst_bytes, err)) ||
      (!pc->skip.mapped &&
       avm_gpu_shadow_reserve(&pc->h_skip, &pc->h_skip_cap, f_skip_bytes, err))) {
    return -1;
  }

  /* The plane is uploaded as one 16-bit sample per 32-bit word (M0). */
  const uint16_t *plane = dgd - (size_t)params->lead_rows * params->stride -
                          params->lead_cols;
  uint32_t *img = pc->img.mapped ? pc->img.mapped : pc->h_img;
  for (size_t i = 0; i < img_n; i++) img[i] = plane[i];
  uint32_t *cid = pc->cid.mapped ? pc->cid.mapped : pc->h_cid;
  for (size_t i = 0; i < cid_n; i++) cid[i] = class_id[i];
  uint32_t *sel = pc->sel.mapped ? pc->sel.mapped : pc->h_sel;
  for (int i = 0; i < 256; i++) sel[i] = filter_selector[i];
  int32_t *flt = (int32_t *)(pc->flt.mapped ? pc->flt.mapped : pc->h_flt);
  size_t taps = 0;
  for (int f = 0; f < 256; f++) {
    for (int t = 0; t < params->num_taps; t++) {
      flt[taps++] = filters[(size_t)f * params->num_taps + t];
    }
  }
  uint32_t *geobuf = pc->geo.mapped ? pc->geo.mapped : pc->h_geo;
  memcpy(geobuf, geo, sizeof(geo));
  uint32_t *skip = pc->skip.mapped ? pc->skip.mapped : pc->h_skip;
  for (size_t i = 0; i < cid_n; i++) {
    skip[i] = params->skip_block ? params->skip_block[i] : 0u;
  }

  const uint16_t *dst_plane = dst - (size_t)params->lead_rows * params->dst_stride -
                              params->lead_cols;
  uint32_t *d32 = pc->dst.mapped ? pc->dst.mapped : pc->h_dst;
  for (size_t i = 0; i < img_n; i++) d32[i] = dst_plane[i];

  if (avm_gpu_shadow_push(ctx, pc->arena, &pc->img, img, f_img_bytes, err) ||
      avm_gpu_shadow_push(ctx, pc->arena, &pc->cid, cid, f_cid_bytes, err) ||
      avm_gpu_shadow_push(ctx, pc->arena, &pc->sel, sel, f_sel_bytes, err) ||
      avm_gpu_shadow_push(ctx, pc->arena, &pc->flt, (uint32_t *)flt,
                          f_flt_bytes, err) ||
      avm_gpu_shadow_push(ctx, pc->arena, &pc->geo, geobuf, f_geo_bytes, err) ||
      avm_gpu_shadow_push(ctx, pc->arena, &pc->dst, d32, f_dst_bytes, err) ||
      avm_gpu_shadow_push(ctx, pc->arena, &pc->skip, skip, f_skip_bytes, err)) {
    return -1;
  }
  const avm_gpu_buf bufs[7] = { pc->img, pc->cid,  pc->sel, pc->flt,
                                pc->geo, pc->dst,  pc->skip };
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
                       (uint32_t)(w * h),
                       params->skip_block ? 1u : 0u };
  const uint32_t groups =
      (uint32_t)((w * h + PCWIENER_WORKGROUP - 1) / PCWIENER_WORKGROUP);
  const int rc = avm_gpu_pipeline_dispatch(ctx, pc->pipe, bufs, 7, &p,
                                           sizeof(p), groups, err);
  if (rc != 0) return rc;

  if (avm_gpu_shadow_pull(ctx, pc->arena, &pc->dst, d32, f_dst_bytes, err)) {
    return -1;
  }
  const uint32_t *res = d32;
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

  if (ensure_buf(ctx, pc->arena, &pc->c_img, &pc->c_img_cap, img_n * 4, err) ||
      ensure_buf(ctx, pc->arena, &pc->c_tsk, &pc->c_tsk_cap, tsk_n * 4, err) ||
      ensure_buf(ctx, pc->arena, &pc->c_lut, &pc->c_lut_cap, 256 * 4 * 4, err) ||
      ensure_buf(ctx, pc->arena, &pc->c_norm, &pc->c_norm_cap, 5 * 4, err) ||
      ensure_buf(ctx, pc->arena, &pc->c_thr, &pc->c_thr_cap, 4 * 4, err) ||
      ensure_buf(ctx, pc->arena, &pc->c_lutcls, &pc->c_lutcls_cap, 4096 * 4,
                 err) ||
      ensure_buf(ctx, pc->arena, &pc->c_out, &pc->c_out_cap, cell_n * 4, err)) {
    return -1;
  }
  if ((!pc->c_img.mapped && avm_gpu_shadow_reserve(&pc->h_cimg,
                                                   &pc->h_cimg_cap,
                                                   img_n * 4, err)) ||
      (!pc->c_tsk.mapped && avm_gpu_shadow_reserve(&pc->h_ctsk,
                                                   &pc->h_ctsk_cap,
                                                   tsk_n * 4, err)) ||
      (!pc->c_lut.mapped && avm_gpu_shadow_reserve(&pc->h_clut,
                                                   &pc->h_clut_cap,
                                                   256 * 4 * 4, err)) ||
      (!pc->c_norm.mapped && avm_gpu_shadow_reserve(&pc->h_cnorm,
                                                    &pc->h_cnorm_cap, 5 * 4,
                                                    err)) ||
      (!pc->c_thr.mapped && avm_gpu_shadow_reserve(&pc->h_cthr,
                                                   &pc->h_cthr_cap, 4 * 4,
                                                   err)) ||
      (!pc->c_lutcls.mapped && avm_gpu_shadow_reserve(&pc->h_clutcls,
                                                      &pc->h_clutcls_cap,
                                                      4096 * 4, err)) ||
      (!pc->c_out.mapped && avm_gpu_shadow_reserve(&pc->h_cout,
                                                   &pc->h_cout_cap,
                                                   cell_n * 4, err))) {
    return -1;
  }
  const VkDeviceSize c_img_bytes = img_n * 4, c_tsk_bytes = tsk_n * 4;
  const VkDeviceSize c_lut_bytes = 256 * 4 * 4, c_norm_bytes = 5 * 4;
  const VkDeviceSize c_thr_bytes = 4 * 4, c_lutcls_bytes = 4096 * 4;
  const VkDeviceSize c_out_bytes = cell_n * 4;

  const uint16_t *plane = dgd - (size_t)params->lead_rows * params->stride -
                          params->lead_cols;
  uint32_t *img = pc->c_img.mapped ? pc->c_img.mapped : pc->h_cimg;
  for (size_t i = 0; i < img_n; i++) img[i] = plane[i];
  uint32_t *tsk = pc->c_tsk.mapped ? pc->c_tsk.mapped : pc->h_ctsk;
  for (size_t i = 0; i < tsk_n; i++) tsk[i] = use_tskip ? tskip[i] : 0u;
  int32_t *lut = (int32_t *)(pc->c_lut.mapped ? pc->c_lut.mapped : pc->h_clut);
  for (int i = 0; i < 256 * 4; i++) lut[i] = params->qval_lut[i];
  int32_t *norm =
      (int32_t *)(pc->c_norm.mapped ? pc->c_norm.mapped : pc->h_cnorm);
  for (int i = 0; i < 5; i++) norm[i] = params->normalizers[i];
  int32_t *thr =
      (int32_t *)(pc->c_thr.mapped ? pc->c_thr.mapped : pc->h_cthr);
  for (int i = 0; i < 4; i++) thr[i] = params->thresholds[i];
  uint32_t *lutcls =
      pc->c_lutcls.mapped ? pc->c_lutcls.mapped : pc->h_clutcls;
  for (int i = 0; i < 4096; i++) lutcls[i] = params->lut_to_class[i];

  if (avm_gpu_shadow_push(ctx, pc->arena, &pc->c_img, img, c_img_bytes, err) ||
      avm_gpu_shadow_push(ctx, pc->arena, &pc->c_tsk, tsk, c_tsk_bytes, err) ||
      avm_gpu_shadow_push(ctx, pc->arena, &pc->c_lut, (uint32_t *)lut,
                          c_lut_bytes, err) ||
      avm_gpu_shadow_push(ctx, pc->arena, &pc->c_norm, (uint32_t *)norm,
                          c_norm_bytes, err) ||
      avm_gpu_shadow_push(ctx, pc->arena, &pc->c_thr, (uint32_t *)thr,
                          c_thr_bytes, err) ||
      avm_gpu_shadow_push(ctx, pc->arena, &pc->c_lutcls, lutcls, c_lutcls_bytes,
                          err)) {
    return -1;
  }
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

  uint32_t *cout = pc->c_out.mapped ? pc->c_out.mapped : pc->h_cout;
  if (avm_gpu_shadow_pull(ctx, pc->arena, &pc->c_out, cout, c_out_bytes, err)) {
    return -1;
  }
  const uint32_t *res = cout;
  for (size_t i = 0; i < cell_n; i++) class_id[i] = (uint8_t)res[i];
  return 0;
}
