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

#include "avm_gpu/vulkan_warp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "avm_gpu/pipeline.h"

#ifndef AVM_GPU_WARP_SPV_PATH
#define AVM_GPU_WARP_SPV_PATH "warp.spv"
#endif
#ifndef AVM_GPU_WARP_EXT_SPV_PATH
#define AVM_GPU_WARP_EXT_SPV_PATH "warp_ext.spv"
#endif
#ifndef AVM_GPU_WARP_SCALED_SPV_PATH
#define AVM_GPU_WARP_SCALED_SPV_PATH "warp_scaled.spv"
#endif

struct avm_gpu_warp {
  avm_gpu_pipeline *pipe;
  avm_gpu_pipeline *ext_pipe;
  avm_gpu_pipeline *scaled_pipe;
  avm_gpu_buf box;
  avm_gpu_buf conv;
  VkDeviceSize conv_cap;
  avm_gpu_buf ext_filt;
  avm_gpu_buf ref, pred, filt;
  VkDeviceSize ref_cap, pred_cap;
  uint32_t *expand;
  size_t expand_cap;
};

typedef struct push_constants {
  uint32_t width, height, stride, p_col, p_row, p_width, p_height, p_stride,
      bd, round_0;
  int32_t m0, m1, m2, m3, m4, m5;
  int32_t alpha, beta, gamma, delta;
  uint32_t ss_x, ss_y;
  uint32_t round_1, compound, do_average, conv_stride, uneven;
  int32_t fwd, bck;
} push_constants;

/* Compound prediction state shared by the three warp kernels (the public
 * typedef lives in vulkan_warp.h). */
typedef avm_gpu_warp_compound warp_compound;

static int warp_upload_conv(avm_gpu_ctx *ctx, avm_gpu_warp *w,
                            const warp_compound *cmp, int h,
                            char err[AVM_GPU_ERR_LEN]) {
  /* A single-reference call binds a block-sized dummy (a one-entry range made
   * the driver return garbage for a shader that declares the binding). */
  const size_t n = cmp ? (size_t)cmp->conv_stride * (size_t)h : 64 * 64;
  if (w->conv_cap < (VkDeviceSize)n * 4) {
    avm_gpu_buf_destroy(ctx, &w->conv);
    if (avm_gpu_buf_create(ctx, (VkDeviceSize)n * 4, &w->conv, err) != 0) {
      return -1;
    }
    w->conv_cap = w->conv.size;
  }
  if (cmp && cmp->conv) {
    uint32_t *cp = w->conv.mapped;
    for (size_t i = 0; i < n; i++) cp[i] = cmp->conv[i];
  }
  return 0;
}

static void warp_download_conv(avm_gpu_warp *w, const warp_compound *cmp,
                               int wd, int h) {
  if (!cmp || !cmp->conv || cmp->do_average) return;
  const uint32_t *cp = w->conv.mapped;
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < wd; x++) {
      cmp->conv[(size_t)y * cmp->conv_stride + x] =
          (uint16_t)cp[(size_t)y * cmp->conv_stride + x];
    }
  }
}

int avm_gpu_warp_create(avm_gpu_ctx *ctx, avm_gpu_warp **out,
                        char err[AVM_GPU_ERR_LEN]) {
  *out = NULL;
  avm_gpu_warp *w = calloc(1, sizeof(*w));
  if (!w) {
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    return -1;
  }
  if (avm_gpu_pipeline_create(ctx, AVM_GPU_WARP_SPV_PATH, "AVM_GPU_WARP_SPV",
                              4, sizeof(push_constants), &w->pipe,
                              err) != 0) {
    free(w);
    return -1;
  }
  if (avm_gpu_pipeline_create(ctx, AVM_GPU_WARP_EXT_SPV_PATH,
                              "AVM_GPU_WARP_EXT_SPV", 5, 26 * 4, &w->ext_pipe,
                              err) != 0) {
    avm_gpu_warp_destroy(ctx, w);
    return -1;
  }
  if (avm_gpu_pipeline_create(ctx, AVM_GPU_WARP_SCALED_SPV_PATH,
                              "AVM_GPU_WARP_SCALED_SPV", 5, 29 * 4,
                              &w->scaled_pipe, err) != 0) {
    avm_gpu_warp_destroy(ctx, w);
    return -1;
  }
  *out = w;
  return 0;
}

void avm_gpu_warp_destroy(avm_gpu_ctx *ctx, avm_gpu_warp *w) {
  if (!w) return;
  avm_gpu_pipeline_destroy(ctx, w->pipe);
  avm_gpu_pipeline_destroy(ctx, w->ext_pipe);
  avm_gpu_pipeline_destroy(ctx, w->scaled_pipe);
  avm_gpu_buf_destroy(ctx, &w->box);
  avm_gpu_buf_destroy(ctx, &w->conv);
  avm_gpu_buf_destroy(ctx, &w->ext_filt);
  avm_gpu_buf_destroy(ctx, &w->ref);
  avm_gpu_buf_destroy(ctx, &w->pred);
  avm_gpu_buf_destroy(ctx, &w->filt);
  free(w->expand);
  free(w);
}

int avm_gpu_warp_affine(avm_gpu_ctx *ctx, avm_gpu_warp *w,
                        const int32_t mat[6], const uint16_t *ref, int width,
                        int height, int stride, uint16_t *pred, int p_col,
                        int p_row, int p_width, int p_height, int p_stride,
                        int bd, int round_0, int round_1, int alpha, int beta,
                        int gamma, int delta, int ss_x, int ss_y,
                        const int16_t *filter, const warp_compound *cmp,
                        char err[AVM_GPU_ERR_LEN]) {
  if (!mat || !ref || !pred || !filter) {
    snprintf(err, AVM_GPU_ERR_LEN, "invalid arguments");
    return -1;
  }
  if (p_width <= 0 || p_height <= 0 || (p_width & 3) || (p_height & 3)) {
    snprintf(err, AVM_GPU_ERR_LEN, "unsupported warp size %dx%d", p_width,
             p_height);
    return -1;
  }
  if (stride < width || p_stride < p_width) {
    snprintf(err, AVM_GPU_ERR_LEN, "stride too small");
    return -1;
  }

  const size_t ref_n = (size_t)stride * height;
  const size_t pred_n = (size_t)p_stride * (p_row + p_height);
  if (ref_n > w->expand_cap) {
    free(w->expand);
    w->expand = malloc(ref_n * sizeof(*w->expand));
    w->expand_cap = w->expand ? ref_n : 0;
    if (!w->expand) {
      snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
      return -1;
    }
  }

  if (w->ref_cap < (VkDeviceSize)ref_n * 4) {
    avm_gpu_buf_destroy(ctx, &w->ref);
    if (avm_gpu_buf_create(ctx, (VkDeviceSize)ref_n * 4, &w->ref, err) != 0) {
      return -1;
    }
    w->ref_cap = w->ref.size;
  }
  if (w->pred_cap < (VkDeviceSize)pred_n * 4) {
    avm_gpu_buf_destroy(ctx, &w->pred);
    if (avm_gpu_buf_create(ctx, (VkDeviceSize)pred_n * 4, &w->pred, err) != 0) {
      return -1;
    }
    w->pred_cap = w->pred.size;
  }
  if (!w->ext_filt.buffer) {
    if (avm_gpu_buf_create(ctx, AVM_GPU_WARP_EXT_FILTER_ROWS * 8 * 4,
                           &w->ext_filt, err) != 0) {
      return -1;
    }
  }
  if (!w->filt.buffer) {
    if (avm_gpu_buf_create(ctx, AVM_GPU_WARP_FILTER_ROWS * 8 * 4, &w->filt,
                           err) != 0) {
      return -1;
    }
  }

  for (size_t i = 0; i < ref_n; i++) w->expand[i] = ref[i];
  memcpy(w->ref.mapped, w->expand, ref_n * 4);
  uint32_t *pp = w->pred.mapped;
  for (size_t i = 0; i < pred_n; i++) pp[i] = pred[i];
  int32_t *fp = w->filt.mapped;
  for (int i = 0; i < AVM_GPU_WARP_FILTER_ROWS * 8; i++) fp[i] = filter[i];

  if (warp_upload_conv(ctx, w, cmp, p_row + p_height, err) != 0) return -1;
  const avm_gpu_buf bufs[4] = { w->ref, w->pred, w->filt, w->conv };
  push_constants pc = { (uint32_t)width,
                        (uint32_t)height,
                        (uint32_t)stride,
                        (uint32_t)p_col,
                        (uint32_t)p_row,
                        (uint32_t)p_width,
                        (uint32_t)p_height,
                        (uint32_t)p_stride,
                        (uint32_t)bd,
                        (uint32_t)round_0,
                        mat[0],
                        mat[1],
                        mat[2],
                        mat[3],
                        mat[4],
                        mat[5],
                        alpha,
                        beta,
                        gamma,
                        delta,
                        (uint32_t)ss_x,
                        (uint32_t)ss_y,
                        (uint32_t)round_1,
                        (uint32_t)(cmp ? 1 : 0),
                        (uint32_t)(cmp && cmp->do_average),
                        (uint32_t)(cmp ? cmp->conv_stride : 0),
                        (uint32_t)(cmp && cmp->uneven),
                        cmp ? cmp->fwd : 0,
                        cmp ? cmp->bck : 0 };
  const uint32_t gx = (uint32_t)((p_width + 7) / 8);
  const uint32_t gy = (uint32_t)((p_height + 7) / 8);
  const int rc = avm_gpu_pipeline_dispatch_2d(ctx, w->pipe, bufs, 4, &pc,
                                              sizeof(pc), gx, gy, err);
  if (rc != 0) return rc;

  for (size_t i = 0; i < pred_n; i++) pred[i] = (uint16_t)pp[i];
  warp_download_conv(w, cmp, p_width, p_height);
  return 0;
}

typedef struct ext_push_constants {
  uint32_t width, height, stride, p_col, p_row, p_width, p_height, p_stride,
      bd, round_0;
  int32_t m0, m1, m2, m3, m4, m5;
  uint32_t ss_x, ss_y;
  uint32_t round_1, compound, do_average, conv_stride, uneven;
  int32_t fwd, bck;
  uint32_t use_box;
} ext_push_constants;

int avm_gpu_warp_affine_ext(avm_gpu_ctx *ctx, avm_gpu_warp *w,
                            const int32_t mat[6], const uint16_t *ref,
                            int width, int height, int stride, uint16_t *pred,
                            int p_col, int p_row, int p_width, int p_height,
                            int p_stride, int bd, int round_0, int round_1,
                            int ss_x, int ss_y, const int16_t *filter,
                            const int32_t *box, int use_box,
                            const warp_compound *cmp, char err[AVM_GPU_ERR_LEN]) {
  if (!mat || !ref || !pred || !filter) {
    snprintf(err, AVM_GPU_ERR_LEN, "invalid arguments");
    return -1;
  }
  if (p_width <= 0 || p_height <= 0 || (p_width & 3) || (p_height & 3)) {
    snprintf(err, AVM_GPU_ERR_LEN, "unsupported warp size %dx%d", p_width,
             p_height);
    return -1;
  }

  const size_t ref_n = (size_t)stride * height;
  const size_t pred_n = (size_t)p_stride * (p_row + p_height);
  if (ref_n > w->expand_cap) {
    free(w->expand);
    w->expand = malloc(ref_n * sizeof(*w->expand));
    w->expand_cap = w->expand ? ref_n : 0;
    if (!w->expand) {
      snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
      return -1;
    }
  }
  if (w->ref_cap < (VkDeviceSize)ref_n * 4) {
    avm_gpu_buf_destroy(ctx, &w->ref);
    if (avm_gpu_buf_create(ctx, (VkDeviceSize)ref_n * 4, &w->ref, err) != 0) {
      return -1;
    }
    w->ref_cap = w->ref.size;
  }
  if (w->pred_cap < (VkDeviceSize)pred_n * 4) {
    avm_gpu_buf_destroy(ctx, &w->pred);
    if (avm_gpu_buf_create(ctx, (VkDeviceSize)pred_n * 4, &w->pred, err) != 0) {
      return -1;
    }
    w->pred_cap = w->pred.size;
  }
  if (!w->ext_filt.buffer) {
    if (avm_gpu_buf_create(ctx, AVM_GPU_WARP_EXT_FILTER_ROWS * 8 * 4,
                           &w->ext_filt, err) != 0) {
      return -1;
    }
  }

  for (size_t i = 0; i < ref_n; i++) w->expand[i] = ref[i];
  memcpy(w->ref.mapped, w->expand, ref_n * 4);
  uint32_t *pp = w->pred.mapped;
  for (size_t i = 0; i < pred_n; i++) pp[i] = pred[i];
  int32_t *fp = w->ext_filt.mapped;
  for (int i = 0; i < AVM_GPU_WARP_EXT_FILTER_ROWS * 8; i++) fp[i] = filter[i];

  if (warp_upload_conv(ctx, w, cmp, p_row + p_height, err) != 0) return -1;
  if (use_box) {
    if (!box) {
      snprintf(err, AVM_GPU_ERR_LEN, "boundary box missing");
      return -1;
    }
    if (!w->box.buffer) {
      if (avm_gpu_buf_create(ctx, 32 * 32 * 4 * sizeof(int32_t), &w->box, err) !=
          0) {
        return -1;
      }
    }
    memcpy(w->box.mapped, box, 32 * 32 * 4 * sizeof(int32_t));
  }
  const avm_gpu_buf bufs[5] = { w->ref, w->pred,   w->ext_filt,
                                 w->conv, w->box };
  ext_push_constants pc = { (uint32_t)width,
                            (uint32_t)height,
                            (uint32_t)stride,
                            (uint32_t)p_col,
                            (uint32_t)p_row,
                            (uint32_t)p_width,
                            (uint32_t)p_height,
                            (uint32_t)p_stride,
                            (uint32_t)bd,
                            (uint32_t)round_0,
                            mat[0],
                            mat[1],
                            mat[2],
                            mat[3],
                            mat[4],
                            mat[5],
                            (uint32_t)ss_x,
                            (uint32_t)ss_y,
                            (uint32_t)round_1,
                            (uint32_t)(cmp ? 1 : 0),
                            (uint32_t)(cmp && cmp->do_average),
                            (uint32_t)(cmp ? cmp->conv_stride : 0),
                            (uint32_t)(cmp && cmp->uneven),
                            cmp ? cmp->fwd : 0,
                            cmp ? cmp->bck : 0,
                            (uint32_t)(use_box ? 1 : 0) };
  const uint32_t gx = (uint32_t)((p_width + 3) / 4);
  const uint32_t gy = (uint32_t)((p_height + 3) / 4);
  const int rc = avm_gpu_pipeline_dispatch_2d(ctx, w->ext_pipe, bufs, 5, &pc,
                                              sizeof(pc), gx, gy, err);
  if (rc != 0) return rc;

  for (size_t i = 0; i < pred_n; i++) pred[i] = (uint16_t)pp[i];
  warp_download_conv(w, cmp, p_width, p_height);
  return 0;
}

typedef struct scaled_push_constants {
  uint32_t width, height, stride, p_col, p_row, p_width, p_height, p_stride,
      bd, round_0;
  int32_t m0, m1, m2, m3, m4, m5;
  uint32_t ss_x, ss_y;
  int32_t x_scale_fp, y_scale_fp, x_step_qn, y_step_qn;
  uint32_t round_1, compound, do_average, conv_stride, uneven;
  int32_t fwd, bck;
} scaled_push_constants;

/* SCALE_SUBPEL_BITS: the scale factors' 1/16 steps carry into 16-bit sub-pel
 * positions as x_step_q4 << (WARPEDMODEL_PREC_BITS - SCALE_SUBPEL_BITS). */
#define AVM_GPU_SCALE_SUBPEL_BITS 10

int avm_gpu_warp_affine_scaled(avm_gpu_ctx *ctx, avm_gpu_warp *w,
                               const int32_t mat[6], const uint16_t *ref,
                               int width, int height, int stride,
                               uint16_t *pred, int p_col, int p_row,
                               int p_width, int p_height, int p_stride, int bd,
                               int round_0, int round_1, int ss_x,
                               int ss_y, int x_scale_fp, int y_scale_fp,
                               int x_step_q4, int y_step_q4,
                               const int32_t *box, const int16_t *filter,
                               const warp_compound *cmp,
                               char err[AVM_GPU_ERR_LEN]) {
  if (!mat || !ref || !pred || !filter || !box) {
    snprintf(err, AVM_GPU_ERR_LEN, "invalid arguments");
    return -1;
  }
  if (p_width <= 0 || p_height <= 0 || p_width > 128 || p_height > 128) {
    snprintf(err, AVM_GPU_ERR_LEN, "unsupported warp size %dx%d", p_width,
             p_height);
    return -1;
  }
  /* x_step_q4 is in 1/1024 units: 128 is a 2x upscale, 4096 a 16x downscale,
   * the widest ratio valid_ref_frame_size() allows. Above 4096 the horizontal
   * pass would need more than im_block's (2 * 4 + EXT_WARP_TAPS) rows. */
  if (x_step_q4 < 64 || y_step_q4 < 64 || x_step_q4 > 4096 ||
      y_step_q4 > 4096) {
    snprintf(err, AVM_GPU_ERR_LEN, "unsupported scale steps %d/%d", x_step_q4,
             y_step_q4);
    return -1;
  }
  if (stride < width || p_stride < p_width) {
    snprintf(err, AVM_GPU_ERR_LEN, "stride too small");
    return -1;
  }

  const size_t ref_n = (size_t)stride * height;
  const size_t pred_n = (size_t)p_stride * (p_row + p_height);
  if (ref_n > w->expand_cap) {
    free(w->expand);
    w->expand = malloc(ref_n * sizeof(*w->expand));
    w->expand_cap = w->expand ? ref_n : 0;
    if (!w->expand) {
      snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
      return -1;
    }
  }
  if (w->ref_cap < (VkDeviceSize)ref_n * 4) {
    avm_gpu_buf_destroy(ctx, &w->ref);
    if (avm_gpu_buf_create(ctx, (VkDeviceSize)ref_n * 4, &w->ref, err) != 0) {
      return -1;
    }
    w->ref_cap = w->ref.size;
  }
  if (w->pred_cap < (VkDeviceSize)pred_n * 4) {
    avm_gpu_buf_destroy(ctx, &w->pred);
    if (avm_gpu_buf_create(ctx, (VkDeviceSize)pred_n * 4, &w->pred, err) != 0) {
      return -1;
    }
    w->pred_cap = w->pred.size;
  }
  /* The boundary box is a MAX_WARP_BD_SIZE^2 grid of 8x8 sub-block limits. */
  if (!w->ext_filt.buffer) {
    if (avm_gpu_buf_create(ctx, AVM_GPU_WARP_EXT_FILTER_ROWS * 8 * 4,
                           &w->ext_filt, err) != 0) {
      return -1;
    }
  }
  if (!w->box.buffer) {
    if (avm_gpu_buf_create(ctx, 32 * 32 * 4 * sizeof(int32_t), &w->box, err) !=
        0) {
      return -1;
    }
  }

  for (size_t i = 0; i < ref_n; i++) w->expand[i] = ref[i];
  memcpy(w->ref.mapped, w->expand, ref_n * 4);
  uint32_t *pp = w->pred.mapped;
  for (size_t i = 0; i < pred_n; i++) pp[i] = pred[i];
  int32_t *fp = w->ext_filt.mapped;
  for (int i = 0; i < AVM_GPU_WARP_EXT_FILTER_ROWS * 8; i++) fp[i] = filter[i];
  memcpy(w->box.mapped, box, 32 * 32 * 4 * sizeof(int32_t));

  if (warp_upload_conv(ctx, w, cmp, p_row + p_height, err) != 0) return -1;
  const avm_gpu_buf bufs[5] = { w->ref, w->pred, w->ext_filt, w->box,
                                w->conv };
  scaled_push_constants pc = {
    (uint32_t)width,
    (uint32_t)height,
    (uint32_t)stride,
    (uint32_t)p_col,
    (uint32_t)p_row,
    (uint32_t)p_width,
    (uint32_t)p_height,
    (uint32_t)p_stride,
    (uint32_t)bd,
    (uint32_t)round_0,
    mat[0],
    mat[1],
    mat[2],
    mat[3],
    mat[4],
    mat[5],
    (uint32_t)ss_x,
    (uint32_t)ss_y,
    x_scale_fp,
    y_scale_fp,
    x_step_q4 << (16 - AVM_GPU_SCALE_SUBPEL_BITS),
    y_step_q4 << (16 - AVM_GPU_SCALE_SUBPEL_BITS),
    (uint32_t)round_1,
    (uint32_t)(cmp ? 1 : 0),
    (uint32_t)(cmp && cmp->do_average),
    (uint32_t)(cmp ? cmp->conv_stride : 0),
    (uint32_t)(cmp && cmp->uneven),
    cmp ? cmp->fwd : 0,
    cmp ? cmp->bck : 0
  };
  const uint32_t gx = (uint32_t)((p_width + 3) / 4);
  const uint32_t gy = (uint32_t)((p_height + 3) / 4);
  const int rc = avm_gpu_pipeline_dispatch_2d(ctx, w->scaled_pipe, bufs, 5, &pc,
                                              sizeof(pc), gx, gy, err);
  if (rc != 0) return rc;

  for (size_t i = 0; i < pred_n; i++) pred[i] = (uint16_t)pp[i];
  warp_download_conv(w, cmp, p_width, p_height);
  return 0;
}
