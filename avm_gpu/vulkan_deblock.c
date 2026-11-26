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

#include "avm_gpu/vulkan_deblock.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "avm_gpu/pipeline.h"

#ifndef AVM_GPU_DEBLOCK_SPV_PATH
#define AVM_GPU_DEBLOCK_SPV_PATH "deblock.spv"
#endif

struct avm_gpu_deblock {
  avm_gpu_pipeline *pipe;
  avm_gpu_buf img;
  VkDeviceSize img_cap;
  uint32_t *plane;
  size_t plane_cap;
};

typedef struct push_constants {
  uint32_t base, pitch, filt_neg, filt_pos, q_thresh, side_thresh, bd, ll_neg,
      ll_pos, count;
} push_constants;

int avm_gpu_deblock_create(avm_gpu_ctx *ctx, avm_gpu_deblock **out,
                           char err[AVM_GPU_ERR_LEN]) {
  *out = NULL;
  avm_gpu_deblock *db = calloc(1, sizeof(*db));
  if (!db) {
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    return -1;
  }
  if (avm_gpu_pipeline_create(ctx, AVM_GPU_DEBLOCK_SPV_PATH,
                              "AVM_GPU_DEBLOCK_SPV", 1, sizeof(push_constants),
                              &db->pipe, err) != 0) {
    free(db);
    return -1;
  }
  *out = db;
  return 0;
}

void avm_gpu_deblock_destroy(avm_gpu_ctx *ctx, avm_gpu_deblock *db) {
  if (!db) return;
  avm_gpu_pipeline_destroy(ctx, db->pipe);
  avm_gpu_buf_destroy(ctx, &db->img);
  free(db->plane);
  free(db);
}

int avm_gpu_deblock_generic(avm_gpu_ctx *ctx, avm_gpu_deblock *db,
                            const uint16_t *plane, int stride, int rows, int x,
                            int y, int filt_neg, int filt_pos, int q_thresh,
                            int side_thresh, int bd, int ll_neg, int ll_pos,
                            int vertical, uint16_t *out,
                            char err[AVM_GPU_ERR_LEN]) {
  if (!plane || !out || stride <= 0 || rows <= 0) {
    snprintf(err, AVM_GPU_ERR_LEN, "invalid arguments");
    return -1;
  }

  const size_t pixels = (size_t)stride * (size_t)rows;
  if (pixels > db->plane_cap) {
    free(db->plane);
    db->plane = malloc(pixels * sizeof(*db->plane));
    db->plane_cap = db->plane ? pixels : 0;
    if (!db->plane) {
      snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
      return -1;
    }
  }
  for (size_t i = 0; i < pixels; i++) db->plane[i] = plane[i];

  if (db->img_cap < (VkDeviceSize)pixels * 4) {
    avm_gpu_buf_destroy(ctx, &db->img);
    if (avm_gpu_buf_create(ctx, (VkDeviceSize)pixels * 4, &db->img, err) != 0) {
      return -1;
    }
    db->img_cap = db->img.size;
  }
  memcpy(db->img.mapped, db->plane, pixels * 4);

  const avm_gpu_buf bufs[1] = { db->img };
  push_constants pc = { (uint32_t)((size_t)y * stride + x),
                        vertical ? 1u : (uint32_t)stride,
                        (uint32_t)filt_neg,
                        (uint32_t)filt_pos,
                        (uint32_t)q_thresh,
                        (uint32_t)side_thresh,
                        (uint32_t)bd,
                        (uint32_t)(ll_neg ? 1 : 0),
                        (uint32_t)(ll_pos ? 1 : 0),
                        4 };
  const int rc =
      avm_gpu_pipeline_dispatch(ctx, db->pipe, bufs, 1, &pc, sizeof(pc), 1, err);
  if (rc != 0) return rc;

  const uint32_t *res = db->img.mapped;
  for (size_t i = 0; i < pixels; i++) out[i] = (uint16_t)res[i];
  return 0;
}
