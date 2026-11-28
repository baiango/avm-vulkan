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

#include "avm_gpu/arena.h"
#include "avm_gpu/pipeline.h"

#ifndef AVM_GPU_DEBLOCK_SPV_PATH
#define AVM_GPU_DEBLOCK_SPV_PATH "deblock.spv"
#endif

struct avm_gpu_deblock {
  avm_gpu_arena *arena; /* device-local arena when AV2_GPU_ARENA=1 */
  uint32_t *host;       /* the slot's mapped pointer or its shadow */
  size_t host_cap;
  avm_gpu_pipeline *pipe;
  avm_gpu_buf img;
  VkDeviceSize img_cap;
  uint32_t *plane;
  size_t plane_cap;
};

typedef struct push_constants {
  uint32_t base, pitch, filt_neg, filt_pos, q_thresh, side_thresh, bd, ll_neg,
      ll_pos, count, step;
} push_constants;

/* Returns the host copy the kernel's buffer is filled from: the mapped pointer
 * for a host-visible buffer, or a shadow that is uploaded before the dispatch
 * and downloaded after it. */
static int db_reserve(avm_gpu_ctx *ctx, avm_gpu_deblock *db,
                      VkDeviceSize bytes, char err[AVM_GPU_ERR_LEN]) {
  if (avm_gpu_slot_reserve(ctx, db->arena, &db->img, &db->img_cap, bytes,
                           err) != 0) {
    return -1;
  }
  if (db->img.mapped) {
    db->host = db->img.mapped;
    return 0;
  }
  if ((size_t)bytes > db->host_cap) {
    free(db->host);
    db->host = malloc((size_t)bytes);
    db->host_cap = db->host ? (size_t)bytes : 0;
    if (!db->host) {
      snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
      return -1;
    }
  }
  return 0;
}

static int db_push(avm_gpu_ctx *ctx, avm_gpu_deblock *db, VkDeviceSize bytes,
                   char err[AVM_GPU_ERR_LEN]) {
  if (db->img.mapped) return 0;
  return avm_gpu_slot_upload(ctx, db->arena, &db->img, db->host, bytes, err);
}

static int db_pull(avm_gpu_ctx *ctx, avm_gpu_deblock *db, VkDeviceSize bytes,
                   char err[AVM_GPU_ERR_LEN]) {
  if (db->img.mapped) return 0;
  return avm_gpu_slot_download(ctx, db->arena, &db->img, db->host, bytes, err);
}

int avm_gpu_deblock_create(avm_gpu_ctx *ctx, avm_gpu_deblock **out,
                           char err[AVM_GPU_ERR_LEN]) {
  *out = NULL;
  avm_gpu_deblock *db = calloc(1, sizeof(*db));
  if (!db) {
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    return -1;
  }
  {
    if (avm_gpu_arena_wanted("deblock")) {
      const char *mb = getenv("AV2_GPU_ARENA_MB");
      const VkDeviceSize bytes =
          (VkDeviceSize)(mb && *mb ? atoi(mb) : 64) * 1024u * 1024u;
      char aerr[AVM_GPU_ERR_LEN];
      if (avm_gpu_arena_create(ctx, bytes, &db->arena, aerr) != 0) {
        db->arena = NULL;
      }
    }
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
  /* host aliases img.mapped on the host-visible path, so it must not be freed
   * after the buffer took it down. */
  if (db->host != db->img.mapped) free(db->host);
  avm_gpu_buf_destroy(ctx, &db->img);
  avm_gpu_arena_destroy(ctx, db->arena);
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

  if (db_reserve(ctx, db, (VkDeviceSize)pixels * 4, err) != 0) return -1;
  memcpy(db->host, db->plane, pixels * 4);
  if (db_push(ctx, db, (VkDeviceSize)pixels * 4, err) != 0) return -1;

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
                        4,
                        (uint32_t)(vertical ? stride : 1) };
  const int rc =
      avm_gpu_pipeline_dispatch(ctx, db->pipe, bufs, 1, &pc, sizeof(pc), 1, err);
  if (rc != 0) return rc;

  if (db_pull(ctx, db, (VkDeviceSize)pixels * 4, err) != 0) return -1;
  const uint32_t *res = db->host;
  for (size_t i = 0; i < pixels; i++) out[i] = (uint16_t)res[i];
  return 0;
}

int avm_gpu_deblock_window(avm_gpu_ctx *ctx, avm_gpu_deblock *db,
                           const uint16_t *p, int across_pitch, int step,
                           int filt_neg, int filt_pos, int q_thresh,
                           int side_thresh, int bd, int ll_neg, int ll_pos,
                           uint16_t *out, char err[AVM_GPU_ERR_LEN]) {
  if (!p || !out || across_pitch <= 0 || step <= 0) {
    snprintf(err, AVM_GPU_ERR_LEN, "invalid arguments");
    return -1;
  }
  /* The filter reads filt/2+2 samples across the edge; the four samples along
   * it are the whole extent in that direction. */
  /* The positive side can reach the full filter length (MAX_DBL_FLT_LEN = 8)
   * and the negative side at most filt_neg; the delta itself reads 2 further
   * out. A smaller window let the kernel write past its buffer. */
  const int wider = filt_neg > 8 ? filt_neg : 8;
  const int reach = wider + 2;
  const int count = 4;
  const int wspan = 2 * reach + 1;
  const int hspan = count;

  if (db->plane_cap < (size_t)wspan * hspan) {
    free(db->plane);
    db->plane = malloc((size_t)wspan * hspan * sizeof(*db->plane));
    db->plane_cap = db->plane ? (size_t)wspan * hspan : 0;
    if (!db->plane) {
      snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
      return -1;
    }
  }
  uint32_t *win = db->plane;
  for (int j = 0; j < hspan; j++) {
    for (int i = 0; i < wspan; i++) {
      win[(size_t)j * wspan + i] = p[j * step + (i - reach) * across_pitch];
    }
  }

  if (db_reserve(ctx, db, (VkDeviceSize)wspan * hspan * 4, err) != 0) {
    return -1;
  }
  uint32_t *img = db->host;
  for (int i = 0; i < wspan * hspan; i++) img[i] = win[i];
  if (db_push(ctx, db, (VkDeviceSize)wspan * hspan * 4, err) != 0) return -1;

  const avm_gpu_buf bufs[1] = { db->img };
  /* The window is laid out so the four samples run along the rows: delta
   * across, step = the window stride. */
  push_constants pc = { (uint32_t)reach, 1u, (uint32_t)filt_neg,
                        (uint32_t)filt_pos, (uint32_t)q_thresh,
                        (uint32_t)side_thresh, (uint32_t)bd,
                        (uint32_t)(ll_neg ? 1 : 0), (uint32_t)(ll_pos ? 1 : 0),
                        (uint32_t)count, (uint32_t)wspan };
  const int rc =
      avm_gpu_pipeline_dispatch(ctx, db->pipe, bufs, 1, &pc, sizeof(pc), 1, err);
  if (rc != 0) return rc;

  if (db_pull(ctx, db, (VkDeviceSize)wspan * hspan * 4, err) != 0) return -1;
  for (int j = 0; j < hspan; j++) {
    for (int i = 0; i < wspan; i++) {
      out[j * step + (i - reach) * across_pitch] =
          (uint16_t)img[(size_t)j * wspan + i];
    }
  }
  return 0;
}
