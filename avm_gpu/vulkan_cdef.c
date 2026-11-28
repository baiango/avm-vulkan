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

#include "avm_gpu/vulkan_cdef.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "avm_gpu/arena.h"
#include "avm_gpu/pipeline.h"

#ifndef AVM_GPU_CDEF_SPV_PATH
#define AVM_GPU_CDEF_SPV_PATH "cdef.spv"
#endif

#define CDEF_VERY_LARGE_VALUE 0x4000

struct avm_gpu_cdef {
  avm_gpu_arena *arena; /* device-local arena when AV2_GPU_ARENA=1 */
  uint32_t *in_host, *out_host; /* mapped pointer or host shadow per slot */
  size_t in_host_cap, out_host_cap;
  avm_gpu_pipeline *pipe;
  avm_gpu_buf in, out;
  VkDeviceSize in_cap, out_cap;
  uint32_t *expand;
  size_t expand_cap;
};

typedef struct push_constants {
  uint32_t stride, i0, j0, w, h, dir, pri_strength, sec_strength, pri_damping,
      sec_damping, coeff_shift, enable_pri, enable_sec, very_large, dstride,
      pad;
} push_constants;

int avm_gpu_cdef_create(avm_gpu_ctx *ctx, avm_gpu_cdef **out,
                        char err[AVM_GPU_ERR_LEN]) {
  *out = NULL;
  avm_gpu_cdef *cd = calloc(1, sizeof(*cd));
  if (!cd) {
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    return -1;
  }
  {
    if (avm_gpu_arena_wanted("cdef")) {
      const char *mb = getenv("AV2_GPU_ARENA_MB");
      const VkDeviceSize bytes =
          (VkDeviceSize)(mb && *mb ? atoi(mb) : 64) * 1024u * 1024u;
      char aerr[AVM_GPU_ERR_LEN];
      if (avm_gpu_arena_create(ctx, bytes, &cd->arena, aerr) != 0) {
        cd->arena = NULL;
      }
    }
  }
  if (avm_gpu_pipeline_create(ctx, AVM_GPU_CDEF_SPV_PATH, "AVM_GPU_CDEF_SPV",
                              2, sizeof(push_constants), &cd->pipe,
                              err) != 0) {
    free(cd);
    return -1;
  }
  *out = cd;
  return 0;
}

void avm_gpu_cdef_destroy(avm_gpu_ctx *ctx, avm_gpu_cdef *cd) {
  if (!cd) return;
  avm_gpu_pipeline_destroy(ctx, cd->pipe);
  /* the host pointers alias the mapped buffers on the host-visible path */
  if (cd->in_host != cd->in.mapped) free(cd->in_host);
  if (cd->out_host != cd->out.mapped) free(cd->out_host);
  avm_gpu_buf_destroy(ctx, &cd->in);
  avm_gpu_buf_destroy(ctx, &cd->out);
  avm_gpu_arena_destroy(ctx, cd->arena);
  free(cd->expand);
  free(cd);
}

int avm_gpu_cdef_filter16(avm_gpu_ctx *ctx, avm_gpu_cdef *cd,
                          const uint16_t *in, int in_rows, int stride, int i0,
                          int j0, int w, int h, int dir, int pri_strength,
                          int sec_strength, int pri_damping, int sec_damping,
                          int coeff_shift, int enable_pri, int enable_sec,
                          uint16_t *dst, int dstride,
                          char err[AVM_GPU_ERR_LEN]) {
  if (!in || !dst || stride <= 0 || in_rows <= 0 || w <= 0 || h <= 0) {
    snprintf(err, AVM_GPU_ERR_LEN, "invalid arguments");
    return -1;
  }
  if (dir < 0 || dir > 7) {
    snprintf(err, AVM_GPU_ERR_LEN, "direction out of range");
    return -1;
  }

  const size_t pixels = (size_t)stride * (size_t)in_rows;
  if (pixels > cd->expand_cap) {
    free(cd->expand);
    cd->expand = malloc(pixels * sizeof(*cd->expand));
    cd->expand_cap = cd->expand ? pixels : 0;
    if (!cd->expand) {
      snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
      return -1;
    }
  }
  for (size_t i = 0; i < pixels; i++) cd->expand[i] = in[i];

  const VkDeviceSize in_bytes = (VkDeviceSize)pixels * 4;
  const VkDeviceSize out_bytes = (VkDeviceSize)dstride * h * 4;
  if (avm_gpu_slot_reserve(ctx, cd->arena, &cd->in, &cd->in_cap, in_bytes,
                           err) != 0 ||
      avm_gpu_slot_reserve(ctx, cd->arena, &cd->out, &cd->out_cap, out_bytes,
                           err) != 0) {
    return -1;
  }
  if (cd->in.mapped) {
    cd->in_host = cd->in.mapped;
  } else if (avm_gpu_shadow_reserve(&cd->in_host, &cd->in_host_cap, in_bytes,
                                    err) != 0) {
    return -1;
  }
  if (cd->out.mapped) {
    cd->out_host = cd->out.mapped;
  } else if (avm_gpu_shadow_reserve(&cd->out_host, &cd->out_host_cap,
                                    out_bytes, err) != 0) {
    return -1;
  }
  memcpy(cd->in_host, cd->expand, pixels * 4);
  if (avm_gpu_shadow_push(ctx, cd->arena, &cd->in, cd->in_host, in_bytes,
                          err) != 0) {
    return -1;
  }

  const avm_gpu_buf bufs[2] = { cd->in, cd->out };
  push_constants pc = { (uint32_t)stride,
                        (uint32_t)i0,
                        (uint32_t)j0,
                        (uint32_t)w,
                        (uint32_t)h,
                        (uint32_t)dir,
                        (uint32_t)pri_strength,
                        (uint32_t)sec_strength,
                        (uint32_t)pri_damping,
                        (uint32_t)sec_damping,
                        (uint32_t)coeff_shift,
                        (uint32_t)(enable_pri ? 1 : 0),
                        (uint32_t)(enable_sec ? 1 : 0),
                        (uint32_t)CDEF_VERY_LARGE_VALUE,
                        (uint32_t)dstride,
                        getenv("AVM_GPU_CDEF_SUM")
                            ? (uint32_t)atoi(getenv("AVM_GPU_CDEF_SUM"))
                            : 0u };
  const uint32_t gx = (uint32_t)((w + 7) / 8);
  const uint32_t gy = (uint32_t)((h + 7) / 8);
  const int rc = avm_gpu_pipeline_dispatch_2d(ctx, cd->pipe, bufs, 2, &pc,
                                              sizeof(pc), gx, gy, err);
  if (rc != 0) return rc;

  if (avm_gpu_shadow_pull(ctx, cd->arena, &cd->out, cd->out_host, out_bytes,
                          err) != 0) {
    return -1;
  }
  const uint32_t *res = cd->out_host;
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      dst[(size_t)y * dstride + x] = (uint16_t)res[y * dstride + x];
    }
  }
  return 0;
}
