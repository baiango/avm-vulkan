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

#include "avm_gpu/vulkan_inter.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "avm_gpu/arena.h"
#include "avm_gpu/pipeline.h"

#ifndef AVM_GPU_INTER_SPV_PATH
#define AVM_GPU_INTER_SPV_PATH "inter.spv"
#endif
#ifndef AVM_GPU_COMPOUND_SPV_PATH
#define AVM_GPU_COMPOUND_SPV_PATH "compound.spv"
#endif
#ifndef AVM_GPU_CWP_SPV_PATH
#define AVM_GPU_CWP_SPV_PATH "cwp.spv"
#endif
#ifndef AVM_GPU_CWPCONV_SPV_PATH
#define AVM_GPU_CWPCONV_SPV_PATH "cwpconv.spv"
#endif

#define INTER_FILTER_ENTRIES (16 * 8)

typedef struct cwp_constants {
  uint32_t src_stride, conv_stride, dst_stride, w, h, bd, x0, y0, sx, sy;
  uint32_t round0, round1, do_average, uneven, fwd, bck, conv_index, mode;
  uint32_t sb, x_step, y_step, packed;
} cwp_constants;

#define INTER_SHADOW_MAX 16

/* Host-side mirror of a slot. With host-visible buffers the mirror *is* the
 * mapped pointer, so nothing changes; an arena slot gets a shadow that the
 * existing fill/read loops keep writing, uploaded before and downloaded after
 * the dispatch. */
typedef struct inter_shadow {
  avm_gpu_buf *buf;
  uint32_t *host;
  size_t bytes;
} inter_shadow;

struct avm_gpu_inter {
  avm_gpu_arena *arena; /* device-local arena when AV2_GPU_ARENA=1 */
  inter_shadow shadow[INTER_SHADOW_MAX];
  avm_gpu_pipeline *pipe;
  avm_gpu_pipeline *blend_pipe;
  avm_gpu_pipeline *cwp_pipe;
  avm_gpu_pipeline *cwpconv_pipe;
  avm_gpu_buf cwpconv_conv, cwpconv_dst;
  VkDeviceSize cwpconv_conv_cap, cwpconv_dst_cap;
  avm_gpu_buf cpred, cref, cout;
  VkDeviceSize cpred_cap, cref_cap, cout_cap;
  avm_gpu_buf s0, s1, mask, bdst;
  VkDeviceSize s0_cap, s1_cap, mask_cap, bdst_cap;
  uint32_t *scratch;
  size_t scratch_cap;
  avm_gpu_buf src, filt, dst, conv;
  VkDeviceSize src_cap, filt_cap, dst_cap, conv_cap;
  uint32_t *expand;
  size_t expand_cap;
};

typedef struct push_constants {
  uint32_t src_stride, dst_stride, sx, sy, w, h, bd, x0, y0, x_step, y_step,
      sr, round0, round1, fvo, sb;
  uint32_t compound, do_average, conv_stride, uneven;
  int32_t fwd, bck;
  uint32_t packed;
} push_constants;

/* Reference planes are uploaded as two little-endian 16-bit samples per word
 * whenever the row stride is even (which maps every row onto whole words),
 * halving the bytes; an odd stride keeps the one-sample-per-word layout. */
static int pack16_for(int stride) { return (stride % 2) == 0; }

static int ensure_buf(avm_gpu_ctx *ctx, avm_gpu_inter *in, avm_gpu_buf *buf,
                      VkDeviceSize *cap, VkDeviceSize need,
                      char err[AVM_GPU_ERR_LEN]) {
  return avm_gpu_slot_reserve(ctx, in->arena, buf, cap, need, err);
}

/* Host pointer the fill loops write to: the mapped buffer, or the shadow. */
static uint32_t *inter_host(avm_gpu_ctx *ctx, avm_gpu_inter *in,
                            avm_gpu_buf *buf, size_t bytes,
                            char err[AVM_GPU_ERR_LEN]) {
  (void)ctx;
  if (buf->mapped) return buf->mapped;
  inter_shadow *slot = NULL;
  for (int i = 0; i < INTER_SHADOW_MAX; i++) {
    if (in->shadow[i].buf == buf) {
      slot = &in->shadow[i];
      break;
    }
    if (!in->shadow[i].buf && !slot) slot = &in->shadow[i];
  }
  if (!slot) {
    snprintf(err, AVM_GPU_ERR_LEN, "shadow table full");
    return NULL;
  }
  if (slot->buf == buf && slot->bytes >= bytes) return slot->host;
  free(slot->host);
  slot->host = malloc(bytes ? bytes : 4);
  /* A fresh host-visible VkBuffer reads as zero and the kernels rely on the
   * same starting state (the blend reads the destination it is about to
   * write), so a new shadow starts zeroed too. */
  if (slot->host) memset(slot->host, 0, bytes ? bytes : 4);
  slot->buf = slot->host ? buf : NULL;
  slot->bytes = slot->host ? bytes : 0;
  if (!slot->host) {
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    return NULL;
  }
  return slot->host;
}

static int inter_push(avm_gpu_ctx *ctx, avm_gpu_inter *in, avm_gpu_buf *buf,
                      size_t bytes, char err[AVM_GPU_ERR_LEN]) {
  if (buf->mapped) return 0;
  uint32_t *host = inter_host(ctx, in, buf, bytes, err);
  if (!host) return -1;
  return avm_gpu_slot_upload(ctx, in->arena, buf, host, bytes, err);
}

static int inter_pull(avm_gpu_ctx *ctx, avm_gpu_inter *in, avm_gpu_buf *buf,
                      size_t bytes, char err[AVM_GPU_ERR_LEN]) {
  if (buf->mapped) return 0;
  uint32_t *host = inter_host(ctx, in, buf, bytes, err);
  if (!host) return -1;
  return avm_gpu_slot_download(ctx, in->arena, buf, host, bytes, err);
}

int avm_gpu_inter_create(avm_gpu_ctx *ctx, avm_gpu_inter **out,
                         char err[AVM_GPU_ERR_LEN]) {
  *out = NULL;
  avm_gpu_inter *in = calloc(1, sizeof(*in));
  if (!in) {
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    return -1;
  }
  const char *arena_env = getenv("AV2_GPU_ARENA");
  if (arena_env && *arena_env && strcmp(arena_env, "0")) {
    const char *mb = getenv("AV2_GPU_ARENA_MB");
    const VkDeviceSize arena_bytes =
        (VkDeviceSize)(mb && *mb ? atoi(mb) : 64) * 1024u * 1024u;
    char aerr[AVM_GPU_ERR_LEN];
    in->arena = NULL;
    if (avm_gpu_arena_create(ctx, arena_bytes, &in->arena, aerr) != 0) {
      if (getenv("AV2_GPU_ARENA_VERBOSE")) {
        fprintf(stderr, "avm_gpu: inter arena unavailable (%s)\n", aerr);
      }
      in->arena = NULL;
    }
  }
  if (avm_gpu_pipeline_create(ctx, AVM_GPU_CWPCONV_SPV_PATH,
                              "AVM_GPU_CWPCONV_SPV", 4, sizeof(cwp_constants),
                              &in->cwpconv_pipe, err) != 0) {
    free(in);
    return -1;
  }
  if (avm_gpu_pipeline_create(ctx, AVM_GPU_INTER_SPV_PATH, "AVM_GPU_INTER_SPV",
                              4, sizeof(push_constants), &in->pipe,
                              err) != 0) {
    free(in);
    return -1;
  }
  if (avm_gpu_pipeline_create(ctx, AVM_GPU_COMPOUND_SPV_PATH,
                              "AVM_GPU_COMPOUND_SPV", 4, 12 * 4,
                              &in->blend_pipe, err) != 0) {
    avm_gpu_inter_destroy(ctx, in);
    return -1;
  }
  if (avm_gpu_pipeline_create(ctx, AVM_GPU_CWP_SPV_PATH, "AVM_GPU_CWP_SPV", 3,
                              8 * 4, &in->cwp_pipe, err) != 0) {
    avm_gpu_inter_destroy(ctx, in);
    return -1;
  }
  *out = in;
  return 0;
}

void avm_gpu_inter_destroy(avm_gpu_ctx *ctx, avm_gpu_inter *in) {
  if (!in) return;
  avm_gpu_pipeline_destroy(ctx, in->pipe);
  avm_gpu_pipeline_destroy(ctx, in->blend_pipe);
  avm_gpu_pipeline_destroy(ctx, in->cwp_pipe);
  avm_gpu_pipeline_destroy(ctx, in->cwpconv_pipe);
  avm_gpu_buf_destroy(ctx, &in->cwpconv_conv);
  avm_gpu_buf_destroy(ctx, &in->cwpconv_dst);
  avm_gpu_buf_destroy(ctx, &in->src);
  avm_gpu_buf_destroy(ctx, &in->filt);
  avm_gpu_buf_destroy(ctx, &in->dst);
  avm_gpu_buf_destroy(ctx, &in->conv);
  avm_gpu_buf_destroy(ctx, &in->s0);
  avm_gpu_buf_destroy(ctx, &in->s1);
  avm_gpu_buf_destroy(ctx, &in->mask);
  avm_gpu_buf_destroy(ctx, &in->bdst);
  avm_gpu_arena_destroy(ctx, in->arena);
  for (int i = 0; i < INTER_SHADOW_MAX; i++) free(in->shadow[i].host);
  avm_gpu_buf_destroy(ctx, &in->cpred);
  avm_gpu_buf_destroy(ctx, &in->cref);
  avm_gpu_buf_destroy(ctx, &in->cout);
  free(in->scratch);
  free(in->expand);
  free(in);
}

static int convolve_impl(avm_gpu_ctx *ctx, avm_gpu_inter *in,
                         const uint16_t *src, int src_stride, int src_rows,
                         int sx, int sy, const int16_t *filter, int x0_q4,
                         int y0_q4, int x_step_q4, int y_step_q4, int w, int h,
                         int bd, uint16_t *dst, int dst_stride, int sr,
                         int round0, int round1, const int16_t *filter_y,
                         int subpel_bits, char err[AVM_GPU_ERR_LEN],
                         const uint16_t *conv, int conv_stride, int do_average,
                         int uneven, int fwd, int bck) {
  if (!src || !filter || !dst) {
    snprintf(err, AVM_GPU_ERR_LEN, "invalid arguments");
    return -1;
  }
  if (w <= 0 || h <= 0 || w > 64 || h > 64) {
    snprintf(err, AVM_GPU_ERR_LEN, "unsupported block size %dx%d", w, h);
    return -1;
  }
  const int smask = (1 << subpel_bits) - 1;
  if (x0_q4 < 0 || x0_q4 > smask || y0_q4 < 0 || y0_q4 > smask) {
    snprintf(err, AVM_GPU_ERR_LEN, "subpel position out of range");
    return -1;
  }
  if (x_step_q4 < 1 || y_step_q4 < 1) {
    snprintf(err, AVM_GPU_ERR_LEN, "step out of range");
    return -1;
  }
  /* base already contains the pixel offset: the last output pixel starts at
   * ((w - 1) * step + phase) >> 4 and its taps reach +4 past that, so the
   * border grows with the scale factor. */
  const int last_bx = ((w - 1) * x_step_q4 + x0_q4) >> subpel_bits;
  const int last_by = ((h - 1) * y_step_q4 + y0_q4) >> subpel_bits;
  if (sx < 3 || sy < 3 || sx + last_bx + 4 > src_stride ||
      sy + last_by + 4 > src_rows) {
    snprintf(err, AVM_GPU_ERR_LEN,
             "block overlaps the filter border (sx %d sy %d w %d h %d steps "
             "%d/%d stride %d rows %d)",
             sx, sy, w, h, x_step_q4, y_step_q4, src_stride, src_rows);
    return -1;
  }

  const size_t pixels = (size_t)src_stride * (size_t)src_rows;
  const int packed = pack16_for(src_stride);
  if (!packed && pixels > in->expand_cap) {
    free(in->expand);
    in->expand = malloc(pixels * sizeof(*in->expand));
    in->expand_cap = in->expand ? pixels : 0;
    if (!in->expand) {
      snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
      return -1;
    }
  }
  if (!packed) {
    for (size_t i = 0; i < pixels; i++) in->expand[i] = src[i];
  }

  const int filt_entries = filter_y ? 2 * INTER_FILTER_ENTRIES : INTER_FILTER_ENTRIES;
  /* The shader always declares the conv binding, so a single-reference call
   * binds a dummy. It has to be block-sized: with a one-entry (4-byte) range
   * the sr path came back with garbage on MoltenVK (zeros and clips in some
   * columns), while 64x64 works for every sr and scaled case. */
  const size_t conv_n = conv ? (size_t)conv_stride * (size_t)h : 64 * 64;
  if (ensure_buf(ctx, in, &in->src, &in->src_cap,
                 (VkDeviceSize)pixels * (packed ? 2 : 4), err) ||
      ensure_buf(ctx, in, &in->filt, &in->filt_cap,
                 (VkDeviceSize)filt_entries * 4, err) ||
      ensure_buf(ctx, in, &in->dst, &in->dst_cap,
                 (VkDeviceSize)dst_stride * h * 4, err) ||
      ensure_buf(ctx, in, &in->conv, &in->conv_cap, (VkDeviceSize)conv_n * 4, err)) {
    return -1;
  }
  const size_t src_bytes = (size_t)pixels * (packed ? 2 : 4);
  const size_t filt_bytes = (size_t)filt_entries * 4;
  const size_t dst_bytes = (size_t)dst_stride * (size_t)h * 4;
  const size_t conv_bytes = conv_n * 4;
  uint8_t *src_host =
      (uint8_t *)inter_host(ctx, in, &in->src, src_bytes, err);
  uint32_t *fp = inter_host(ctx, in, &in->filt, filt_bytes, err);
  uint32_t *dst_host = inter_host(ctx, in, &in->dst, dst_bytes, err);
  uint32_t *conv_host = inter_host(ctx, in, &in->conv, conv_bytes, err);
  if (!src_host || !fp || !dst_host || !conv_host) return -1;
  if (packed) {
    memcpy(src_host, src, (size_t)pixels * 2);
  } else {
    memcpy(src_host, in->expand, (size_t)pixels * 4);
  }
  for (int i = 0; i < INTER_FILTER_ENTRIES; i++) fp[i] = (uint32_t)filter[i];
  if (filter_y) {
    for (int i = 0; i < INTER_FILTER_ENTRIES; i++) {
      fp[INTER_FILTER_ENTRIES + i] = (uint32_t)filter_y[i];
    }
  }
  memset(dst_host, 0, dst_bytes);
  if (conv) {
    for (size_t i = 0; i < conv_n; i++) conv_host[i] = conv[i];
  }
  if (inter_push(ctx, in, &in->src, src_bytes, err) != 0 ||
      inter_push(ctx, in, &in->filt, filt_bytes, err) != 0 ||
      inter_push(ctx, in, &in->dst, dst_bytes, err) != 0 ||
      inter_push(ctx, in, &in->conv, conv_bytes, err) != 0) {
    return -1;
  }

  const avm_gpu_buf bufs[4] = { in->src, in->filt, in->dst, in->conv };
  push_constants pc = { (uint32_t)src_stride,
                        (uint32_t)dst_stride,
                        (uint32_t)sx,
                        (uint32_t)sy,
                        (uint32_t)w,
                        (uint32_t)h,
                        (uint32_t)bd,
                        (uint32_t)x0_q4,
                        (uint32_t)y0_q4,
                        (uint32_t)x_step_q4,
                        (uint32_t)y_step_q4,
                        (uint32_t)sr,
                        (uint32_t)round0,
                        (uint32_t)round1,
                        (uint32_t)(filter_y ? INTER_FILTER_ENTRIES : 0),
                        (uint32_t)subpel_bits,
                        (uint32_t)(conv ? 1 : 0),
                        (uint32_t)(do_average ? 1 : 0),
                        (uint32_t)conv_stride,
                        (uint32_t)(uneven ? 1 : 0),
                        fwd,
                        bck,
                        (uint32_t)packed };
  const uint32_t gx = (uint32_t)((w + 7) / 8);
  const uint32_t gy = (uint32_t)((h + 7) / 8);
  const int rc = avm_gpu_pipeline_dispatch_2d(ctx, in->pipe, bufs, 4, &pc,
                                              sizeof(pc), gx, gy, err);
  if (rc != 0) return rc;

  if (inter_pull(ctx, in, &in->dst, dst_bytes, err) != 0) return -1;
  if (conv && !do_average &&
      inter_pull(ctx, in, &in->conv, conv_bytes, err) != 0) {
    return -1;
  }
  const uint32_t *res = dst_host;
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      dst[(size_t)y * dst_stride + x] = (uint16_t)res[y * dst_stride + x];
    }
  }
  if (conv && !do_average) {
    /* The first reference's output is the high-precision buffer. */
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        ((uint16_t *)conv)[(size_t)y * conv_stride + x] =
            (uint16_t)conv_host[(size_t)y * conv_stride + x];
      }
    }
  }
  return 0;
}

int avm_gpu_inter_convolve(avm_gpu_ctx *ctx, avm_gpu_inter *in,
                           const uint16_t *src, int src_stride, int src_rows,
                           int sx, int sy, const int16_t *filter, int x0_q4,
                           int y0_q4, int w, int h, int bd, uint16_t *dst,
                           int dst_stride, char err[AVM_GPU_ERR_LEN]) {
  return avm_gpu_inter_convolve_scaled(ctx, in, src, src_stride, src_rows, sx,
                                       sy, filter, x0_q4, y0_q4, 16, 16, w, h,
                                       bd, dst, dst_stride, err);
}

typedef struct blend_pc {
  uint32_t s0_stride, s1_stride, dst_stride, mask_stride, w, h, subw, subh, bd,
      mode;
  int32_t round_offset;
  uint32_t round_bits;
} blend_pc;

int avm_gpu_inter_blend_a64(avm_gpu_ctx *ctx, avm_gpu_inter *in,
                            const uint16_t *src0, int s0_stride,
                            const uint16_t *src1, int s1_stride,
                            const uint8_t *mask, int mask_stride, int w, int h,
                            int subw, int subh, int bd, uint16_t *dst,
                            int dst_stride, char err[AVM_GPU_ERR_LEN]) {
  if (!src0 || !src1 || !mask || !dst) {
    snprintf(err, AVM_GPU_ERR_LEN, "invalid arguments");
    return -1;
  }
  if (w <= 0 || h <= 0 || w > 64 || h > 64) {
    snprintf(err, AVM_GPU_ERR_LEN, "unsupported block size %dx%d", w, h);
    return -1;
  }
  /* A subsampled mask is a full-resolution mask that the kernel boxes down
   * (2x2, 1x2 or 2x1 samples per output sample, as avm_highbd_blend_a64_mask_c
   * does), so its stride covers 2x the block width. */
  const int mw = subw ? 2 * w : w;
  const int mh = subh ? 2 * h : h;
  if (mask_stride < mw) {
    snprintf(err, AVM_GPU_ERR_LEN, "mask stride too small");
    return -1;
  }

  const size_t s0_n = (size_t)s0_stride * h;
  const size_t s1_n = (size_t)s1_stride * h;
  const size_t mk_n = (size_t)mask_stride * mh;
  const size_t dst_n = (size_t)dst_stride * h;

  if (ensure_buf(ctx, in, &in->s0, &in->s0_cap, (VkDeviceSize)s0_n * 4, err) ||
      ensure_buf(ctx, in, &in->s1, &in->s1_cap, (VkDeviceSize)s1_n * 4, err) ||
      ensure_buf(ctx, in, &in->mask, &in->mask_cap, (VkDeviceSize)mk_n * 4, err) ||
      ensure_buf(ctx, in, &in->bdst, &in->bdst_cap, (VkDeviceSize)dst_n * 4, err)) {
    return -1;
  }

  const size_t s0_bytes = s0_n * 4, s1_bytes = s1_n * 4;
  const size_t mk_bytes = mk_n * 4, bdst_bytes = dst_n * 4;
  uint32_t *p0 = inter_host(ctx, in, &in->s0, s0_bytes, err);
  uint32_t *p1 = inter_host(ctx, in, &in->s1, s1_bytes, err);
  uint32_t *pm = inter_host(ctx, in, &in->mask, mk_bytes, err);
  uint32_t *bdst_host = inter_host(ctx, in, &in->bdst, bdst_bytes, err);
  if (!p0 || !p1 || !pm || !bdst_host) return -1;
  for (size_t i = 0; i < s0_n; i++) p0[i] = src0[i];
  for (size_t i = 0; i < s1_n; i++) p1[i] = src1[i];
  for (size_t i = 0; i < mk_n; i++) pm[i] = mask[i];
  if (inter_push(ctx, in, &in->s0, s0_bytes, err) != 0 ||
      inter_push(ctx, in, &in->s1, s1_bytes, err) != 0 ||
      inter_push(ctx, in, &in->mask, mk_bytes, err) != 0 ||
      inter_push(ctx, in, &in->bdst, bdst_bytes, err) != 0) {
    return -1;
  }

  const avm_gpu_buf bufs[4] = { in->s0, in->s1, in->mask, in->bdst };
  blend_pc pc = { (uint32_t)s0_stride, (uint32_t)s1_stride,
                  (uint32_t)dst_stride, (uint32_t)mask_stride,
                  (uint32_t)w,         (uint32_t)h,
                  (uint32_t)(subw ? 1 : 0), (uint32_t)(subh ? 1 : 0),
                  (uint32_t)bd,        0, 0, 0 };
  const uint32_t gx = (uint32_t)((w + 7) / 8);
  const uint32_t gy = (uint32_t)((h + 7) / 8);
  const int rc = avm_gpu_pipeline_dispatch_2d(ctx, in->blend_pipe, bufs, 4, &pc,
                                              sizeof(pc), gx, gy, err);
  if (rc != 0) return rc;

  if (inter_pull(ctx, in, &in->bdst, bdst_bytes, err) != 0) return -1;
  const uint32_t *res = bdst_host;
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      dst[(size_t)y * dst_stride + x] = (uint16_t)res[y * dst_stride + x];
    }
  }
  return 0;
}

typedef struct cwp_pc {
  uint32_t pred_stride, ref_stride, out_stride, w, h;
  int32_t fwd_offset, bck_offset;
  uint32_t pad;
} cwp_pc;

int avm_gpu_inter_cwp(avm_gpu_ctx *ctx, avm_gpu_inter *in, const uint16_t *pred,
                      int pred_stride, const uint16_t *ref, int ref_stride,
                      int w, int h, int fwd_offset, int bck_offset,
                      uint16_t *out, int out_stride,
                      char err[AVM_GPU_ERR_LEN]) {
  if (!pred || !ref || !out) {
    snprintf(err, AVM_GPU_ERR_LEN, "invalid arguments");
    return -1;
  }
  if (w <= 0 || h <= 0 || w > 64 || h > 64) {
    snprintf(err, AVM_GPU_ERR_LEN, "unsupported block size %dx%d", w, h);
    return -1;
  }
  const size_t pn = (size_t)pred_stride * h;
  const size_t rn = (size_t)ref_stride * h;
  const size_t on = (size_t)out_stride * h;
  if (ensure_buf(ctx, in, &in->cpred, &in->cpred_cap, (VkDeviceSize)pn * 4, err) ||
      ensure_buf(ctx, in, &in->cref, &in->cref_cap, (VkDeviceSize)rn * 4, err) ||
      ensure_buf(ctx, in, &in->cout, &in->cout_cap, (VkDeviceSize)on * 4, err)) {
    return -1;
  }
  const size_t pn_bytes = pn * 4, rn_bytes = rn * 4, on_bytes = on * 4;
  uint32_t *pp = inter_host(ctx, in, &in->cpred, pn_bytes, err);
  uint32_t *rp = inter_host(ctx, in, &in->cref, rn_bytes, err);
  uint32_t *cout_host = inter_host(ctx, in, &in->cout, on_bytes, err);
  if (!pp || !rp || !cout_host) return -1;
  for (size_t i = 0; i < pn; i++) pp[i] = pred[i];
  for (size_t i = 0; i < rn; i++) rp[i] = ref[i];
  if (inter_push(ctx, in, &in->cpred, pn_bytes, err) != 0 ||
      inter_push(ctx, in, &in->cref, rn_bytes, err) != 0 ||
      inter_push(ctx, in, &in->cout, on_bytes, err) != 0) {
    return -1;
  }

  const avm_gpu_buf bufs[3] = { in->cpred, in->cref, in->cout };
  cwp_pc pc = { (uint32_t)pred_stride, (uint32_t)ref_stride,
                (uint32_t)out_stride,  (uint32_t)w,
                (uint32_t)h,           (int32_t)fwd_offset,
                (int32_t)bck_offset,   0 };
  const uint32_t gx = (uint32_t)((w + 7) / 8);
  const uint32_t gy = (uint32_t)((h + 7) / 8);
  const int rc = avm_gpu_pipeline_dispatch_2d(ctx, in->cwp_pipe, bufs, 3, &pc,
                                              sizeof(pc), gx, gy, err);
  if (rc != 0) return rc;
  if (inter_pull(ctx, in, &in->cout, on_bytes, err) != 0) return -1;
  const uint32_t *res = cout_host;
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      out[(size_t)y * out_stride + x] = (uint16_t)res[y * out_stride + x];
    }
  }
  return 0;
}

int avm_gpu_inter_convolve_scaled(avm_gpu_ctx *ctx, avm_gpu_inter *in,
                                  const uint16_t *src, int src_stride,
                                  int src_rows, int sx, int sy,
                                  const int16_t *filter, int x0_q4,
                                  int y0_q4, int x_step_q4, int y_step_q4,
                                  int w, int h, int bd, uint16_t *dst,
                                  int dst_stride, char err[AVM_GPU_ERR_LEN]) {
  return convolve_impl(ctx, in, src, src_stride, src_rows, sx, sy, filter, x0_q4,
                       y0_q4, x_step_q4, y_step_q4, w, h, bd, dst, dst_stride, 0,
                       0, 0, NULL, 4, err, NULL, 0, 0, 0, 0, 0);
}

int avm_gpu_inter_convolve_sr(avm_gpu_ctx *ctx, avm_gpu_inter *in,
                              const uint16_t *src, int src_stride, int src_rows,
                              int sx, int sy, const int16_t *filter_x,
                              const int16_t *filter_y, int subpel_x_qn,
                              int subpel_y_qn, int w, int h, int bd, int round0,
                              int round1, uint16_t *dst, int dst_stride,
                              char err[AVM_GPU_ERR_LEN]) {
  /* av2_highbd_convolve_2d_sr_c() for an unscaled reference: steps are 16 and
   * the intermediate is 16-bit without the inter-pass clip. */
  return convolve_impl(ctx, in, src, src_stride, src_rows, sx, sy, filter_x,
                       subpel_x_qn, subpel_y_qn, 16, 16, w, h, bd, dst,
                       dst_stride, 1, round0, round1, filter_y, 4, err, NULL, 0,
                       0, 0, 0, 0);
}

int avm_gpu_inter_convolve_scaled_sr(
    avm_gpu_ctx *ctx, avm_gpu_inter *in, const uint16_t *src, int src_stride,
    int src_rows, int sx, int sy, const int16_t *filter_x,
    const int16_t *filter_y, int subpel_x_qn, int x_step_qn, int subpel_y_qn,
    int y_step_qn, int w, int h, int bd, int round0, int round1, uint16_t *dst,
    int dst_stride, char err[AVM_GPU_ERR_LEN]) {
  /* av2_highbd_convolve_2d_scale_c() for one reference: subpel positions and
   * steps in 1/1024 units (SCALE_SUBPEL_BITS), same two-pass rounding as the
   * unscaled sr path. */
  return convolve_impl(ctx, in, src, src_stride, src_rows, sx, sy, filter_x,
                       subpel_x_qn, subpel_y_qn, x_step_qn, y_step_qn, w, h, bd,
                       dst, dst_stride, 1, round0, round1, filter_y, 10, err,
                       NULL, 0, 0, 0, 0, 0);
}

int avm_gpu_inter_convolve_scaled_compound(
    avm_gpu_ctx *ctx, avm_gpu_inter *in, const uint16_t *src, int src_stride,
    int src_rows, int sx, int sy, const int16_t *filter_x,
    const int16_t *filter_y, int subpel_x_qn, int x_step_qn, int subpel_y_qn,
    int y_step_qn, int w, int h, int bd, int round0, int round1,
    uint16_t *conv, int conv_stride, int do_average, int uneven, int fwd,
    int bck, uint16_t *dst, int dst_stride, char err[AVM_GPU_ERR_LEN]) {
  if (!conv) {
    snprintf(err, AVM_GPU_ERR_LEN, "compound needs a conv buffer");
    return -1;
  }
  /* av2_highbd_convolve_2d_scale_c() with conv_params->is_compound: the same
   * two-pass filter as the single reference, but the first reference writes
   * the 16-bit buffer and the second averages it with its own sum. */
  return convolve_impl(ctx, in, src, src_stride, src_rows, sx, sy, filter_x,
                       subpel_x_qn, subpel_y_qn, x_step_qn, y_step_qn, w, h, bd,
                       dst, dst_stride, 1, round0, round1, filter_y, 10, err,
                       conv, conv_stride, do_average, uneven, fwd, bck);
}

int avm_gpu_inter_cwp_convolve(
    avm_gpu_ctx *ctx, avm_gpu_inter *in, const uint16_t *src, int src_stride,
    int src_rows, int sx, int sy, const int16_t *filter_x,
    const int16_t *filter_y, int subpel_x_qn, int subpel_y_qn, int w, int h,
    int bd, int round0, int round1, uint16_t *conv, int conv_stride,
    int conv_index, int do_average, int uneven, int fwd, int bck, int scaled,
    int x_step_qn, int y_step_qn, uint16_t *dst, int dst_stride,
    char err[AVM_GPU_ERR_LEN]) {
  if (!src || !filter_x || !filter_y || !conv || !dst) {
    snprintf(err, AVM_GPU_ERR_LEN, "invalid arguments");
    return -1;
  }
  if (w <= 0 || h <= 0 || w > 64 || h > 64 || sx < 3 || sy < 3) {
    snprintf(err, AVM_GPU_ERR_LEN, "unsupported block %dx%d at %d,%d", w, h, sx,
             sy);
    return -1;
  }
  const int sb = scaled ? 10 : 4;
  const int smask = (1 << sb) - 1;
  const int x_step = scaled ? x_step_qn : 16;
  const int y_step = scaled ? y_step_qn : 16;
  if (subpel_x_qn < 0 || subpel_x_qn > smask || subpel_y_qn < 0 ||
      subpel_y_qn > smask || x_step <= 0 || y_step <= 0 ||
      x_step > (1 << sb) || y_step > (1 << sb)) {
    snprintf(err, AVM_GPU_ERR_LEN, "subpel position or step out of range");
    return -1;
  }
  /* The taps reach last_base + 4 past the block origin on each axis. */
  const int last_bx = ((w - 1) * x_step + subpel_x_qn) >> sb;
  const int last_by = ((h - 1) * y_step + subpel_y_qn) >> sb;
  if (sx + last_bx + 4 > src_stride || sy + last_by + 4 > src_rows) {
    snprintf(err, AVM_GPU_ERR_LEN, "block overlaps the filter border");
    return -1;
  }

  const size_t pixels = (size_t)src_stride * (size_t)src_rows;
  const int packed = pack16_for(src_stride);
  if (!packed && pixels > in->expand_cap) {
    free(in->expand);
    in->expand = malloc(pixels * sizeof(*in->expand));
    in->expand_cap = in->expand ? pixels : 0;
    if (!in->expand) {
      snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
      return -1;
    }
  }
  if (!packed) {
    for (size_t i = 0; i < pixels; i++) in->expand[i] = src[i];
  }

  const size_t conv_n = (size_t)conv_stride * (size_t)h;
  const size_t dst_n = (size_t)dst_stride * (size_t)h;
  if (ensure_buf(ctx, in, &in->src, &in->src_cap,
                 (VkDeviceSize)pixels * (packed ? 2 : 4), err) ||
      ensure_buf(ctx, in, &in->filt, &in->filt_cap,
                 (VkDeviceSize)(2 * INTER_FILTER_ENTRIES) * 4, err) ||
      ensure_buf(ctx, in, &in->cwpconv_conv, &in->cwpconv_conv_cap,
                 (VkDeviceSize)conv_n * 4, err) ||
      ensure_buf(ctx, in, &in->cwpconv_dst, &in->cwpconv_dst_cap,
                 (VkDeviceSize)dst_n * 4, err)) {
    return -1;
  }
  const size_t src_bytes = (size_t)pixels * (packed ? 2 : 4);
  const size_t filt_bytes = (size_t)(2 * INTER_FILTER_ENTRIES) * 4;
  const size_t conv_bytes = conv_n * 4, dst_bytes = dst_n * 4;
  uint8_t *src_host = (uint8_t *)inter_host(ctx, in, &in->src, src_bytes, err);
  int *fp = (int *)inter_host(ctx, in, &in->filt, filt_bytes, err);
  uint32_t *cv = inter_host(ctx, in, &in->cwpconv_conv, conv_bytes, err);
  uint32_t *dp = inter_host(ctx, in, &in->cwpconv_dst, dst_bytes, err);
  if (!src_host || !fp || !cv || !dp) return -1;
  if (packed) {
    memcpy(src_host, src, (size_t)pixels * 2);
  } else {
    memcpy(src_host, in->expand, (size_t)pixels * 4);
  }
  for (int i = 0; i < INTER_FILTER_ENTRIES; i++) {
    fp[i] = (int)filter_x[i];
    fp[INTER_FILTER_ENTRIES + i] = (int)filter_y[i];
  }
  for (size_t i = 0; i < conv_n; i++) cv[i] = conv[i];
  for (size_t i = 0; i < dst_n; i++) dp[i] = dst[i];
  if (inter_push(ctx, in, &in->src, src_bytes, err) != 0 ||
      inter_push(ctx, in, &in->filt, filt_bytes, err) != 0 ||
      inter_push(ctx, in, &in->cwpconv_conv, conv_bytes, err) != 0 ||
      inter_push(ctx, in, &in->cwpconv_dst, dst_bytes, err) != 0) {
    return -1;
  }

  const avm_gpu_buf bufs[4] = { in->src, in->filt, in->cwpconv_conv,
                                in->cwpconv_dst };
  cwp_constants pc = { (uint32_t)src_stride,
                       (uint32_t)conv_stride,
                       (uint32_t)dst_stride,
                       (uint32_t)w,
                       (uint32_t)h,
                       (uint32_t)bd,
                       (uint32_t)subpel_x_qn,
                       (uint32_t)subpel_y_qn,
                       (uint32_t)sx,
                       (uint32_t)sy,
                       (uint32_t)round0,
                       (uint32_t)round1,
                       (uint32_t)do_average,
                       (uint32_t)uneven,
                       (uint32_t)fwd,
                       (uint32_t)bck,
                       (uint32_t)conv_index,
                       (uint32_t)(scaled ? 0
                                         : (subpel_x_qn == 0
                                                ? (subpel_y_qn == 0 ? 3 : 2)
                                                : (subpel_y_qn == 0 ? 1 : 0))),
                       (uint32_t)sb,
                       (uint32_t)x_step,
                       (uint32_t)y_step,
                       (uint32_t)packed };
  const uint32_t gx = (uint32_t)((w + 7) / 8);
  const uint32_t gy = (uint32_t)((h + 7) / 8);
  const int rc = avm_gpu_pipeline_dispatch_2d(ctx, in->cwpconv_pipe, bufs, 4,
                                              &pc, sizeof(pc), gx, gy, err);
  if (rc != 0) return rc;

  if (inter_pull(ctx, in, &in->cwpconv_conv, conv_bytes, err) != 0) return -1;
  if (!(conv_index == 0 && do_average == 0) &&
      inter_pull(ctx, in, &in->cwpconv_dst, dst_bytes, err) != 0) {
    return -1;
  }
  const uint32_t *res_conv = cv;
  for (size_t i = 0; i < conv_n; i++) conv[i] = (uint16_t)res_conv[i];
  if (!(conv_index == 0 && do_average == 0)) {
    const uint32_t *res_dst = dp;
    for (size_t i = 0; i < dst_n; i++) dst[i] = (uint16_t)res_dst[i];
  }
  return 0;
}

int avm_gpu_inter_blend_a64_d16(avm_gpu_ctx *ctx, avm_gpu_inter *in,
                                const uint16_t *src0, int s0_stride,
                                const uint16_t *src1, int s1_stride,
                                const uint8_t *mask, int mask_stride, int w,
                                int h, int subw, int subh, int bd,
                                int round_offset, int round_bits, uint16_t *dst,
                                int dst_stride, char err[AVM_GPU_ERR_LEN]) {
  if (!src0 || !src1 || !mask || !dst) {
    snprintf(err, AVM_GPU_ERR_LEN, "invalid arguments");
    return -1;
  }
  if (w <= 0 || h <= 0 || w > 64 || h > 64) {
    snprintf(err, AVM_GPU_ERR_LEN, "unsupported block size %dx%d", w, h);
    return -1;
  }
  if (round_bits < 1 || round_bits > 24) {
    snprintf(err, AVM_GPU_ERR_LEN, "unsupported round_bits %d", round_bits);
    return -1;
  }
  const int mw = subw ? 2 * w : w;
  const int mh = subh ? 2 * h : h;
  if (mask_stride < mw) {
    snprintf(err, AVM_GPU_ERR_LEN, "mask stride too small");
    return -1;
  }

  const size_t s0_n = (size_t)s0_stride * h;
  const size_t s1_n = (size_t)s1_stride * h;
  const size_t mk_n = (size_t)mask_stride * mh;
  const size_t dst_n = (size_t)dst_stride * h;

  if (ensure_buf(ctx, in, &in->s0, &in->s0_cap, (VkDeviceSize)s0_n * 4, err) ||
      ensure_buf(ctx, in, &in->s1, &in->s1_cap, (VkDeviceSize)s1_n * 4, err) ||
      ensure_buf(ctx, in, &in->mask, &in->mask_cap, (VkDeviceSize)mk_n * 4, err) ||
      ensure_buf(ctx, in, &in->bdst, &in->bdst_cap, (VkDeviceSize)dst_n * 4, err)) {
    return -1;
  }

  const size_t s0_bytes = s0_n * 4, s1_bytes = s1_n * 4;
  const size_t mk_bytes = mk_n * 4, bdst_bytes = dst_n * 4;
  uint32_t *p0 = inter_host(ctx, in, &in->s0, s0_bytes, err);
  uint32_t *p1 = inter_host(ctx, in, &in->s1, s1_bytes, err);
  uint32_t *pm = inter_host(ctx, in, &in->mask, mk_bytes, err);
  uint32_t *bdst_host = inter_host(ctx, in, &in->bdst, bdst_bytes, err);
  if (!p0 || !p1 || !pm || !bdst_host) return -1;
  for (size_t i = 0; i < s0_n; i++) p0[i] = src0[i];
  for (size_t i = 0; i < s1_n; i++) p1[i] = src1[i];
  for (size_t i = 0; i < mk_n; i++) pm[i] = mask[i];
  if (inter_push(ctx, in, &in->s0, s0_bytes, err) != 0 ||
      inter_push(ctx, in, &in->s1, s1_bytes, err) != 0 ||
      inter_push(ctx, in, &in->mask, mk_bytes, err) != 0 ||
      inter_push(ctx, in, &in->bdst, bdst_bytes, err) != 0) {
    return -1;
  }

  const avm_gpu_buf bufs[4] = { in->s0, in->s1, in->mask, in->bdst };
  blend_pc pc = { (uint32_t)s0_stride,
                  (uint32_t)s1_stride,
                  (uint32_t)dst_stride,
                  (uint32_t)mask_stride,
                  (uint32_t)w,
                  (uint32_t)h,
                  (uint32_t)(subw ? 1 : 0),
                  (uint32_t)(subh ? 1 : 0),
                  (uint32_t)bd,
                  1,
                  (int32_t)round_offset,
                  (uint32_t)round_bits };
  const uint32_t gx = (uint32_t)((w + 7) / 8);
  const uint32_t gy = (uint32_t)((h + 7) / 8);
  const int rc = avm_gpu_pipeline_dispatch_2d(ctx, in->blend_pipe, bufs, 4, &pc,
                                              sizeof(pc), gx, gy, err);
  if (rc != 0) return rc;

  if (inter_pull(ctx, in, &in->bdst, bdst_bytes, err) != 0) return -1;
  const uint32_t *res = bdst_host;
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      dst[(size_t)y * dst_stride + x] = (uint16_t)res[y * dst_stride + x];
    }
  }
  return 0;
}
