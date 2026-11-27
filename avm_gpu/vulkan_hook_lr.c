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

#include "avm_gpu/vulkan_hook_lr.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "avm_gpu/pcwiener.h"
#include "avm_gpu/vulkan_gpu.h"

/* Restoration stripes carry three border samples on every side (RESTORATION_
 * BORDER_VERT/HORZ). The kernels read up to three rows/columns outside the
 * visible area (gradient rows r-2..r+2 at the top, filter taps +-3), so the
 * window has to carry all three of those border samples. */
#define LR_LEAD 3
#define LR_MAX_W 64
#define LR_MAX_H 512
#define LR_STRIDE (LR_MAX_W + 2 * LR_LEAD + 2)

typedef struct lr_hook_state {
  int state; /* 0 = uninitialised, 1 = off, 2 = ready */
  avm_gpu_ctx *ctx;
  avm_gpu_pcwiener *pc;
  unsigned long units;
  uint16_t plane[LR_MAX_H + 2 * LR_LEAD][LR_STRIDE];
  uint16_t dst_plane[LR_MAX_H + 2 * LR_LEAD][LR_STRIDE];
  uint8_t cells[((LR_MAX_W + 3) / 4) * ((LR_MAX_H + 3) / 4)];
  /* Classification is computed per 4x4 cell in a compact grid; the caller's
   * class-id buffer is indexed by MI column (4 samples), so the two strides
   * differ and the classes are copied across. */
  uint8_t classes[((LR_MAX_W + 3) / 4) * ((LR_MAX_H + 3) / 4)];
} lr_hook_state;

static lr_hook_state g_hook;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static void print_stats(void) {
  if (getenv("AV2_GPU_LR_STATS")) {
    fprintf(stderr, "avm_gpu: %lu loop-restoration procunits on the GPU\n",
            g_hook.units);
  }
}

/* Caller holds g_lock. */
static int hook_ready_locked(void) {
  if (g_hook.state) return g_hook.state == 2;

  const char *env = getenv("AV2_GPU_LR");
  if (!env || !*env || !strcmp(env, "0")) {
    g_hook.state = 1;
    return 0;
  }
  char err[AVM_GPU_ERR_LEN];
  if (avm_gpu_ctx_create(&g_hook.ctx, err) != 0 ||
      avm_gpu_pcwiener_create(g_hook.ctx, &g_hook.pc, err) != 0) {
    fprintf(stderr, "avm_gpu: LR hook disabled (%s)\n", err);
    avm_gpu_ctx_destroy(g_hook.ctx);
    g_hook.ctx = NULL;
    g_hook.state = 1;
    return 0;
  }
  atexit(print_stats);
  fprintf(stderr, "avm_gpu: loop-restoration hook enabled on %s\n",
          avm_gpu_device_name(g_hook.ctx));
  g_hook.state = 2;
  return 1;
}

int avm_gpu_hook_pcwiener(const uint16_t *dgd, int width, int height,
                          int stride, uint16_t *dst, int dst_stride,
                          const uint8_t *tskip, int tskip_stride,
                          uint8_t *class_id, int class_id_stride, int bd,
                          int classify_only, const int16_t (*filters)[13],
                          const uint8_t *selector, const int32_t *qval_lut,
                          const int32_t *normalizers, const int32_t *thresholds,
                          const uint8_t *lut_to_class, int prec_feature,
                          int thr_shift, int tskip_zero,
                          const uint8_t *skip_block, int skip_stride,
                          int num_taps, const int (*config)[3], int num_pixels,
                          int prec_bits) {
  if (!dgd || !dst || !class_id || !skip_block || width <= 0 || height <= 0 ||
      width > LR_MAX_W || height > LR_MAX_H || stride < width + 3 ||
      dst_stride < width + 3) {
    return 0;
  }

  pthread_mutex_lock(&g_lock);
  int handled = 0;
  if (hook_ready_locked()) {
    /* One row of lead above, three below (the filter reads r + 3) and the same
     * on the columns; the stripe's own border supplies all of it. */
    const int rows = height + 3 + LR_LEAD;
    const int cols = width + 2 * LR_LEAD;
    for (int y = -LR_LEAD; y < rows - LR_LEAD; y++) {
      memcpy(&g_hook.plane[y + LR_LEAD][0], dgd + (size_t)y * stride - LR_LEAD,
             (size_t)(cols + 2) * sizeof(uint16_t));
      memcpy(&g_hook.dst_plane[y + LR_LEAD][0],
             dst + (size_t)y * dst_stride - LR_LEAD,
             (size_t)(cols + 2) * sizeof(uint16_t));
    }
    const uint16_t *plane = &g_hook.plane[LR_LEAD][LR_LEAD];
    uint16_t *dplane = &g_hook.dst_plane[LR_LEAD][LR_LEAD];

    const int cell_stride = (width + 3) / 4;
    const int cell_rows = (height + 3) / 4;
    for (int by = 0; by < cell_rows; by++) {
      for (int bx = 0; bx < cell_stride; bx++) {
        g_hook.cells[by * cell_stride + bx] = skip_block[by * skip_stride + bx];
      }
    }

    char err[AVM_GPU_ERR_LEN];
    avm_gpu_pcwiener_classify_params cp;
    memset(&cp, 0, sizeof(cp));
    cp.width = width;
    cp.height = height;
    cp.stride = LR_STRIDE;
    cp.rows = rows;
    cp.lead_rows = LR_LEAD;
    cp.lead_cols = LR_LEAD;
    cp.bd = bd;
    cp.cell_stride = cell_stride;
    cp.cell_rows = cell_rows;
    cp.tskip_stride = tskip_zero ? 0 : tskip_stride;
    cp.qval_lut = qval_lut;
    cp.normalizers = normalizers;
    cp.thresholds = thresholds;
    cp.lut_to_class = lut_to_class;
    cp.prec_feature = prec_feature;
    cp.thr_shift = thr_shift;

    if (avm_gpu_pcwiener_classify(g_hook.ctx, g_hook.pc, &cp, plane,
                                  tskip_zero ? NULL : tskip, g_hook.classes,
                                  err) == 0) {
      handled = 1;
      if (!classify_only) {
        avm_gpu_pcwiener_params fp;
        memset(&fp, 0, sizeof(fp));
        fp.width = width;
        fp.height = height;
        fp.stride = LR_STRIDE;
        fp.dst_stride = LR_STRIDE;
        fp.rows = rows;
        fp.lead_rows = LR_LEAD;
        fp.lead_cols = LR_LEAD;
        fp.bd = bd;
        /* The filter reads the compact grid the classifier wrote. */
        fp.class_id_stride = cell_stride;
        fp.num_sym = num_pixels / 2;
        fp.singleton_index = config[num_pixels - 1][2];
        fp.prec_bits = prec_bits;
        fp.num_taps = num_taps;
        fp.config = config;
        fp.skip_block = g_hook.cells;
        handled = avm_gpu_pcwiener_filter(g_hook.ctx, g_hook.pc, &fp, plane,
                                          g_hook.classes, selector,
                                          &filters[0][0], dplane, err) == 0;
      }
      if (handled) {
        g_hook.units++;
        /* Mirror the reference, which stores this procunit's classes in the
         * caller's buffer at (i >> MI_SIZE_LOG2, j >> MI_SIZE_LOG2). */
        for (int by = 0; by < cell_rows; by++) {
          for (int bx = 0; bx < cell_stride; bx++) {
            class_id[by * class_id_stride + bx] =
                g_hook.classes[by * cell_stride + bx];
          }
        }
        if (!classify_only) {
          for (int y = 0; y < height; y++) {
            memcpy(dst + (size_t)y * dst_stride, &dplane[y * LR_STRIDE],
                   (size_t)width * sizeof(uint16_t));
          }
        }
      }
    } else {
      fprintf(stderr, "avm_gpu: LR hook fell back (%s)\n", err);
    }
  }
  pthread_mutex_unlock(&g_lock);
  return handled;
}
