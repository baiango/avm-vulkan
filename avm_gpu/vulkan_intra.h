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

#ifndef AVM_AVM_GPU_VULKAN_INTRA_H_
#define AVM_AVM_GPU_VULKAN_INTRA_H_

#include <stdint.h>

#include "avm_gpu/vulkan_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Must match the mode numbering in kernels/intra.comp. */
typedef enum avm_gpu_intra_mode {
  AVM_GPU_INTRA_DC = 0,
  AVM_GPU_INTRA_V = 1,
  AVM_GPU_INTRA_H = 2,
  AVM_GPU_INTRA_DC_LEFT = 3,
  AVM_GPU_INTRA_DC_TOP = 4,
  AVM_GPU_INTRA_DC_128 = 5,
  AVM_GPU_INTRA_SMOOTH = 6,
  AVM_GPU_INTRA_SMOOTH_V = 7,
  AVM_GPU_INTRA_SMOOTH_H = 8,
  AVM_GPU_INTRA_PAETH = 9
} avm_gpu_intra_mode;

typedef struct avm_gpu_intra avm_gpu_intra;

int avm_gpu_intra_create(avm_gpu_ctx *ctx, avm_gpu_intra **out,
                         char err[AVM_GPU_ERR_LEN]);
void avm_gpu_intra_destroy(avm_gpu_ctx *ctx, avm_gpu_intra *intra);

/*
 * Block-level availability rule used by AV2: with no neighbours at all the
 * block uses DC_128, with only one neighbour the DC variant of that side, and
 * with both the full DC. Returns one of the mode enum values.
 */
int avm_gpu_intra_auto_mode(int has_above, int has_left);

/*
 * Builds the above/left edge arrays for a block at (x, y) of a plane, with
 * AVM's frame clamping: samples past the visible frame repeat the last
 * visible column/row, and an unavailable side is filled with the neutral value
 * 1 << (bd - 1). above must hold bw samples, left must hold bh samples.
 * Returns 0 on success.
 */
void avm_gpu_intra_edges(const uint16_t *plane, int stride, int x, int y,
                         int bw, int bh, int frame_w, int frame_h, int bd,
                         int has_above, int has_left, uint16_t *above,
                         uint16_t *left);

/*
 * Predicts one bw x bh block into dst with the given row stride, reading above
 * (bw samples, plus above[bw] for SMOOTH*) and left (bh samples, plus
 * left[bh] for SMOOTH*). Matches
 * avm_highbd_{dc,v,h,dc_left,dc_top,dc_128}_predictor_*_c bit for bit.
 */
/*
 * Directional prediction (zones 1-3), matching
 * av2_highbd_dr_prediction_z{1,2,3}_c. above and left point at the top-left
 * sample, exactly like the C reference (above[-1] is the top-left, above[0]
 * the first above sample). The caller must provide VALID samples for
 * above[-4 .. bw+bh+7] and left[-4 .. bw+bh+7], because steep zone-2 angles
 * read past the top-left into the extended row, as they do in AVM.
 * zone is 1, 2 or 3.
 */
int avm_gpu_intra_dr_predict(avm_gpu_ctx *ctx, avm_gpu_intra *intra,
                             const uint16_t *above, const uint16_t *left,
                             int bw, int bh, int zone, int dx, int dy,
                             int mrl_index, uint16_t *dst, int stride,
                             char err[AVM_GPU_ERR_LEN]);

/*
 * Same as avm_gpu_intra_predict(), but for the modes that need the top-left
 * sample (PAETH) and the far edges (SMOOTH*: above[bw], left[bh]).
 */
int avm_gpu_intra_predict_tl(avm_gpu_ctx *ctx, avm_gpu_intra *intra,
                             const uint16_t *above, const uint16_t *left,
                             int top_left, int bw, int bh, int mode, int bd,
                             uint16_t *dst, int stride,
                             char err[AVM_GPU_ERR_LEN]);

int avm_gpu_intra_predict(avm_gpu_ctx *ctx, avm_gpu_intra *intra,
                          const uint16_t *above, const uint16_t *left, int bw,
                          int bh, int mode, int bd, uint16_t *dst, int stride,
                          char err[AVM_GPU_ERR_LEN]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AVM_AVM_GPU_VULKAN_INTRA_H_
