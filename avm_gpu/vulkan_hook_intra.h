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

#ifndef AVM_AVM_GPU_VULKAN_HOOK_INTRA_H_
#define AVM_AVM_GPU_VULKAN_HOOK_INTRA_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Intra prediction seam for the modes av2_build_intra_predictors_high_default()
 * reaches: DC (with the n_top/n_left availability variant), SMOOTH,
 * SMOOTH_V, SMOOTH_H and PAETH. above points at the first above sample and
 * left at the first left sample of the neighbour arrays the builder already
 * filled (edge filtering included); mode is AVM's PREDICTION_MODE. Directional
 * modes, DIP, MRL, palette and CFL stay on the CPU. Returns 1 when the GPU
 * wrote the block. Inert unless AV2_GPU_INTRA=1; AV2_GPU_INTRA_STATS=1 prints
 * the number of blocks at exit.
 */
int avm_gpu_hook_intra(int mode, uint16_t *dst, int dst_stride,
                       const uint16_t *above, const uint16_t *left, int w,
                       int h, int bd, int n_top_px, int n_left_px);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AVM_AVM_GPU_VULKAN_HOOK_INTRA_H_
