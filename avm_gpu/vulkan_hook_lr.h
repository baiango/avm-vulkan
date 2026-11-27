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

#ifndef AVM_AVM_GPU_VULKAN_HOOK_LR_H_
#define AVM_AVM_GPU_VULKAN_HOOK_LR_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Loop-restoration seam for the PC-Wiener engine.
 *
 * Called from av2/common/restoration.c::pc_wiener_stripe_highbd() for one
 * restoration procunit when the library is built with AVM_GPU_HOOKS. It runs
 * the classification (and, unless classify_only, the filtering) of
 * apply_pc_wiener_highbd() on the Vulkan kernels and returns 1 when the GPU
 * handled it; 0 means the caller must run the C function.
 *
 * Everything the C derives from the quantiser arrives as an argument (the
 * qval/tskip LUT and the feature normalisers), as do the filter bank, the
 * class -> filter selector, the thresholds and the class LUT, so this file owns
 * no tables.
 *
 * Inert unless AV2_GPU_LR=1. AV2_GPU_LR_STATS=1 prints the number of
 * procunits the GPU handled at exit.
 */
int avm_gpu_hook_pcwiener(
    const uint16_t *dgd, int width, int height, int stride, uint16_t *dst,
    int dst_stride, const uint8_t *tskip, int tskip_stride,
    uint8_t *wiener_class_id, int wiener_class_id_stride, int bit_depth,
    int classify_only, const int16_t (*filters)[13], const uint8_t *selector,
    const int32_t *qval_lut, const int32_t *normalizers,
    const int32_t *thresholds, const uint8_t *lut_to_class, int prec_feature,
    int thr_shift, int tskip_zero, const uint8_t *skip_block,
    int skip_stride, int num_taps, const int (*config)[3], int num_pixels,
    int prec_bits);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AVM_AVM_GPU_VULKAN_HOOK_LR_H_
