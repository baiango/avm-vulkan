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

#ifndef AVM_AVM_GPU_VULKAN_PCWIENER_H_
#define AVM_AVM_GPU_VULKAN_PCWIENER_H_

#include <stdint.h>

#include "avm_gpu/vulkan_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * PC-Wiener loop-restoration filter stage.
 *
 * av2/common/restoration.c::apply_pc_wiener_highbd() classifies each 4x4 block
 * (class id) and then filters the block with pcwiener_filters_luma[
 * filter_selector[class id]]. This module does the filtering half on the GPU,
 * bit-identically to av2_convolve_symmetric_highbd_c() for the same class ids,
 * selector and filter bank; classification is not part of it.
 */
typedef struct avm_gpu_pcwiener avm_gpu_pcwiener;

/* The NonsepFilterConfig the PC-Wiener path builds in restoration.c:
 * prec_bits = PC_WIENER_PREC_FILTER, config = wienerns_simd_config_y. */
typedef struct avm_gpu_pcwiener_params {
  int width, height;   /* visible area */
  int stride;          /* dgd row stride, in samples */
  int dst_stride;      /* dst row stride, in samples */
  int rows;            /* upload rows: height + 2 * lead_rows */
  int lead_rows;       /* border rows above/below the visible origin */
  int lead_cols;       /* border columns left of the visible origin */
  int bd;              /* bit depth */
  int class_id_stride; /* class-id plane stride, in 4x4 blocks */
  int num_sym;         /* symmetric tap pairs, num_pixels / 2 */
  int singleton_index; /* filter position of the unconstrained centre tap */
  int prec_bits;       /* filter_config->prec_bits */
  int num_taps;        /* taps per filter (NUM_PC_WIENER_TAPS_LUMA) */
  const int (*config)[3]; /* NonsepFilterConfig::config, num_pixels entries */
} avm_gpu_pcwiener_params;

int avm_gpu_pcwiener_create(avm_gpu_ctx *ctx, avm_gpu_pcwiener **out,
                            char err[AVM_GPU_ERR_LEN]);
void avm_gpu_pcwiener_destroy(avm_gpu_ctx *ctx, avm_gpu_pcwiener *pc);

/*
 * dgd must point at the visible origin and have lead_rows / lead_cols of valid
 * border samples around it (the same assumption the C makes). filters holds
 * num_filters * num_taps taps; filter_selector maps a class id to a filter.
 */
int avm_gpu_pcwiener_filter(avm_gpu_ctx *ctx, avm_gpu_pcwiener *pc,
                            const avm_gpu_pcwiener_params *params,
                            const uint16_t *dgd, const uint8_t *class_id,
                            const uint8_t *filter_selector,
                            const int16_t *filters, uint16_t *dst,
                            char err[AVM_GPU_ERR_LEN]);

/*
 * Classification half of apply_pc_wiener_highbd(): one invocation per 4x4
 * block, producing the class ids the filter stage consumes. The quantiser-
 * dependent tables (qval/tskip LUT, feature normalisers, thresholds and the
 * class LUT) are inputs, taken from the same place the C gets them
 * (fill_qval_given_tskip_lut, set_feature_normalizers, pc_wiener_thresholds,
 * pc_wiener_lut_to_class_index).
 */
typedef struct avm_gpu_pcwiener_classify_params {
  int width, height;   /* visible area */
  int stride;          /* dgd row stride, in samples */
  int rows;            /* upload rows (>= height + 2 + lead_rows) */
  int lead_rows;       /* border rows above the visible origin */
  int lead_cols;       /* border columns left of the visible origin */
  int bd;              /* bit depth */
  int cell_stride;     /* class-id plane stride, in 4x4 blocks */
  int cell_rows;       /* class-id plane rows */
  int tskip_stride;    /* tx-skip plane stride in blocks, 0 when unused */
  const int32_t *qval_lut;     /* [256][NUM_PC_WIENER_FEATURES] */
  const int32_t *normalizers;  /* [NUM_PC_WIENER_FEATURES + 1] */
  const int32_t *thresholds;   /* [NUM_PC_WIENER_FEATURES] */
  const uint8_t *lut_to_class; /* [PC_WIENER_LUT_SIZE] */
  int prec_feature;            /* PC_WIENER_PREC_FEATURE */
  int thr_shift;               /* pc_wiener_threshold_shift */
} avm_gpu_pcwiener_classify_params;

int avm_gpu_pcwiener_classify(avm_gpu_ctx *ctx, avm_gpu_pcwiener *pc,
                              const avm_gpu_pcwiener_classify_params *params,
                              const uint16_t *dgd, const uint8_t *tskip,
                              uint8_t *class_id, char err[AVM_GPU_ERR_LEN]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AVM_AVM_GPU_VULKAN_PCWIENER_H_
