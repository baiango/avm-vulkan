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

#ifndef AVM_AVM_GPU_VULKAN_CDEF_H_
#define AVM_AVM_GPU_VULKAN_CDEF_H_

#include <stdint.h>

#include "avm_gpu/vulkan_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct avm_gpu_cdef avm_gpu_cdef;

int avm_gpu_cdef_create(avm_gpu_ctx *ctx, avm_gpu_cdef **out,
                        char err[AVM_GPU_ERR_LEN]);
void avm_gpu_cdef_destroy(avm_gpu_ctx *ctx, avm_gpu_cdef *cdef);

/*
 * One CDEF block, matching av2_cdef_filter_16_{0,1,2,3}_c: in is the CDEF
 * in-buffer laid out with the given stride, (i0, j0) the block origin inside
 * it. enable_pri / enable_sec select the 0..3 wrapper variant.
 * dst receives w x h filtered samples with dstride.
 */
int avm_gpu_cdef_filter16(avm_gpu_ctx *ctx, avm_gpu_cdef *cdef,
                          const uint16_t *in, int in_rows, int stride, int i0,
                          int j0, int w, int h, int dir, int pri_strength,
                          int sec_strength, int pri_damping, int sec_damping,
                          int coeff_shift, int enable_pri, int enable_sec,
                          uint16_t *dst, int dstride,
                          char err[AVM_GPU_ERR_LEN]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AVM_AVM_GPU_VULKAN_CDEF_H_
