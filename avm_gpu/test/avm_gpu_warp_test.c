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

/*
 * Parity gate for the Vulkan warped-motion kernel against
 * av2_highbd_warp_affine_c() (single-reference case).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "av2/common/convolve.h"
#include "av2/common/warped_motion.h"

/* warped_motion.h declares the dispatch wrapper, not the _c kernel. */
void av2_highbd_warp_affine_c(const int32_t *mat, const uint16_t *ref,
                              int width, int height, int stride,
                              uint16_t *pred, int p_col, int p_row, int p_width,
                              int p_height, int p_stride, int subsampling_x,
                              int subsampling_y, int bd,
                              ConvolveParams *conv_params, int16_t alpha,
                              int16_t beta, int16_t gamma, int16_t delta);

#include "avm_gpu/vulkan_gpu.h"
#include "avm_gpu/vulkan_warp.h"

#define REF_W 96
#define REF_H 96
#define PRED_STRIDE 96
#define BD 10

static unsigned int rng_state = 0x6a09e667u;
static unsigned int rnd(void) {
  rng_state = rng_state * 1664525u + 1013904223u;
  return rng_state >> 8;
}

int main(void) {
  static const int32_t kMat[3][6] = {
    { 0, 0, 65536, 0, 0, 65536 },
    { 0, 0, 65336, 900, -700, 65736 },
    { 120, -80, 64136, 1800, -1300, 66436 },
  };
  static const int kShear[3][4] = {
    { 0, 0, 0, 0 },
    { 4, -3, 2, -2 },
    { 12, -8, 9, -7 },
  };
  static const int kSize[4][2] = {
    { 8, 8 }, { 16, 16 }, { 16, 8 }, { 32, 16 },
  };

  char err[AVM_GPU_ERR_LEN];
  avm_gpu_ctx *ctx = NULL;
  if (avm_gpu_ctx_create(&ctx, err) != 0) {
    fprintf(stderr, "avm_gpu: %s\n", err);
    return 77; /* skip: no Vulkan device */
  }
  avm_gpu_warp *warp = NULL;
  if (avm_gpu_warp_create(ctx, &warp, err) != 0) {
    fprintf(stderr, "avm_gpu warp: %s\n", err);
    avm_gpu_ctx_destroy(ctx);
    return 1;
  }
  printf("device: %s\n", avm_gpu_device_name(ctx));

  uint16_t *ref = malloc(sizeof(uint16_t) * REF_W * REF_H);
  uint16_t *cpu = malloc(sizeof(uint16_t) * PRED_STRIDE * REF_H);
  uint16_t *gpu = malloc(sizeof(uint16_t) * PRED_STRIDE * REF_H);
  if (!ref || !cpu || !gpu) return 1;
  for (int i = 0; i < REF_W * REF_H; i++) ref[i] = (uint16_t)(rnd() & 0x3ff);

  ConvolveParams cp;
  memset(&cp, 0, sizeof(cp));
  cp.is_compound = 0;
  cp.round_0 = 2;
  cp.round_1 = 0;

  const int p_col = 8, p_row = 8;
  int cases = 0, failures = 0;

  for (size_t m = 0; m < 3; m++) {
    for (size_t sh = 0; sh < 3; sh++) {
      for (size_t s = 0; s < 4; s++) {
        const int pw = kSize[s][0];
        const int ph = kSize[s][1];
        memset(cpu, 0, sizeof(uint16_t) * PRED_STRIDE * REF_H);
        memset(gpu, 0, sizeof(uint16_t) * PRED_STRIDE * REF_H);

        av2_highbd_warp_affine_c(kMat[m], ref, REF_W, REF_H, REF_W, cpu, p_col,
                                 p_row, pw, ph, PRED_STRIDE, 0, 0, BD, &cp,
                                 (int16_t)kShear[sh][0], (int16_t)kShear[sh][1],
                                 (int16_t)kShear[sh][2],
                                 (int16_t)kShear[sh][3]);

        if (avm_gpu_warp_affine(ctx, warp, kMat[m], ref, REF_W, REF_H, REF_W,
                                gpu, p_col, p_row, pw, ph, PRED_STRIDE, BD,
                                cp.round_0, kShear[sh][0], kShear[sh][1],
                                kShear[sh][2], kShear[sh][3], 0, 0,
                                &av2_warped_filter[0][0], err) != 0) {
          fprintf(stderr, "avm_gpu warp: %s\n", err);
          return 1;
        }

        int diff = 0;
        for (int y = 0; y < REF_H; y++) {
          for (int x = 0; x < PRED_STRIDE; x++) {
            if (cpu[y * PRED_STRIDE + x] != gpu[y * PRED_STRIDE + x]) diff++;
          }
        }
        cases++;
        if (diff) {
          failures++;
          if (failures < 6) {
            printf("  mat %zu shear %zu %dx%d: %d samples differ\n", m, sh, pw,
                   ph, diff);
          }
        }
      }
    }
  }

  /* Chroma: the projection shifts the block centre up to luma coordinates
   * and back down, and the blocks are 4 samples in chroma units. */
  static const int kSub[3][2] = { { 1, 1 }, { 1, 0 }, { 0, 1 } };
  static const int kChromaSize[3][2] = { { 4, 4 }, { 8, 8 }, { 8, 4 } };
  for (size_t m = 0; m < 3; m++) {
    for (size_t sh = 0; sh < 3; sh++) {
      for (size_t sub = 0; sub < 3; sub++) {
        for (size_t s = 0; s < 3; s++) {
          const int pw = kChromaSize[s][0];
          const int ph = kChromaSize[s][1];
          memset(cpu, 0, sizeof(uint16_t) * PRED_STRIDE * REF_H);
          memset(gpu, 0, sizeof(uint16_t) * PRED_STRIDE * REF_H);

          av2_highbd_warp_affine_c(kMat[m], ref, REF_W, REF_H, REF_W, cpu,
                                   p_col, p_row, pw, ph, PRED_STRIDE,
                                   kSub[sub][0], kSub[sub][1], BD, &cp,
                                   (int16_t)kShear[sh][0],
                                   (int16_t)kShear[sh][1],
                                   (int16_t)kShear[sh][2],
                                   (int16_t)kShear[sh][3]);
          if (avm_gpu_warp_affine(ctx, warp, kMat[m], ref, REF_W, REF_H, REF_W,
                                  gpu, p_col, p_row, pw, ph, PRED_STRIDE, BD,
                                  cp.round_0, kShear[sh][0], kShear[sh][1],
                                  kShear[sh][2], kShear[sh][3], kSub[sub][0],
                                  kSub[sub][1], &av2_warped_filter[0][0],
                                  err) != 0) {
            fprintf(stderr, "avm_gpu warp: %s\n", err);
            return 1;
          }
          int diff = 0;
          for (int y = 0; y < REF_H; y++) {
            for (int x = 0; x < PRED_STRIDE; x++) {
              if (cpu[y * PRED_STRIDE + x] != gpu[y * PRED_STRIDE + x]) diff++;
            }
          }
          cases++;
          if (diff) {
            failures++;
            if (failures < 8) {
              printf("  chroma mat %zu shear %zu ss %d/%d %dx%d: %d differ\n", m,
                     sh, kSub[sub][0], kSub[sub][1], pw, ph, diff);
            }
          }
        }
      }
    }
  }

  printf("%-8s %6s %s\n", "warp", "cases", failures ? "MISMATCH" : "ok");
  printf("  warp affine: %d cases\n", cases);

  free(ref);
  free(cpu);
  free(gpu);
  avm_gpu_warp_destroy(ctx, warp);
  avm_gpu_ctx_destroy(ctx);
  printf("\n%s\n", failures ? "FAILED" : "ALL PARITY CASES PASSED");
  return failures ? 1 : 0;
}
