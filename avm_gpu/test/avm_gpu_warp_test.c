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
#include "av2/common/scale.h"
#include "avm_dsp/avm_dsp_common.h"
#include "av2/common/warped_motion.h"

/* warped_motion.h declares the dispatch wrapper, not the _c kernel. */
void av2_highbd_warp_affine_c(const int32_t *mat, const uint16_t *ref,
                              int width, int height, int stride,
                              uint16_t *pred, int p_col, int p_row, int p_width,
                              int p_height, int p_stride, int subsampling_x,
                              int subsampling_y, int bd,
                              ConvolveParams *conv_params, int16_t alpha,
                              int16_t beta, int16_t gamma, int16_t delta);

void av2_ext_highbd_warp_affine_c(const int32_t *mat, const uint16_t *ref,
                                  int width, int height, int stride,
                                  uint16_t *pred, int p_col, int p_row,
                                  int p_width, int p_height, int p_stride,
                                  int subsampling_x, int subsampling_y, int bd,
                                  ConvolveParams *conv_params,
                                  int use_warp_bd_box, PadBlock *warp_bd_box);

/* The scaled variant is not a dispatch wrapper; warped_motion.c exports it. */
void av2_ext_highbd_warp_affine_scaled_c(
    const int32_t *mat, const uint16_t *ref, int width, int height, int stride,
    uint16_t *pred, int p_col, int p_row, int p_width, int p_height,
    int p_stride, int subsampling_x, int subsampling_y, int bd,
    ConvolveParams *conv_params, const struct scale_factors *sf,
    int use_warp_bd_box, PadBlock *warp_bd_box);

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
                                cp.round_0, cp.round_1, kShear[sh][0],
                                kShear[sh][1], kShear[sh][2], kShear[sh][3], 0,
                                0, &av2_warped_filter[0][0], NULL, err) != 0) {
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
                                  cp.round_0, cp.round_1, kShear[sh][0],
                                  kShear[sh][1], kShear[sh][2], kShear[sh][3],
                                  kSub[sub][0], kSub[sub][1],
                                  &av2_warped_filter[0][0], NULL, err) != 0) {
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

  /* Extended-range warp: 4x4 chunks and the 6-tap ext filter. */
  static const int kExtSize[4][2] = { { 4, 4 }, { 8, 8 }, { 8, 4 }, { 16, 16 } };
  for (size_t m = 0; m < 3; m++) {
    for (size_t sub = 0; sub < 3; sub++) {
      for (size_t s = 0; s < 4; s++) {
        const int pw = kExtSize[s][0];
        const int ph = kExtSize[s][1];
        memset(cpu, 0, sizeof(uint16_t) * PRED_STRIDE * REF_H);
        memset(gpu, 0, sizeof(uint16_t) * PRED_STRIDE * REF_H);
        av2_ext_highbd_warp_affine_c(kMat[m], ref, REF_W, REF_H, REF_W, cpu,
                                     p_col, p_row, pw, ph, PRED_STRIDE,
                                     kSub[sub][0], kSub[sub][1], BD, &cp, 0,
                                     NULL);
        if (avm_gpu_warp_affine_ext(ctx, warp, kMat[m], ref, REF_W, REF_H,
                                    REF_W, gpu, p_col, p_row, pw, ph,
                                    PRED_STRIDE, BD, cp.round_0, cp.round_1,
                                    kSub[sub][0], kSub[sub][1],
                                    &av2_ext_warped_filter[0][0], NULL, 0,
                                    NULL, err) != 0) {
          fprintf(stderr, "avm_gpu warp ext: %s\n", err);
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
            printf("  warp-ext mat %zu ss %d/%d %dx%d: %d differ\n", m,
                   kSub[sub][0], kSub[sub][1], pw, ph, diff);
          }
        }
      }
    }
  }

  /* Scaled (resized-reference) warp: av2_ext_highbd_warp_affine_scaled_c()
   * with the scale factors of a resized reference and a boundary box. */
  {
    static const int kRefDim[3][2] = { { 96, 96 }, { 64, 64 }, { 80, 64 } };
    static const int kCurDim[3][2] = { { 96, 96 }, { 96, 96 }, { 48, 96 } };
    static const int kScaleSize[4][2] = {
      { 4, 4 }, { 8, 8 }, { 8, 4 }, { 16, 16 },
    };
    static int32_t box[32 * 32 * 4];
    for (size_t d = 0; d < 3; d++) {
      struct scale_factors sf;
      av2_setup_scale_factors_for_frame(&sf, kRefDim[d][0], kRefDim[d][1],
                                        kCurDim[d][0], kCurDim[d][1]);
      const int rw = kRefDim[d][0];
      const int rh = kRefDim[d][1];
      /* Each 8x8 sub-block gets its own limits, as
       * av2_get_reference_area_with_padding_single() would produce: the
       * sub-block's area moved by a few pixels and clipped to the frame. */
      for (int by = 0; by < 32; by++) {
        for (int bx = 0; bx < 32; bx++) {
          const int i = by * 32 + bx;
          const int mvx = (int)(rnd() % 33) - 16;
          const int mvy = (int)(rnd() % 33) - 16;
          const int x0 = AVMMAX(0, bx * 8 + mvx - 8);
          const int y0 = AVMMAX(0, by * 8 + mvy - 8);
          const int x1 = AVMMIN(rw, bx * 8 + mvx + 16);
          const int y1 = AVMMIN(rh, by * 8 + mvy + 16);
          box[i * 4 + 0] = x0;
          box[i * 4 + 1] = AVMMAX(x0 + 1, x1);
          box[i * 4 + 2] = y0;
          box[i * 4 + 3] = AVMMAX(y0 + 1, y1);
        }
      }
      for (size_t m = 0; m < 3; m++) {
        for (size_t sub = 0; sub < 3; sub++) {
          for (size_t s = 0; s < 4; s++) {
            const int pw = kScaleSize[s][0];
            const int ph = kScaleSize[s][1];
            memset(cpu, 0, sizeof(uint16_t) * PRED_STRIDE * REF_H);
            memset(gpu, 0, sizeof(uint16_t) * PRED_STRIDE * REF_H);
            av2_ext_highbd_warp_affine_scaled_c(
                kMat[m], ref, rw, rh, REF_W, cpu, p_col, p_row, pw, ph,
                PRED_STRIDE, kSub[sub][0], kSub[sub][1], BD, &cp, &sf, 1,
                (PadBlock *)box);
            if (avm_gpu_warp_affine_scaled(
                    ctx, warp, kMat[m], ref, rw, rh, REF_W, gpu, p_col, p_row,
                    pw, ph, PRED_STRIDE, BD, cp.round_0, cp.round_1,
                    kSub[sub][0], kSub[sub][1], sf.x_scale_fp, sf.y_scale_fp,
                    sf.x_step_q4, sf.y_step_q4, box,
                    &av2_ext_warped_filter[0][0], NULL, err) != 0) {
              fprintf(stderr, "avm_gpu warp scaled: %s\n", err);
              return 1;
            }
            int diff = 0;
            for (int y = 0; y < REF_H; y++) {
              for (int x = 0; x < PRED_STRIDE; x++) {
                if (cpu[y * PRED_STRIDE + x] != gpu[y * PRED_STRIDE + x])
                  diff++;
              }
            }
            cases++;
            if (diff) {
              failures++;
              if (failures < 8) {
                printf("  warp-scaled ref %dx%d cur %dx%d mat %zu ss %d/%d "
                       "%dx%d: %d differ\n",
                       rw, rh, kCurDim[d][0], kCurDim[d][1], m, kSub[sub][0],
                       kSub[sub][1], pw, ph, diff);
              }
            }
          }
        }
      }
    }
  }

  /* Scaled warp regression: a reference of 64x64 with a wide stride, a 16x16
   * block whose rotated projection puts the later chunks' rescaled corners at
   * negative y, and a boundary box that clips nearly every read. This is the
   * case a resized-reference stream failed on: the 64x32 multiply of the
   * projection sign-extended the low word, so those chunks' position was
   * corrupted. */
  {
    const int rw = 64, rh = 64, st = 1152;
    static const int32_t kRegMat[2][6] = {
      { 79296, 12224, 73920, -256, -512, 65472 },
      { 79296, 12224, 73920, 0, -512, 65472 },
    };
    uint16_t *plane = calloc((size_t)st * rh, sizeof(*plane));
    uint16_t *rc = calloc((size_t)st * 32, sizeof(*rc));
    uint16_t *rg = calloc((size_t)st * 32, sizeof(*rg));
    static int32_t rbox[32 * 32 * 4];
    struct scale_factors rsf;
    if (!plane || !rc || !rg) return 1;
    av2_setup_scale_factors_for_frame(&rsf, rw, rh, 57, 57);
    for (int y = 0; y < rh; y++) {
      for (int x = 0; x < rw; x++) {
        plane[(size_t)y * st + x] = (uint16_t)(rnd() & 0x3ff);
      }
    }
    for (int i = 0; i < 32 * 32; i++) {
      rbox[i * 4 + 0] = 29;
      rbox[i * 4 + 1] = 45;
      rbox[i * 4 + 2] = 0;
      rbox[i * 4 + 3] = 12;
    }
    for (size_t m = 0; m < 2; m++) {
      memset(rc, 0, (size_t)st * 32 * sizeof(*rc));
      memset(rg, 0, (size_t)st * 32 * sizeof(*rg));
      av2_ext_highbd_warp_affine_scaled_c(
          kRegMat[m], plane, rw, rh, st, rc, 16, 0, 16, 16, st, 0, 0, BD, &cp,
          &rsf, 1, (PadBlock *)rbox);
      if (avm_gpu_warp_affine_scaled(
              ctx, warp, kRegMat[m], plane, rw, rh, st, rg, 16, 0, 16, 16, st,
              BD, cp.round_0, cp.round_1, 0, 0, rsf.x_scale_fp, rsf.y_scale_fp,
              rsf.x_step_q4, rsf.y_step_q4, rbox,
              &av2_ext_warped_filter[0][0], NULL, err) != 0) {
        fprintf(stderr, "avm_gpu warp scaled regression: %s\n", err);
        return 1;
      }
      int diff = 0;
      for (int y = 0; y < 16; y++) {
        for (int x = 0; x < 16; x++) {
          if (rc[(size_t)y * st + x] != rg[(size_t)y * st + x]) diff++;
        }
      }
      cases++;
      failures += diff != 0;
      if (diff) {
        printf("  warp-scaled regression mat %zu: %d differ\n", m, diff);
      }
    }
    free(plane);
    free(rc);
    free(rg);
  }

  /* Compound (two-reference) warp: the first call writes the high-precision
   * buffer, the second averages it and writes the pixels. Checked against all
   * three C variants with conv_params->is_compound, plain and weighted. */
  {
    static const int kCmpSize[3][2] = { { 8, 8 }, { 16, 8 }, { 16, 16 } };
    static uint16_t conv_cpu[PRED_STRIDE * REF_H], conv_gpu[PRED_STRIDE * REF_H];
    /* The two references must differ, otherwise the weighted and the plain
     * average agree and a broken weighting would hide. */
    uint16_t *ref2 = malloc(sizeof(uint16_t) * REF_W * REF_H);
    if (!ref2) return 1;
    for (int i = 0; i < REF_W * REF_H; i++) {
      ref2[i] = (uint16_t)(rnd() & ((1 << BD) - 1));
    }
    for (size_t m = 0; m < 2; m++) {         /* identity-ish and sheared */
      for (size_t s = 0; s < 3; s++) {       /* block sizes */
        for (int avg = 0; avg < 2; avg++) {  /* plain / weighted */
          for (int variant = 0; variant < 3; variant++) { /* affine/ext/scaled */
            const int pw = kCmpSize[s][0], ph = kCmpSize[s][1];
            ConvolveParams cc =
                get_conv_params_no_round(0, 0, conv_cpu, PRED_STRIDE, 1, BD);
            ConvolveParams cg =
                get_conv_params_no_round(0, 0, conv_gpu, PRED_STRIDE, 1, BD);
            if (avg) {
              cc.fwd_offset = cg.fwd_offset = 10;
              cc.bck_offset = cg.bck_offset = 6;
            }
            avm_gpu_warp_compound cmp_gpu;
            memset(&cmp_gpu, 0, sizeof(cmp_gpu));
            cmp_gpu.conv = conv_gpu;
            cmp_gpu.conv_stride = PRED_STRIDE;
            cmp_gpu.uneven = avg;
            cmp_gpu.fwd = cg.fwd_offset;
            cmp_gpu.bck = cg.bck_offset;
            struct scale_factors sf;
            av2_setup_scale_factors_for_frame(&sf, REF_W, REF_H, 80, 72);
            static int32_t cbox[32 * 32 * 4];
            for (int i = 0; i < 32 * 32; i++) {
              cbox[i * 4 + 0] = 0;
              cbox[i * 4 + 1] = REF_W;
              cbox[i * 4 + 2] = 0;
              cbox[i * 4 + 3] = REF_H;
            }
            int diff = 0;
            /* Pass 0 is the first reference (writes the high-precision
             * buffer), pass 1 the second (averages it with its own sum). */
            memset(conv_cpu, 0, sizeof(conv_cpu));
            memset(conv_gpu, 0, sizeof(conv_gpu));
            for (int pass = 0; pass < 2; pass++) {
              const uint16_t *r = pass ? ref2 : ref;
              memset(cpu, 0, sizeof(uint16_t) * PRED_STRIDE * REF_H);
              memset(gpu, 0, sizeof(uint16_t) * PRED_STRIDE * REF_H);
              cc.do_average = pass;
              cg.do_average = pass;
              cmp_gpu.do_average = pass;
              if (variant == 0) {
                av2_highbd_warp_affine_c(
                    kMat[m], r, REF_W, REF_H, REF_W, cpu, p_col, p_row, pw,
                    ph, PRED_STRIDE, 0, 0, BD, &cc, (int16_t)kShear[1][0],
                    (int16_t)kShear[1][1], (int16_t)kShear[1][2],
                    (int16_t)kShear[1][3]);
                if (avm_gpu_warp_affine(
                        ctx, warp, kMat[m], r, REF_W, REF_H, REF_W, gpu,
                        p_col, p_row, pw, ph, PRED_STRIDE, BD, cg.round_0,
                        cg.round_1, kShear[1][0], kShear[1][1],
                        kShear[1][2], kShear[1][3], 0, 0,
                        &av2_warped_filter[0][0], &cmp_gpu, err) != 0) {
                  fprintf(stderr, "avm_gpu warp compound: %s\n", err);
                  return 1;
                }
              } else if (variant == 1) {
                av2_ext_highbd_warp_affine_c(
                    kMat[m], r, REF_W, REF_H, REF_W, cpu, p_col, p_row, pw,
                    ph, PRED_STRIDE, 0, 0, BD, &cc, 0, NULL);
                if (avm_gpu_warp_affine_ext(
                        ctx, warp, kMat[m], r, REF_W, REF_H, REF_W, gpu,
                        p_col, p_row, pw, ph, PRED_STRIDE, BD, cg.round_0,
                        cg.round_1, 0, 0, &av2_ext_warped_filter[0][0], NULL, 0,
                        &cmp_gpu, err) != 0) {
                  fprintf(stderr, "avm_gpu warp ext compound: %s\n", err);
                  return 1;
                }
              } else {
                av2_ext_highbd_warp_affine_scaled_c(
                    kMat[m], r, REF_W, REF_H, REF_W, cpu, p_col, p_row, pw,
                    ph, PRED_STRIDE, 0, 0, BD, &cc, &sf, 1, (PadBlock *)cbox);
                if (avm_gpu_warp_affine_scaled(
                        ctx, warp, kMat[m], r, REF_W, REF_H, REF_W, gpu,
                        p_col, p_row, pw, ph, PRED_STRIDE, BD, cg.round_0,
                        cg.round_1, 0, 0, sf.x_scale_fp, sf.y_scale_fp,
                        sf.x_step_q4, sf.y_step_q4, cbox,
                        &av2_ext_warped_filter[0][0], &cmp_gpu, err) != 0) {
                  fprintf(stderr, "avm_gpu warp scaled compound: %s\n", err);
                  return 1;
                }
              }
              for (int y = 0; y < ph; y++) {
                for (int x = 0; x < pw; x++) {
                  if (conv_cpu[y * PRED_STRIDE + x] !=
                      conv_gpu[y * PRED_STRIDE + x]) {
                    diff++;
                  }
                  if (cpu[y * PRED_STRIDE + x] != gpu[y * PRED_STRIDE + x]) {
                    diff++;
                  }
                }
              }
            }
            cases++;
            failures += diff != 0;
            if (diff) {
              printf("  warp compound mat %zu %dx%d avg %d variant %d: %d "
                     "samples\n",
                     m, pw, ph, avg, variant, diff);
            }
          }
        }
      }
    }
    free(ref2);
  }

  /* Extended-range warp with a warp boundary box (the encoder always sets one
   * on that path): single reference and compound, clipped boxes. */
  {
    static const int kBoxSize[4][2] = { { 4, 4 }, { 8, 8 }, { 12, 8 }, { 16, 16 } };
    static uint16_t conv_cpu[PRED_STRIDE * REF_H], conv_gpu[PRED_STRIDE * REF_H];
    static int32_t box[32 * 32 * 4];
    uint16_t *ref2 = malloc(sizeof(uint16_t) * REF_W * REF_H);
    if (!ref2) return 1;
    for (int i = 0; i < REF_W * REF_H; i++) {
      ref2[i] = (uint16_t)(rnd() & ((1 << BD) - 1));
    }
    for (size_t m = 0; m < 3; m++) {
      for (size_t s = 0; s < 4; s++) {
        for (int compound = 0; compound < 2; compound++) {
          for (int avg = 0; avg < 2; avg++) {
            const int pw = kBoxSize[s][0], ph = kBoxSize[s][1];
            /* Each 8x8 sub-block gets its own limits: the sub-block's area
             * moved by a few pixels and clipped, as
             * av2_get_reference_area_with_padding_single() produces. */
            for (int by = 0; by < 32; by++) {
              for (int bx = 0; bx < 32; bx++) {
                const int i = by * 32 + bx;
                const int mvx = (int)(rnd() % 17) - 8;
                const int mvy = (int)(rnd() % 17) - 8;
                const int x0 = AVMMAX(0, bx * 8 + mvx - 4);
                const int y0 = AVMMAX(0, by * 8 + mvy - 4);
                const int x1 = AVMMIN(REF_W, bx * 8 + mvx + 12);
                const int y1 = AVMMIN(REF_H, by * 8 + mvy + 12);
                box[i * 4 + 0] = x0;
                box[i * 4 + 1] = AVMMAX(x0 + 1, x1);
                box[i * 4 + 2] = y0;
                box[i * 4 + 3] = AVMMAX(y0 + 1, y1);
              }
            }
            ConvolveParams cc = get_conv_params_no_round(0, 0, conv_cpu,
                                                         PRED_STRIDE, compound,
                                                         BD);
            ConvolveParams cg = get_conv_params_no_round(0, 0, conv_gpu,
                                                         PRED_STRIDE, compound,
                                                         BD);
            if (avg) {
              cc.fwd_offset = cg.fwd_offset = 11;
              cc.bck_offset = cg.bck_offset = 5;
            }
            avm_gpu_warp_compound cmp_gpu;
            memset(&cmp_gpu, 0, sizeof(cmp_gpu));
            cmp_gpu.conv = conv_gpu;
            cmp_gpu.conv_stride = PRED_STRIDE;
            cmp_gpu.uneven = avg;
            cmp_gpu.fwd = cg.fwd_offset;
            cmp_gpu.bck = cg.bck_offset;
            memset(conv_cpu, 0, sizeof(conv_cpu));
            memset(conv_gpu, 0, sizeof(conv_gpu));
            int diff = 0;
            const int passes = compound ? 2 : 1;
            for (int pass = 0; pass < passes; pass++) {
              const uint16_t *r = pass ? ref2 : ref;
              cc.do_average = pass;
              cg.do_average = pass;
              cmp_gpu.do_average = pass;
              memset(cpu, 0, sizeof(uint16_t) * PRED_STRIDE * REF_H);
              memset(gpu, 0, sizeof(uint16_t) * PRED_STRIDE * REF_H);
              av2_ext_highbd_warp_affine_c(
                  kMat[m], r, REF_W, REF_H, REF_W, cpu, p_col, p_row, pw, ph,
                  PRED_STRIDE, 0, 0, BD, &cc, 1, (PadBlock *)box);
              if (avm_gpu_warp_affine_ext(
                      ctx, warp, kMat[m], r, REF_W, REF_H, REF_W, gpu, p_col,
                      p_row, pw, ph, PRED_STRIDE, BD, cg.round_0, cg.round_1,
                      0, 0, &av2_ext_warped_filter[0][0], box, 1,
                      compound ? &cmp_gpu : NULL, err) != 0) {
                fprintf(stderr, "avm_gpu warp ext box: %s\n", err);
                return 1;
              }
              for (int y = 0; y < ph; y++) {
                for (int x = 0; x < pw; x++) {
                  if (conv_cpu[y * PRED_STRIDE + x] !=
                      conv_gpu[y * PRED_STRIDE + x]) {
                    diff++;
                  }
                  if (cpu[y * PRED_STRIDE + x] != gpu[y * PRED_STRIDE + x]) {
                    diff++;
                  }
                }
              }
            }
            cases++;
            failures += diff != 0;
            if (diff) {
              printf("  warp ext box mat %zu %dx%d compound %d avg %d: %d "
                     "samples\n",
                     m, pw, ph, compound, avg, diff);
            }
          }
        }
      }
    }
    free(ref2);
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
