/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at aomedia.org/license/patent-license/.
 */

/*
 * Parity gate for the Vulkan CCSO block pass against
 * ccso_filter_block_hbd_wo_buf_c() from av2/common/ccso.c.
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "av2/common/ccso.h"

#include "avm_gpu/vulkan_gpu.h"
#include "avm_gpu/vulkan_ccso.h"

#define PIC_W 64
#define PIC_H 64
#define BD 10
/* AVM derives shift_bits = bit_depth - max_band_log2: 16 bands with
 * max_band_log2 = 4, up to 64 bands (CCSO_BAND_NUM) with max_band_log2 = 6. */
static const int kShiftBits[2] = { BD - 4, BD - 6 };

/* av2/common/ccso.h declares derive_ccso_sample_pos() but not the block
 * filter itself. */
void ccso_filter_block_hbd_wo_buf_c(
    const uint16_t *src_y, uint16_t *dst_yuv, const int x, const int y,
    const int pic_width, const int pic_height, int *src_cls,
    const int8_t *offset_buf, const int src_y_stride, const int dst_stride,
    const int y_uv_hscale, const int y_uv_vscale, const int thr,
    const int neg_thr, const int *src_loc, const int max_val,
    const int blk_size_x, const int blk_size_y, const bool isSingleBand,
    const uint8_t shift_bits, const int edge_clf, const uint8_t ccso_bo_only);

static unsigned int rng_state = 0x3c6ef372u;
static unsigned int rnd(void) {
  rng_state = rng_state * 1664525u + 1013904223u;
  return rng_state >> 8;
}

/* The gates call AVM's own reference functions, several of which are RTCD
 * pointers (NULL on x86 until the tables are set up - arm64 resolves them to C
 * at link time, which is why MoltenVK never showed this). The encoder calls
 * these at init; the gates do the same. */
void av2_rtcd(void);
void avm_dsp_rtcd(void);

int main(void) {
  av2_rtcd();
  avm_dsp_rtcd();
  static const int kThr[3][2] = { { 8, -8 }, { 16, -4 }, { 4, -16 } };
  static const int kBlk[3] = { 4, 8, 16 };

  char err[AVM_GPU_ERR_LEN];
  avm_gpu_ctx *ctx = NULL;
  if (avm_gpu_ctx_create(&ctx, err) != 0) {
    fprintf(stderr, "avm_gpu: %s\n", err);
    return 77; /* skip: no Vulkan device */
  }
  avm_gpu_ccso *cc = NULL;
  if (avm_gpu_ccso_create(ctx, &cc, err) != 0) {
    fprintf(stderr, "avm_gpu ccso: %s\n", err);
    avm_gpu_ctx_destroy(ctx);
    return 1;
  }
  printf("device: %s\n", avm_gpu_device_name(ctx));

  uint16_t *src = malloc(sizeof(uint16_t) * PIC_W * PIC_H);
  uint16_t *dst0 = malloc(sizeof(uint16_t) * PIC_W * PIC_H);
  uint16_t *cpu = malloc(sizeof(uint16_t) * PIC_W * PIC_H);
  uint16_t *gpu = malloc(sizeof(uint16_t) * PIC_W * PIC_H);
  int8_t lut[AVM_GPU_CCSO_LUT_ENTRIES];
  if (!src || !dst0 || !cpu || !gpu) return 1;

  const int max_val = (1 << BD) - 1;
  const int x = 8, y = 8;
  int cases = 0, failures = 0;

  for (int sb = 0; sb < 2; sb++) {
  const int shift_bits = kShiftBits[sb];
  for (int support = 0; support < 7; support++) {
    for (int edge = 0; edge < 2; edge++) {
      for (size_t t = 0; t < 3; t++) {
        for (int single = 0; single < 2; single++) {
          for (int bo = 0; bo < 2; bo++) {
            for (size_t b = 0; b < 3; b++) {
              for (int i = 0; i < PIC_W * PIC_H; i++) {
                src[i] = (uint16_t)(rnd() & (max_val));
                dst0[i] = (uint16_t)(rnd() & (max_val));
              }
              for (int i = 0; i < AVM_GPU_CCSO_LUT_ENTRIES; i++) {
                lut[i] = (int8_t)((int)(rnd() % 65) - 32);
              }
              memcpy(cpu, dst0, sizeof(uint16_t) * PIC_W * PIC_H);

              int src_loc[2];
              derive_ccso_sample_pos(src_loc, PIC_W, (uint8_t)support);

              int src_cls[2] = { 0, 0 };
              /* AVM hands this helper row-window pointers, so the reference
               * gets the planes advanced to the block origin. */
              ccso_filter_block_hbd_wo_buf_c(
                  src + y * PIC_W, cpu + y * PIC_W, x, y, PIC_W, PIC_H,
                  src_cls, lut, PIC_W, PIC_W, 0, 0, kThr[t][0], kThr[t][1],
                  src_loc, max_val, kBlk[b], kBlk[b], single != 0,
                  (uint8_t)shift_bits, edge, (uint8_t)bo);

              if (avm_gpu_ccso_block(ctx, cc, src, PIC_W, dst0, PIC_W, PIC_W,
                                     PIC_H, x, y, lut, src_loc[0], src_loc[1],
                                     0, 0, kThr[t][0], kThr[t][1], max_val,
                                     kBlk[b], kBlk[b], single, shift_bits,
                                     edge, bo, gpu, err) != 0) {
                fprintf(stderr, "avm_gpu ccso: %s\n", err);
                return 1;
              }

              int diff = 0;
              for (int i = 0; i < PIC_W * PIC_H; i++) {
                if (cpu[i] != gpu[i]) diff++;
              }
              cases++;
              if (diff) {
                failures++;
                if (failures < 6) {
                  printf("  support %d edge %d thr %d single %d bo %d blk %d: "
                         "%d samples differ\n",
                         support, edge, kThr[t][0], single, bo, kBlk[b], diff);
                }
              }
            }
          }
        }
      }
    }
  }
  }

  printf("%-8s %8s %s\n", "ccso", "cases", failures ? "MISMATCH" : "ok");
  printf("  ccso block: %d cases\n", cases);

  free(src);
  free(dst0);
  free(cpu);
  free(gpu);
  avm_gpu_ccso_destroy(ctx, cc);
  avm_gpu_ctx_destroy(ctx);
  printf("\n%s\n", failures ? "FAILED" : "ALL PARITY CASES PASSED");
  return failures ? 1 : 0;
}
