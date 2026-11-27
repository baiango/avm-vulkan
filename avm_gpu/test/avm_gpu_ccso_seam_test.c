/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at aomedia.org/license/software-license/bsd-3-c/.  If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * aomedia.org/license/patent-license/.
 */

/*
 * Seam gate for the CCSO call site: runs the hook and
 * ccso_filter_block_hbd_wo_buf_c() over the encoder's own geometry - a luma
 * reference padded by CCSO_PADDING_SIZE, the destination plane with an aligned
 * stride, 32x32 processing units including the ones at the frame edges - and
 * requires bit-identical output. A block the hook declines counts as a failure
 * unless it sits on the first block row or column, where the C runs instead.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config/avm_dsp_rtcd.h"

#include "av2/common/ccso.h"

#include "avm_gpu/vulkan_ccso.h"
#include "avm_gpu/vulkan_gpu.h"
#include "avm_gpu/vulkan_hook_ccso.h"

/* The frame size the encoder actually used in the failing stream (192 wide,
 * luma stride 768), with a shorter height to keep the gate quick. */
#define PIC_W 192
#define PIC_H 64
#define EXT_STRIDE (PIC_W + 2 * CCSO_PADDING_SIZE)
#define EXT_ROWS (PIC_H + 2 * CCSO_PADDING_SIZE)
#define DST_STRIDE 768
#define UNIT 32
#define BD 8

void ccso_filter_block_hbd_wo_buf_c(
    const uint16_t *src_y, uint16_t *dst_yuv, const int x, const int y,
    const int pic_width, const int pic_height, int *src_cls,
    const int8_t *offset_buf, const int src_y_stride, const int dst_stride,
    const int y_uv_hscale, const int y_uv_vscale, const int thr,
    const int neg_thr, const int *src_loc, const int max_val,
    const int blk_size_x, const int blk_size_y, const bool isSingleBand,
    const uint8_t shift_bits, const int edge_clf, const uint8_t ccso_bo_only);

static unsigned int rng_state = 0x9e3779b9u;
static unsigned int rnd(void) {
  rng_state = rng_state * 1664525u + 1013904223u;
  return rng_state >> 8;
}

int main(void) {
  static const int kThr[3] = { 8, 16, 32 };
  setenv("AV2_GPU_CCSO", "1", 1);

  char err[AVM_GPU_ERR_LEN];
  avm_gpu_ctx *ctx = NULL;
  if (avm_gpu_ctx_create(&ctx, err) != 0) {
    fprintf(stderr, "avm_gpu: %s\n", err);
    return 77; /* skip: no Vulkan device */
  }
  printf("device: %s\n", avm_gpu_device_name(ctx));

  /* CCSO_PADDING_SIZE rows/columns of padding on every side, which is what the
   * encoder's extended reference carries. */
  uint16_t *src = malloc(sizeof(uint16_t) * EXT_STRIDE * EXT_ROWS);
  uint16_t *dst0 = malloc(sizeof(uint16_t) * DST_STRIDE * PIC_H);
  uint16_t *cpu = malloc(sizeof(uint16_t) * DST_STRIDE * PIC_H);
  uint16_t *gpu = malloc(sizeof(uint16_t) * DST_STRIDE * PIC_H);
  int8_t lut[AVM_GPU_CCSO_LUT_ENTRIES];
  if (!src || !dst0 || !cpu || !gpu) return 1;

  const int max_val = (1 << BD) - 1;
  int cases = 0, failures = 0, on_gpu = 0, declined_ok = 0;

  for (int support = 0; support < 7; support++) {
    int src_loc[2];
    derive_ccso_sample_pos(src_loc, EXT_STRIDE, (uint8_t)support);
    for (int edge = 0; edge < 2; edge++) {
      for (int bo = 0; bo < 2; bo++) {
        for (int single = 0; single < 2; single++) {
          for (size_t t = 0; t < sizeof(kThr) / sizeof(kThr[0]); t++) {
            const int thr = kThr[t];
            /* One frame of data per configuration, so the blocks are correlated
             * the way neighbouring processing units are. */
            for (int i = 0; i < EXT_STRIDE * EXT_ROWS; i++) {
              src[i] = (uint16_t)(rnd() & max_val);
            }
            for (int i = 0; i < DST_STRIDE * PIC_H; i++) {
              dst0[i] = (uint16_t)(rnd() & max_val);
            }
            for (int i = 0; i < AVM_GPU_CCSO_LUT_ENTRIES; i++) {
              lut[i] = (int8_t)((int)(rnd() % 65) - 32);
            }
            /* The encoder advances the reference to the block row and passes
             * absolute x/y, exactly as apply_ccso_filter() does. */
            const uint16_t *src_pad = src + CCSO_PADDING_SIZE * EXT_STRIDE +
                                      CCSO_PADDING_SIZE;
            for (int by = 0; by < PIC_H; by += UNIT) {
              const uint16_t *src_row = src_pad + (size_t)by * EXT_STRIDE;
              for (int bx = 0; bx < PIC_W; bx += UNIT) {
                memcpy(cpu, dst0, sizeof(uint16_t) * DST_STRIDE * PIC_H);
                memcpy(gpu, dst0, sizeof(uint16_t) * DST_STRIDE * PIC_H);
                const uint16_t *srow = src_row;
                uint16_t *crow = cpu + (size_t)by * DST_STRIDE;
                int src_cls[2] = { 0, 0 };
                ccso_filter_block_hbd_wo_buf_c(
                    srow, crow, bx, by, PIC_W, PIC_H, src_cls, lut, EXT_STRIDE,
                    DST_STRIDE, 0, 0, thr, -thr, src_loc, max_val, UNIT, UNIT,
                    single != 0, (uint8_t)(BD - 4), edge, (uint8_t)bo);
                const int handled = avm_gpu_hook_ccso(
                    srow, gpu + (size_t)by * DST_STRIDE, bx, by, PIC_W, PIC_H,
                    lut, EXT_STRIDE, DST_STRIDE, 0, 0, thr, -thr, src_loc[0],
                    src_loc[1], max_val, UNIT, UNIT, single, BD - 4, edge, bo);
                cases++;

                const int want_cpu = !(bx < 2 || by < 1);
                if (!handled) {
                  if (!want_cpu) {
                    declined_ok++;
                    continue;
                  }
                  failures++;
                  if (failures < 8) {
                    printf("  sup %d clf %d bo %d single %d thr %d at (%d,%d): "
                           "the hook declined a block it has to take\n",
                           support, edge, bo, single, thr, bx, by);
                  }
                  continue;
                }
                on_gpu++;
                int diff = 0, fx = 0, fy = 0;
                for (int y = 0; y < UNIT && by + y < PIC_H; y++) {
                  for (int x = 0; x < UNIT && bx + x < PIC_W; x++) {
                    if (cpu[(size_t)(by + y) * DST_STRIDE + bx + x] !=
                        gpu[(size_t)(by + y) * DST_STRIDE + bx + x]) {
                      if (!diff) {
                        fx = x;
                        fy = y;
                      }
                      diff++;
                    }
                  }
                }
                if (diff) {
                  failures++;
                  if (failures < 8) {
                    printf("  sup %d clf %d bo %d single %d thr %d at (%d,%d): "
                           "%d samples differ, first (%d,%d) gpu %d cpu %d\n",
                           support, edge, bo, single, thr, bx, by, diff, fx, fy,
                           gpu[(size_t)(by + fy) * DST_STRIDE + bx + fx],
                           cpu[(size_t)(by + fy) * DST_STRIDE + bx + fx]);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  /* Chroma: the reference is the (subsampled-by-1) luma plane, the plane is
   * 96x96 with its own aligned stride. */
  {
    enum { CW = 96, CHH = 96, CDST = 384 };
    uint16_t *cdst0 = malloc(sizeof(uint16_t) * CDST * CHH);
    uint16_t *ccpu = malloc(sizeof(uint16_t) * CDST * CHH);
    uint16_t *cgpu = malloc(sizeof(uint16_t) * CDST * CHH);
    int chroma_cases = 0, chroma_fail = 0, chroma_on_gpu = 0;
    for (int support = 0; support < 7; support++) {
      int src_loc[2];
      derive_ccso_sample_pos(src_loc, EXT_STRIDE, (uint8_t)support);
      for (int edge = 0; edge < 2; edge++) {
        for (int bo = 0; bo < 2; bo++) {
          for (int i = 0; i < EXT_STRIDE * EXT_ROWS; i++)
            src[i] = (uint16_t)(rnd() & max_val);
          for (int i = 0; i < CDST * CHH; i++)
            cdst0[i] = (uint16_t)(rnd() & max_val);
          for (int i = 0; i < AVM_GPU_CCSO_LUT_ENTRIES; i++)
            lut[i] = (int8_t)((int)(rnd() % 65) - 32);
          /* apply_ccso_filter() advances the reference by the block row in
           * luma rows: vscale steps of the padded luma plane. */
          const uint16_t *src_pad =
              src + CCSO_PADDING_SIZE * EXT_STRIDE + CCSO_PADDING_SIZE;
          for (int by = 0; by < CHH; by += UNIT) {
            const uint16_t *src_row = src_pad + (size_t)(by << 1) * EXT_STRIDE;
            for (int bx = 0; bx < CW; bx += UNIT) {
              memcpy(ccpu, cdst0, sizeof(uint16_t) * CDST * CHH);
              memcpy(cgpu, cdst0, sizeof(uint16_t) * CDST * CHH);
              int src_cls[2] = { 0, 0 };
              const int thr = kThr[1];
              ccso_filter_block_hbd_wo_buf_c(
                  src_row, ccpu + (size_t)by * CDST, bx, by, CW, CHH, src_cls,
                  lut, EXT_STRIDE, CDST, 1, 1, thr, -thr, src_loc, max_val,
                  UNIT, UNIT, 1, (uint8_t)(BD - 4), edge, (uint8_t)bo);
              const int handled = avm_gpu_hook_ccso(
                  src_row, cgpu + (size_t)by * CDST, bx, by, CW, CHH, lut,
                  EXT_STRIDE, CDST, 1, 1, thr, -thr, src_loc[0], src_loc[1],
                  max_val, UNIT, UNIT, 1, BD - 4, edge, bo);
              chroma_cases++;
              const int want = !(bx < 2 || by < 1);
              if (!handled) {
                if (want) {
                  chroma_fail++;
                  if (chroma_fail < 6)
                    printf("  chroma sup %d at (%d,%d): declined\n", support,
                           bx, by);
                }
                continue;
              }
              chroma_on_gpu++;
              int diff = 0, fx = 0, fy = 0;
              for (int y = 0; y < UNIT; y++) {
                for (int x = 0; x < UNIT; x++) {
                  if (ccpu[(size_t)(by + y) * CDST + bx + x] !=
                      cgpu[(size_t)(by + y) * CDST + bx + x]) {
                    if (!diff) {
                      fx = x;
                      fy = y;
                    }
                    diff++;
                  }
                }
              }
              if (diff) {
                chroma_fail++;
                if (chroma_fail < 6) {
                  printf("  chroma sup %d clf %d bo %d at (%d,%d): %d differ, "
                         "first (%d,%d) gpu %d cpu %d\n",
                         support, edge, bo, bx, by, diff, fx, fy,
                         cgpu[(size_t)(by + fy) * CDST + bx + fx],
                         ccpu[(size_t)(by + fy) * CDST + bx + fx]);
                }
              }
            }
          }
        }
      }
    }
    failures += chroma_fail;
    printf("  ccso seam chroma: %d cases, %d on the GPU, %d differ\n",
           chroma_cases, chroma_on_gpu, chroma_fail);
    free(cdst0);
    free(ccpu);
    free(cgpu);
  }

  if (on_gpu == 0) {
    printf("FAIL: the hook never filtered a block - the gate is vacuous\n");
    failures++;
  }
  printf("%-10s %8s %s (%d blocks on the GPU, %d at the left/top edge left to "
         "the C)\n",
         "ccso seam", "cases", failures ? "MISMATCH" : "ok", on_gpu,
         declined_ok);
  printf("  ccso seam: %d cases\n", cases);

  free(src);
  free(dst0);
  free(cpu);
  free(gpu);
  avm_gpu_ctx_destroy(ctx);
  printf("\n%s\n", failures ? "FAILED" : "ALL PARITY CASES PASSED");
  return failures ? 1 : 0;
}
