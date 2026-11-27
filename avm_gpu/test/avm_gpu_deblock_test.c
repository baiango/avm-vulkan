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
 * Parity gate for the Vulkan generic deblock filter against
 * avm_highbd_lpf_{horizontal,vertical}_generic_c.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config/avm_dsp_rtcd.h"

#include "avm_gpu/vulkan_gpu.h"
#include "avm_gpu/vulkan_deblock.h"

#define SIDE 48
#define STRIDE SIDE
#define BD_10 10
#define BD_8 8

static unsigned int rng_state = 0x2545f491u;
static unsigned int rnd(void) {
  rng_state = rng_state * 1664525u + 1013904223u;
  return rng_state >> 8;
}

int main(void) {
  static const int kWidths[8][2] = { { 4, 4 },  { 8, 8 },   { 4, 8 },
                                     { 8, 10 }, { 10, 14 }, { 14, 18 },
                                     { 18, 18 }, { 0, 4 } };
  static const int kLL[3][2] = { { 0, 0 }, { 1, 0 }, { 0, 1 } };

  char err[AVM_GPU_ERR_LEN];
  avm_gpu_ctx *ctx = NULL;
  if (avm_gpu_ctx_create(&ctx, err) != 0) {
    fprintf(stderr, "avm_gpu: %s\n", err);
    return 77; /* skip: no Vulkan device */
  }
  avm_gpu_deblock *db = NULL;
  if (avm_gpu_deblock_create(ctx, &db, err) != 0) {
    fprintf(stderr, "avm_gpu deblock: %s\n", err);
    avm_gpu_ctx_destroy(ctx);
    return 1;
  }
  printf("device: %s\n", avm_gpu_device_name(ctx));

  uint16_t *plane = malloc(sizeof(uint16_t) * SIDE * SIDE);
  uint16_t *cpu = malloc(sizeof(uint16_t) * SIDE * SIDE);
  uint16_t *gpu = malloc(sizeof(uint16_t) * SIDE * SIDE);
  if (!plane || !cpu || !gpu) return 1;

  const int x = 20, y = 20;
  int cases = 0, failures = 0, filtered = 0;

  /* Every (q, side) the encoder realistically sends, plus a 16-bit thorough
   * sweep: a small q is what exposes the threshold comparisons, and the earlier
   * sweep at q >= 8 missed two kernel defects. */
  enum { MAX_PAIRS = 4096 };
  static int pairs[MAX_PAIRS][2];
  int npairs = 0;
  for (int qv = 1; qv <= 16; qv++) {
    for (int sv = 1; sv <= 16; sv++) {
      pairs[npairs][0] = qv;
      pairs[npairs][1] = sv;
      npairs++;
    }
  }
  for (int i = 0; i < 1024; i++) {
    pairs[npairs][0] = 1 + (int)(rnd() % 63);
    pairs[npairs][1] = 1 + (int)(rnd() % 63);
    npairs++;
  }

  for (size_t w = 0; w < sizeof(kWidths) / sizeof(kWidths[0]); w++) {
    for (int pi = 0; pi < npairs; pi++) {
      for (size_t l = 0; l < 3; l++) {
        for (int bidx = 0; bidx < 2; bidx++) {
        const int BD = bidx ? BD_8 : BD_10;
        for (int vertical = 0; vertical < 2; vertical++) {
          for (int yy = 0; yy < SIDE; yy++) {
            for (int xx = 0; xx < SIDE; xx++) {
              const int side = vertical ? (xx < x) : (yy < y);
              plane[yy * SIDE + xx] =
                  (uint16_t)(512 + (side ? 0 : 5) + (int)(rnd() % 3));
            }
          }
          memcpy(cpu, plane, sizeof(uint16_t) * SIDE * SIDE);
          const uint16_t qv = (uint16_t)pairs[pi][0];
          const uint16_t sv = (uint16_t)pairs[pi][1];
          if (vertical) {
            avm_highbd_lpf_vertical_generic_c(
                cpu + y * STRIDE + x, STRIDE, kWidths[w][0], kWidths[w][1], &qv,
                &sv, BD, kLL[l][0], kLL[l][1]);
          } else {
            avm_highbd_lpf_horizontal_generic_c(
                cpu + y * STRIDE + x, STRIDE, kWidths[w][0], kWidths[w][1], &qv,
                &sv, BD, kLL[l][0], kLL[l][1]);
          }

          if (avm_gpu_deblock_generic(ctx, db, plane, STRIDE, SIDE, x, y,
                                      kWidths[w][0], kWidths[w][1], qv, sv, BD,
                                      kLL[l][0], kLL[l][1], vertical, gpu,
                                      err) != 0) {
            fprintf(stderr, "avm_gpu deblock: %s\n", err);
            return 1;
          }

          int diff = 0, touched = 0;
          for (int i = 0; i < SIDE * SIDE; i++) {
            if (cpu[i] != gpu[i]) diff++;
            if (cpu[i] != plane[i]) touched++;
          }
          cases++;
          /* Random samples make filt_choice() return 0 almost always, so an
           * earlier revision of this gate passed without filtering anything
           * (a deliberate kernel control went unnoticed). The counter below
           * makes a vacuous sweep a failure. */
          filtered += touched != 0;
          if (diff) {
            failures++;
            if (failures < 12) {
              printf("  %s bd%d w(%d,%d) q%d s%d ll(%d,%d): %d samples differ\n",
                     vertical ? "vertical" : "horizontal", BD, kWidths[w][0],
                     kWidths[w][1], qv, sv, kLL[l][0], kLL[l][1], diff);
            }
          }
        }
        }
      }
    }
  }

  /* The encode path uses the windowed entry point: same input, only the
   * filter's reach is uploaded. It must agree with the whole-plane one. */
  int wcases = 0, wfail = 0;
  for (size_t w = 0; w < sizeof(kWidths) / sizeof(kWidths[0]); w++) {
    for (size_t l = 0; l < 3; l++) {
      for (int vertical = 0; vertical < 2; vertical++) {
        for (int yy = 0; yy < SIDE; yy++) {
          for (int xx = 0; xx < SIDE; xx++) {
            const int side = vertical ? (xx < x) : (yy < y);
            plane[yy * SIDE + xx] =
                (uint16_t)(512 + (side ? 0 : 5) + (int)(rnd() % 3));
          }
        }
        memcpy(cpu, plane, sizeof(uint16_t) * SIDE * SIDE);
        memcpy(gpu, plane, sizeof(uint16_t) * SIDE * SIDE);
        const int qv = 4 + (int)(rnd() % 8);
        const int sv = 4 + (int)(rnd() % 12);
        const uint16_t q16 = (uint16_t)qv, s16 = (uint16_t)sv;
        if (vertical) {
          avm_highbd_lpf_vertical_generic_c(cpu + y * STRIDE + x, STRIDE,
                                            kWidths[w][0], kWidths[w][1], &q16,
                                            &s16, BD_8, kLL[l][0], kLL[l][1]);
        } else {
          avm_highbd_lpf_horizontal_generic_c(
              cpu + y * STRIDE + x, STRIDE, kWidths[w][0], kWidths[w][1], &q16,
              &s16, BD_8, kLL[l][0], kLL[l][1]);
        }
        if (avm_gpu_deblock_window(ctx, db, plane + y * STRIDE + x,
                                   vertical ? 1 : STRIDE,
                                   vertical ? STRIDE : 1, kWidths[w][0],
                                   kWidths[w][1], qv, sv, BD_8, kLL[l][0],
                                   kLL[l][1], gpu + y * STRIDE + x, err) != 0) {
          fprintf(stderr, "avm_gpu deblock window: %s\n", err);
          return 1;
        }
        int diff = 0;
        for (int i = 0; i < SIDE * SIDE; i++) diff += cpu[i] != gpu[i];
        wcases++;
        if (diff) {
          wfail++;
          if (wfail < 6) {
            printf("  window %s w(%d,%d) q%d s%d ll(%d,%d): %d differ\n",
                   vertical ? "vertical" : "horizontal", kWidths[w][0],
                   kWidths[w][1], qv, sv, kLL[l][0], kLL[l][1], diff);
          }
        }
      }
    }
  }
  failures += wfail;
  printf("  window: %d cases, %d differ\n", wcases, wfail);

  printf("%-10s %8s %s\n", "filter", "cases", failures ? "MISMATCH" : "ok");
  printf("  generic deblock: %d cases, %d filtered\n", cases, filtered);

  free(plane);
  free(cpu);
  free(gpu);
  avm_gpu_deblock_destroy(ctx, db);
  avm_gpu_ctx_destroy(ctx);
  printf("\n%s\n", failures ? "FAILED" : "ALL PARITY CASES PASSED");
  return failures ? 1 : 0;
}
