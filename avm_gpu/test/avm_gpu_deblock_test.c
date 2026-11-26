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
#define BD 10

static unsigned int rng_state = 0x2545f491u;
static unsigned int rnd(void) {
  rng_state = rng_state * 1664525u + 1013904223u;
  return rng_state >> 8;
}

int main(void) {
  static const int kWidths[7][2] = { { 4, 4 },  { 8, 8 },   { 4, 8 },
                                     { 8, 10 }, { 10, 14 }, { 14, 18 },
                                     { 0, 4 } };
  static const int kQ[3] = { 8, 24, 63 };
  static const int kSide[3] = { 4, 16, 40 };
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
  int cases = 0, failures = 0;

  for (size_t w = 0; w < sizeof(kWidths) / sizeof(kWidths[0]); w++) {
    for (size_t q = 0; q < 3; q++) {
      for (size_t s = 0; s < 3; s++) {
        for (size_t l = 0; l < 3; l++) {
          for (int vertical = 0; vertical < 2; vertical++) {
            for (int i = 0; i < SIDE * SIDE; i++) {
              plane[i] = (uint16_t)(rnd() & 0x3ff);
            }
            memcpy(cpu, plane, sizeof(uint16_t) * SIDE * SIDE);
            const uint16_t qv = (uint16_t)kQ[q];
            const uint16_t sv = (uint16_t)kSide[s];
            if (vertical) {
              avm_highbd_lpf_vertical_generic_c(cpu + y * STRIDE + x, 1,
                                                kWidths[w][0], kWidths[w][1],
                                                &qv, &sv, BD, kLL[l][0],
                                                kLL[l][1]);
            } else {
              avm_highbd_lpf_horizontal_generic_c(
                  cpu + y * STRIDE + x, STRIDE, kWidths[w][0], kWidths[w][1],
                  &qv, &sv, BD, kLL[l][0], kLL[l][1]);
            }

            if (avm_gpu_deblock_generic(ctx, db, plane, STRIDE, SIDE, x, y,
                                        kWidths[w][0], kWidths[w][1], kQ[q],
                                        kSide[s], BD, kLL[l][0], kLL[l][1],
                                        vertical, gpu, err) != 0) {
              fprintf(stderr, "avm_gpu deblock: %s\n", err);
              return 1;
            }

            int diff = 0;
            for (int i = 0; i < SIDE * SIDE; i++) {
              if (cpu[i] != gpu[i]) diff++;
            }
            cases++;
            if (diff) {
              failures++;
              if (failures < 6) {
                printf(
                    "  %s w(%d,%d) q%d s%d ll(%d,%d): %d samples differ\n",
                    vertical ? "vertical" : "horizontal", kWidths[w][0],
                    kWidths[w][1], kQ[q], kSide[s], kLL[l][0], kLL[l][1], diff);
              }
            }
          }
        }
      }
    }
  }

  printf("%-10s %8s %s\n", "filter", "cases", failures ? "MISMATCH" : "ok");
  printf("  generic deblock: %d cases\n", cases);

  free(plane);
  free(cpu);
  free(gpu);
  avm_gpu_deblock_destroy(ctx, db);
  avm_gpu_ctx_destroy(ctx);
  printf("\n%s\n", failures ? "FAILED" : "ALL PARITY CASES PASSED");
  return failures ? 1 : 0;
}
