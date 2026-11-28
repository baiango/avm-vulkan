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
 * Parity gate for the Vulkan CDEF block filter against
 * av2_cdef_filter_16_{0,1,2,3}_c.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "av2/common/cdef_block.h"

#include "avm_gpu/vulkan_gpu.h"
#include "avm_gpu/vulkan_cdef.h"

#define BUF_ROWS 24
#define DSTRIDE 16

/* av2/common/cdef_block.h only exports the function-pointer typedef. */
typedef void (*cdef_filter_fn)(uint16_t *, int, const uint16_t *, int, int, int,
                               int, int, int, int, int);
void av2_cdef_filter_16_0_c(uint16_t *const dst16, int dstride,
                            const uint16_t *in, int pri_strength,
                            int sec_strength, int dir, int pri_damping,
                            int sec_damping, int coeff_shift, int block_width,
                            int block_height);
void av2_cdef_filter_16_1_c(uint16_t *const dst16, int dstride,
                            const uint16_t *in, int pri_strength,
                            int sec_strength, int dir, int pri_damping,
                            int sec_damping, int coeff_shift, int block_width,
                            int block_height);
void av2_cdef_filter_16_2_c(uint16_t *const dst16, int dstride,
                            const uint16_t *in, int pri_strength,
                            int sec_strength, int dir, int pri_damping,
                            int sec_damping, int coeff_shift, int block_width,
                            int block_height);
void av2_cdef_filter_16_3_c(uint16_t *const dst16, int dstride,
                            const uint16_t *in, int pri_strength,
                            int sec_strength, int dir, int pri_damping,
                            int sec_damping, int coeff_shift, int block_width,
                            int block_height);

static unsigned int rng_state = 0x68e31da4u;
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
  static const int kStrength[5][2] = { { 4, 2 },  { 15, 4 }, { 63, 8 },
                                       { 0, 4 },  { 8, 0 } };
  static const int kDamp[3] = { 3, 4, 6 };
  static const int kShift[2] = { 0, 2 };
  static const cdef_filter_fn kRef[4] = {
    av2_cdef_filter_16_0_c, av2_cdef_filter_16_1_c, av2_cdef_filter_16_2_c,
    av2_cdef_filter_16_3_c
  };

  char err[AVM_GPU_ERR_LEN];
  avm_gpu_ctx *ctx = NULL;
  if (avm_gpu_ctx_create(&ctx, err) != 0) {
    fprintf(stderr, "avm_gpu: %s\n", err);
    return 77; /* skip: no Vulkan device */
  }
  avm_gpu_cdef *cdef = NULL;
  if (avm_gpu_cdef_create(ctx, &cdef, err) != 0) {
    fprintf(stderr, "avm_gpu cdef: %s\n", err);
    avm_gpu_ctx_destroy(ctx);
    return 1;
  }
  printf("device: %s\n", avm_gpu_device_name(ctx));
  printf("  CDEF_BSTRIDE %d, CDEF_VERY_LARGE %d\n", CDEF_BSTRIDE,
         CDEF_VERY_LARGE);

  const int stride = CDEF_BSTRIDE;
  uint16_t *in = malloc(sizeof(uint16_t) * stride * BUF_ROWS);
  uint16_t *cpu = malloc(sizeof(uint16_t) * DSTRIDE * 8);
  uint16_t *gpu = malloc(sizeof(uint16_t) * DSTRIDE * 8);
  if (!in || !cpu || !gpu) return 1;

  const int i0 = 8, j0 = 16;
  const int w = 8, h = 8;
  int cases = 0, failures = 0;

  for (int v = 0; v < 6; v++) {
    for (int i = 0; i < stride * BUF_ROWS; i++) {
      in[i] = (uint16_t)(rnd() & 0x3ff);
    }
    /* Sprinkle the "very large" sentinel so the min/max rules are exercised. */
    for (int i = 0; i < stride * BUF_ROWS / 8; i++) {
      in[rnd() % (stride * BUF_ROWS)] = CDEF_VERY_LARGE;
    }

    for (int dir = 0; dir < 8; dir++) {
      for (size_t s = 0; s < 5; s++) {
        for (size_t d = 0; d < 3; d++) {
          for (size_t sh = 0; sh < 2; sh++) {
            for (int variant = 0; variant < 4; variant++) {
              const int en_pri = variant != 2 && variant != 3;
              const int en_sec = variant == 0 || variant == 2;
              const int pri_damping = kDamp[d];
              const int sec_damping = kDamp[d] + 1;
              memset(cpu, 0, sizeof(uint16_t) * DSTRIDE * 8);
              memset(gpu, 0, sizeof(uint16_t) * DSTRIDE * 8);

              kRef[variant](cpu, DSTRIDE, in + i0 * stride + j0,
                            kStrength[s][0], kStrength[s][1], dir, pri_damping,
                            sec_damping, kShift[sh], w, h);

              if (avm_gpu_cdef_filter16(
                      ctx, cdef, in, BUF_ROWS, stride, i0, j0, w, h, dir,
                      kStrength[s][0], kStrength[s][1], pri_damping,
                      sec_damping, kShift[sh], en_pri, en_sec, gpu, DSTRIDE,
                      err) != 0) {
                fprintf(stderr, "avm_gpu cdef: %s\n", err);
                return 1;
              }

              int diff = 0;
              for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                  if (cpu[y * DSTRIDE + x] != gpu[y * DSTRIDE + x]) diff++;
                }
              }
              cases++;
              /* CDEF_DUMP_CASE pins the dump to the first failing configuration
               * found on the NVIDIA box (dir 0, pri 4, sec 2, damp 4/5, shift 2,
               * variant 0), so the sum probe can look at that case alone. */
              const int pinned = getenv("CDEF_DUMP_CASE") != NULL;
              const int wanted =
                  !pinned || (dir == 0 && kStrength[s][0] == 4 &&
                              kStrength[s][1] == 2 && pri_damping == 4 &&
                              sec_damping == 5 && kShift[sh] == 2 && variant == 0);
              if (diff && wanted && getenv("CDEF_DUMP")) {
                static int dumped = 0;
                if (dumped++ == 0) {
                  /* The taps reach two rows/columns beyond the block; print
                   * the whole bordered window the kernel can read. */
                  const int r0 = -2, r1 = h + 1, c0 = -4, c1 = w + 3;
                  for (int y = r0; y <= r1; y++) {
                    printf("DUMP win y%+d:", y);
                    for (int x = c0; x <= c1; x++)
                      printf(" %5d", in[(i0 + y) * stride + j0 + x]);
                    printf("\n");
                  }
                  for (int y = 0; y < h; y++) {
                    printf("DUMP in  y%d:", y);
                    for (int x = 0; x < w; x++)
                      printf(" %4d", in[(i0 + y) * stride + j0 + x]);
                    printf("\nDUMP cpu y%d:", y);
                    for (int x = 0; x < w; x++)
                      printf(" %4d", cpu[y * DSTRIDE + x]);
                    printf("\nDUMP gpu y%d:", y);
                    for (int x = 0; x < w; x++)
                      printf(" %4d", gpu[y * DSTRIDE + x]);
                    printf("\n");
                  }
                  printf("DUMP case dir %d pri %d sec %d pdamp %d sdamp %d shift %d "
                         "variant %d\n",
                         dir, kStrength[s][0], kStrength[s][1], pri_damping,
                         sec_damping, kShift[sh], variant);
                }
              }
              if (diff) {
                failures++;
                if (failures < 6) {
                  printf("  dir %d pri %d sec %d damp %d shift %d var %d: "
                         "%d samples differ\n",
                         dir, kStrength[s][0], kStrength[s][1], pri_damping,
                         kShift[sh], variant, diff);
                }
              }
            }
          }
        }
      }
    }
  }

  printf("%-10s %8s %s\n", "cdef", "cases", failures ? "MISMATCH" : "ok");
  printf("  cdef filter16: %d cases\n", cases);

  free(in);
  free(cpu);
  free(gpu);
  avm_gpu_cdef_destroy(ctx, cdef);
  avm_gpu_ctx_destroy(ctx);
  printf("\n%s\n", failures ? "FAILED" : "ALL PARITY CASES PASSED");
  return failures ? 1 : 0;
}
