/* Diagnostic: the CWP first pass, C vs kernel vs hand evaluation. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config/avm_dsp_rtcd.h"
#include "av2/common/blockd.h"
#include "av2/common/convolve.h"
#include "av2/common/filter.h"

#include "avm_gpu/vulkan_gpu.h"
#include "avm_gpu/vulkan_inter.h"

void av2_highbd_cwp_convolve_2d_c(const uint16_t *src, int src_stride,
                                  uint16_t *dst, int dst_stride, int w, int h,
                                  const InterpFilterParams *filter_params_x,
                                  const InterpFilterParams *filter_params_y,
                                  const int subpel_x_qn, const int subpel_y_qn,
                                  ConvolveParams *conv_params, int bd);

#define SRC_W 64
#define SRC_H 64
#define CONV_STRIDE 64
#define BD 10

static uint16_t plane[SRC_W * SRC_H];

int main(void) {
  char err[AVM_GPU_ERR_LEN];
  avm_gpu_ctx *ctx = NULL;
  if (avm_gpu_ctx_create(&ctx, err) != 0) {
    printf("no device: %s\n", err);
    return 77;
  }
  avm_gpu_inter *inter = NULL;
  if (avm_gpu_inter_create(ctx, &inter, err) != 0) {
    printf("avm_gpu inter: %s\n", err);
    return 1;
  }
  /* Deterministic plane, so the taps can be checked by hand. */
  for (int y = 0; y < SRC_H; y++) {
    for (int x = 0; x < SRC_W; x++) {
      plane[y * SRC_W + x] = (uint16_t)((100 + 7 * x + 3 * y) & 0x3ff);
    }
  }
  InterpFilterParams fp_x = { (const int16_t *)av2_sub_pel_filters_8, 8 };
  InterpFilterParams fp_y = { (const int16_t *)av2_sub_pel_filters_8, 8 };
  uint16_t cpu_conv[CONV_STRIDE * 64], gpu_conv[CONV_STRIDE * 64];
  uint16_t cpu_out[CONV_STRIDE * 64], gpu_out[CONV_STRIDE * 64];
  const int w = 64, h = 64, sx = 8, sy = 8;
  const int use_gpu = getenv("CWP_DBG_NOGPU") == NULL;

  for (int px = 0; px <= 5; px++) {
    for (int py = 0; py <= 0; py++) {
      ConvolveParams conv =
          get_conv_params_no_round(0, 0, cpu_conv, CONV_STRIDE, 1, BD);
      conv.do_average = 0;
      memset(cpu_conv, 0, sizeof(cpu_conv));
      memset(gpu_conv, 0, sizeof(gpu_conv));
      av2_highbd_cwp_convolve_2d_c(plane + sy * SRC_W + sx, SRC_W, cpu_out,
                                   CONV_STRIDE, w, h, &fp_x, &fp_y, px, py,
                                   &conv, BD);
      if (use_gpu && avm_gpu_inter_cwp_convolve(
              ctx, inter, plane, SRC_W, SRC_H, sx, sy,
              (const int16_t *)av2_sub_pel_filters_8,
              (const int16_t *)av2_sub_pel_filters_8, px, py, w, h, BD,
              conv.round_0, conv.round_1, gpu_conv, CONV_STRIDE, 0, 0, 0, 0, 0,
              0, 16, 16, gpu_out, CONV_STRIDE, err) != 0) {
        printf("kernel: %s\n", err);
        return 1;
      }
      /* Hand evaluation of pixel (0,0) with the C's own steps. */
      const int16_t *fx = av2_sub_pel_filters_8[px];
      const int16_t *fy = av2_sub_pel_filters_8[py];
      int im[8];
      for (int ky = 0; ky < 8; ky++) {
        int sum = 1 << (BD + 6);
        for (int kx = 0; kx < 8; kx++) {
          sum += fx[kx] * plane[(sy + ky - 3) * SRC_W + (sx + kx - 3)];
        }
        im[ky] = (int16_t)((sum + (1 << (conv.round_0 - 1))) >> conv.round_0);
      }
      int vsum = 1 << (BD + 14 - conv.round_0);
      for (int ky = 0; ky < 8; ky++) vsum += fy[ky] * im[ky];
      const unsigned hand = (unsigned)((vsum + (1 << (conv.round_1 - 1))) >>
                                       conv.round_1) & 0xffffu;
      printf("subpel (%d,%d) r0=%d r1=%d: C %u  GPU %u  hand %u  %s\n", px, py,
             conv.round_0, conv.round_1, cpu_conv[0], gpu_conv[0], hand,
             (cpu_conv[0] == gpu_conv[0] && cpu_conv[0] == hand) ? "agree"
                                                                : "DIFFER");
    }
  }
  avm_gpu_inter_destroy(ctx, inter);
  avm_gpu_ctx_destroy(ctx);
  return 0;
}
