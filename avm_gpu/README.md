# avm_gpu — Vulkan compute backend for AVM

Status: **M0 complete, kernels growing.** Nine kernel families (`highbd SAD`,
`highbd variance/SSE`, forward DCT 4x4/8x8, inverse DCT (DCT and ADST, square
and rectangular 4..32) with reconstruction add, intra prediction (DC/V/H/DC_LEFT/DC_TOP/DC_128, SMOOTH,
SMOOTH_V, SMOOTH_H, PAETH and the directional zones 1-3), inter 8-tap subpel
convolve, generic deblock, CDEF block filter, CCSO block pass), ten gates, and an encoder call site that produces
**byte-identical bitstreams**. Everything here is C11 and talks to Vulkan directly; there is no
CUDA, no wgpu and no C++ in the module. The data model is still M0 (see below),
and throughput is not claimed — this milestone is about correctness.

## What M0 proves

- `avm_gpu_sad_*` prices a batch of SAD candidates on the GPU and returns values
  **bit-identical** to AVM's own `avm_highbd_sadWxH_c` reference (gate:
  `avm_gpu/test/avm_gpu_sad_test.c`, 9 block sizes, 128 random candidates each,
  modulo 16-bit planes).
- `avm_gpu_var_*` returns `variance` and `sse` bit-identical to
  `avm_highbd_{8,10,12}_varianceWxH_c` for 8/10/12-bit data (gate:
  `test/avm_gpu_var_test.c`, 3 bitdepths x 9 block sizes x 128 candidates).
- **The encoder runs on it.** With `AV2_GPU_SAD=1`, `avmenc` prices SAD blocks
  through the batched hook in motion search (one dispatch per 4-candidate
  group) and produces a bitstream identical to the CPU run (gate:
  `test/run_encoder_parity.sh`, 713k blocks, md5 compared).
- `avm_gpu_txfm_*` reproduces `avm_fdct4x4_c` and `avm_highbd_fdct8x8_c`
  bit-for-bit, including the 64-bit `tran_high_t` products, which are formed
  with `imulExtended` because MoltenVK reports `shaderInt64 = 0` (gate:
  `test/avm_gpu_txfm_test.c`, 4x4 and 8x8, 12-bit and small residual ranges).
- `avm_gpu_intra_*` matches `avm_highbd_{dc,v,h,dc_left,dc_top,dc_128}_predictor_*_c`
  for 4x4…64x64 (gate: `test/avm_gpu_intra_test.c`), and carries AVM's block
  availability rule (no neighbours -> DC_128, one side -> DC_LEFT/DC_TOP) plus
  the frame-clamped edge builder. `avm_gpu_intra_dr_predict()` adds zones 1-3
  and matches `av2_highbd_dr_prediction_z{1,2,3}_c` over 6 direction/size/mrl
  combinations x 5 sizes x mrl 0-2, pixel for pixel. The SMOOTH family and
  PAETH match `avm_highbd_{smooth,smooth_v,smooth_h,paeth}_predictor_*_c`
  for 4x4...64x64; SMOOTH* reads above[bw]/left[bh] and PAETH the top-left.
- `avm_gpu_itxfm_add_rect()` reproduces `inv_txfm_c()` (`av2/common/idct.c`)
  for DCT_DCT in all 16 square and rectangular sizes from 4 to 32 and for the
  ADST primary types (ADST_DCT, DCT_ADST, ADST_ADST) up to 16, including
  `clamp_buf`, the per-size `inv_tx_shift` pairs, the `NewInvSqrt2` pre-scale
  for odd log2(w)+log2(h), the transposed pass layout (the row pass uses the
  width matrix with a height stride) and `highbd_clip_pixel_add` (gate:
  `test/avm_gpu_itxfm_test.c`, non-overlapping blocks per size). The 1D pass
  is the transform matrix itself (`tx_kernel_{dct2,adst}_sizeN[INV_TXFM][0]`),
  which is exactly what the idct.c kernels evaluate; |M| <= 90 keeps every
  product inside 32 bits.
- `avm_gpu_inter_convolve()` matches `avm_highbd_convolve8_c()` with
  `x_step_q4 = y_step_q4 = 16`: 6 subpel positions x 5 block sizes, whole-pel
  included (gate: `test/avm_gpu_inter_test.c`).
- `avm_gpu_deblock_generic()` matches `avm_highbd_lpf_{horizontal,vertical}_
  generic_c()` — filter choice plus the asymmetric delta application — over
  378 combinations of width pair, q/side thresholds, lossless flags and
  orientation (gate: `test/avm_gpu_deblock_test.c`).
- `avm_gpu_cdef_filter16()` matches `av2_cdef_filter_16_{0,1,2,3}_c` — the
  constrained directional filter, its tap tables, the `CDEF_VERY_LARGE`
  min/max handling and the final clamp — over 5760 combinations of direction,
  strengths, damping, coeff shift and wrapper variant (gate:
  `test/avm_gpu_cdef_test.c`).
- `avm_gpu_ccso_block()` matches `ccso_filter_block_hbd_wo_buf_c()`: the
  `cal_filter_support` quantisation for both `edge_clf` values, band
  extraction, LUT index and clamp, over 1008 combinations of support, edge
  classifier, thresholds, band count, block size and bo-only mode (gate:
  `test/avm_gpu_ccso_test.c`).
- The runtime reaches a device on both Linux (native Vulkan) and macOS
  (MoltenVK), including the portability-driver handshake MoltenVK needs.

## Layout

| Path | Contents |
|---|---|
| `vulkan_gpu.{h,c}` | instance/device/queue/command pool/fence, host-visible buffers, sync submit |
| `pipeline.{h,c}` | shared SPIR-V/descriptor/push-constant/pipeline setup and dispatch for every kernel |
| `vulkan_sad.{h,c}` | resident planes + batched SAD dispatch, pipeline and descriptor caching |
| `vulkan_var.{h,c}` | resident planes + batched variance/SSE, bitdepth scaling and the AVM variance formula on the host |
| `vulkan_txfm.{h,c}` | resident residual plane + batched forward DCT (4x4/8x8) |
| `vulkan_intra.{h,c}` | intra DC/V/H/DC_LEFT/DC_TOP/DC_128, availability rule, clamped edge builder |
| `vulkan_itxfm.{h,c}` | batched inverse DCT 4x4/8x8 with reconstruction add |
| `vulkan_inter.{h,c}` | 8-tap subpel inter prediction (convolve8) |
| `vulkan_deblock.{h,c}` | generic asymmetric deblock, horizontal and vertical |
| `vulkan_cdef.{h,c}` | CDEF block filter (the four 16-bit wrapper variants) |
| `vulkan_ccso.{h,c}` | CCSO block pass with the 64-band LUT |
| `kernels/txfm.comp` | DCT butterflies with 64-bit products emulated via `imulExtended` |
| `kernels/intra.comp` | intra predictors, one invocation per row |
| `kernels/intra_dir.comp` | directional zones 1-3, one invocation per pixel, with the edge guard window |
| `kernels/itxfm.comp` | inverse DCT for 4/8/16/32 driven by AVM's inverse kernel matrix |
| `kernels/inter.comp` | 8-tap separable subpel filter, one invocation per pixel |
| `kernels/deblock.comp` | `filt_choice` + `filt_generic_asym`, one invocation per 4-sample call |
| `kernels/cdef.comp` | CDEF `constrain` + pri/sec taps, one invocation per pixel |
| `kernels/ccso.comp` | `cal_filter_support` + band/class LUT lookup, one invocation per sample |
| `kernels/sad.comp` | GLSL compute shader, one workgroup per candidate |
| `kernels/variance.comp` | same shape; carries 64-bit SSE as an (hi, lo) u32 pair |
| `test/avm_gpu_sad_test.c` | parity gate against `avm_highbd_sad*_c` (exit 77 = no device) |
| `test/avm_gpu_var_test.c` | parity gate against `avm_highbd_{8,10,12}_variance*_c` |
| `test/avm_gpu_txfm_test.c` | parity gate against `avm_fdct4x4_c` / `avm_highbd_fdct8x8_c` |
| `test/avm_gpu_intra_test.c` | parity gate against the `avm_highbd_*_predictor_*` set, plus rule/edge checks |
| `test/avm_gpu_itxfm_test.c` | parity gate against `inv_txfm_c()` |
| `test/avm_gpu_inter_test.c` | parity gate against `avm_highbd_convolve8_c()` |
| `test/avm_gpu_deblock_test.c` | parity gate against the generic lpf pair |
| `test/avm_gpu_cdef_test.c` | parity gate against `av2_cdef_filter_16_{0..3}_c` |
| `test/avm_gpu_ccso_test.c` | parity gate against `ccso_filter_block_hbd_wo_buf_c()` |
| `vulkan_hook.{h,c}` | encoder seam: `avm_gpu_hook_sad()`, lazy device init, env-gated |
| `test/run_encoder_parity.sh` | end-to-end gate: CPU vs Vulkan encode, md5 must match |
| `test/make_test_clip.py` | deterministic 3-frame 64x64 4:2:0 clip for that gate |

## Data model

- **Two layouts, one kernel.** With `AVM_GPU_PACK16=1` planes are uploaded as
  **two little-endian 16-bit samples per 32-bit word** (M1): the host `memcpy`s
  the plane straight into the SSBO and the shader unpacks. Without it the M0
  layout (one sample per word) is used. Both layouts are gated by the same
  parity tests, and the packed one halves buffer size and upload bytes.
  Packed mode requires even strides; odd strides fall back to the M0 layout.
- Buffers are host-visible and coherent: planes are uploaded by `memcpy` and
  results read back after the fence. M1 stages through device-local memory with
  a copy queue and a persistent arena (the same shape as the VRAM arena in the
  encoder tree).
- **No 64-bit shader integers.** SSE reaches 2^33 at 12 bit for a 64x64 block,
  so `variance.comp` accumulates it as an (hi, lo) u32 pair and reduces with
  explicit carry. That keeps the kernel portable to backends without
  `Int64`, and the bitdepth scaling (`ROUND_POWER_OF_TWO(sse, 4|8)`) plus the
  `sse - sum^2/(w*h)` formula run on the host in 64-bit C, verbatim from
  `avm_dsp/variance.c`.
- One dispatch per call, one workgroup per candidate, `local_size_x = 256`.
  A 64×64 block costs 16 rows/thread; the batch is what fills the GPU. M2
  should fold the per-frame batch into AVM's own dispatch so there is one
  submit per frame, not one per candidate group.

## Encoder integration

Two seams, selected by `AV2_GPU_SAD`:

| value | path | shape |
|---|---|---|
| unset / `0` | CPU only | the hook is never initialised |
| `1` / `batch` | batched | `av2/encoder/mcomp.c` prices each 4-candidate group of the diamond search in one dispatch |
| `single` | per-call | `avm_dsp/sad.c::highbd_sad()` dispatches one block per call (debugging) |
| `all` | both | per-call fallback plus batched groups |

Both entries fall through to the exact CPU SAD when no device initialises, so a
machine without Vulkan still encodes. `AV2_GPU_SAD_STATS=1` prints blocks and
dispatches at exit.

```sh
AV2_GPU_SAD=1 AV2_GPU_SAD_STATS=1 ./avmenc --limit=3 --good --cpu-used=8 \
    --threads=1 -o out.ivf clip.y4m
```

### End-to-end gate

`ctest -R avm_gpu_encoder_parity` (or `test/run_encoder_parity.sh /path/to/avmenc`)
encodes the same clip twice and requires identical md5s. It fails if the hook
never initialises, prices zero blocks, or if blocks/dispatch <= 1 (i.e. the
batched path did not actually run), so neither a silent CPU fallback nor a
regression to per-call dispatching can pass.

Measured on Apple M4 via MoltenVK, 64x64, 3 frames, `--cpu-used=8 --threads=1`:

| run | blocks | dispatches | wall time | bitstream |
|---|---:|---:|---:|---|
| CPU SAD | - | - | 1.86 s | 1020 B, md5 `cd833e6f...` |
| `AV2_GPU_SAD=single` (one dispatch per SAD call) | 1,211,438 | 1,211,438 | 440.5 s | md5 `cd833e6f...` |
| `AV2_GPU_SAD=1` (batched, 4 candidates per dispatch) | 712,996 | 178,249 | 73.9 s | md5 `cd833e6f...` |
| `AV2_GPU_SAD=1 AVM_GPU_PACK16=1` (batched + packed storage) | 712,996 | 178,249 | 35.7 s | md5 `cd833e6f...` |

Batching, the reusable expansion scratch and packed storage took the GPU encode
from 440.5 s to 35.7 s (12.3x) with the same bitstream. It is still ~19x slower
than the CPU on this machine: each dispatch costs ~0.2 ms through MoltenVK and
the block is re-packed per call. The remaining fixes are structural, not micro:

1. Price the **whole search step** (up to 12 candidates) instead of 4-point
   groups, cutting dispatch count another ~3x.
2. Keep the **source and reference frames resident** for the duration of a
   block's search instead of re-packing planes per call.
3. Add the device-local arena so planes are staged once per frame instead of
   per call (packed storage already halves the traffic).

## Build and run

```sh
cmake -S . -B build-vk -G Ninja \
      -DAV2_ENABLE_VULKAN=1 -DENABLE_TESTS=1 -DENABLE_EXAMPLES=0
cmake --build build-vk --target avmenc avm_gpu_sad_test avm_gpu_var_test
ctest --test-dir build-vk -R avm_gpu_
```

The build needs `glslangValidator` (package `glslang`) and the Vulkan loader
(`vulkan-loader`; `molten-vk` too on macOS).

On macOS the loader needs an ICD path at run time:

```sh
export DYLD_LIBRARY_PATH=/opt/homebrew/lib
export VK_DRIVER_FILES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json
```

`AVM_GPU_SAD_SPV` overrides the SPIR-V path baked in at compile time.

## Measured (Apple M4, MoltenVK 1.4.2, Vulkan 1.3.357 via a 1.1 device)

128 candidates per case, resident planes, one dispatch + fence wait:

| block | CPU (1 thread, AVM C ref) | GPU |
|---|---:|---:|
| 8×8 | 0.007 ms | 0.27 ms |
| 16×16 | 0.019 ms | 0.37 ms |
| 32×32 | 0.015 ms | 0.22 ms |
| 64×64 | 0.047 ms | 0.24 ms |

Parity: **all cases exact**. Throughput: **not yet** — per-call fence latency
(~0.2 ms through MoltenVK) dominates work this small, and the M4 is a
correctness host, not an encoder target. Do not read these numbers as an
encoder speedup; the NVIDIA boxes are where the throughput case has to be made.

## Completion checklist (M0)

- [x] Vulkan runtime in C (instance/device/queue/pool/fence, buffers, submit)
- [x] highbd SAD kernel + parity gate (9 sizes x 128 candidates)
- [x] variance/SSE kernel + parity gate (3 bitdepths x 9 sizes x 128 candidates)
- [x] forward DCT 4x4/8x8 kernel + parity gate (2 sizes x 2 residual ranges)
- [x] intra prediction kernel + parity gate (10 modes x 5 sizes, availability, edges)
- [x] directional intra zones 1-3 + parity gate (90 cases x pixel compare)
- [x] inverse DCT + ADST, 16 square/rectangular sizes 4..32 + parity gate
- [x] inter 8-tap subpel convolve + parity gate (6 subpel positions x 5 sizes)
- [x] generic deblock (horizontal + vertical) + parity gate (378 cases)
- [x] CDEF block filter + parity gate (5760 cases)
- [x] CCSO block pass + parity gate (1008 cases)
- [x] encoder call site with byte-identical bitstream gate
- [x] docs, CMake option, ctest wiring (`ctest -R avm_gpu_`)

## Not done (M1/M2)

- Entropy coding, CDF state, headers, rate control: CPU by design.
- Data model: packed 16-bit storage exists behind `AVM_GPU_PACK16=1` (SAD
  only); variance and transform still widen to 32-bit words. Buffers are
  host-visible and re-uploaded per call; the device-local arena is still to do.
- Kernel families still missing: the identity (IDTX) and DDT inverse types and
  the 64-wide transform sizes, compound/warp inter and scaled reference
  prediction (x_step_q4 != 16), and loop restoration. Note that AVM 1.0's
  restoration is the PC-Wiener engine (`apply_pc_wiener_highbd`,
  `apply_wienerns_class_id_highbd`), not the classic Wiener/self-guided pair,
  and it needs the feature/class context; that port is a separate effort.
- The encoder uses the **single-candidate** path. Replacing a search loop with
  one batched dispatch per step (and keeping frames resident) is what can turn
  this into a speed win; the current 236x slowdown is per-call overhead, not
  kernel cost.
