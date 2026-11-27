# avm_gpu — Vulkan compute backend for AVM

Status: **M0 complete, kernels growing.** Nine kernel families (`highbd SAD`,
`highbd variance/SSE`, forward DCT 4x4/8x8, inverse DCT (DCT, ADST, IDTX, FLIPADST and the DDT substitutions,
square and rectangular 4..64) with reconstruction add, intra prediction (DC/V/H/DC_LEFT/DC_TOP/DC_128, SMOOTH,
SMOOTH_V, SMOOTH_H, PAETH and the directional zones 1-3), inter 8-tap subpel convolve (unscaled and
scaled), compound A64 mask blending, weighted (CWP) compound and affine
(warped) prediction (regular and extended-range filters), generic deblock, CDEF block filter, CCSO block pass), ten gates, and an encoder call site that produces
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
  ADST primary types (ADST_DCT, DCT_ADST, ADST_ADST) up to 16 and for IDTX /
  V_DCT / H_DCT / V_ADST / H_ADST (identity is a scaled diagonal matrix:
  128, 181, 256, 362 for sizes 4, 8, 16, 32) and for FLIPADST_DCT /
  DCT_FLIPADST / FLIPADST_FLIPADST / ADST_FLIPADST / FLIPADST_ADST (DCT8 maps
  to inv_txfm_fdst_sizeN_c) and for the DDT substitution (with use_ddt the
  8/16-point DST7/DCT8 kernels become DDTX/FDDT; REPLACE_ADST8/16 are 1 here,
  REPLACE_ADST4 is 0 so 4x4 keeps ADST), including
  `clamp_buf`, the per-size `inv_tx_shift` pairs, the `NewInvSqrt2` pre-scale
  for odd log2(w)+log2(h), the transposed pass layout (the row pass uses the
  width matrix with a height stride) and `highbd_clip_pixel_add` (gate:
  `test/avm_gpu_itxfm_test.c`, non-overlapping blocks per size). The 1D pass
  is the transform matrix itself (`tx_kernel_{dct2,adst}_sizeN[INV_TXFM][0]`),
  which is exactly what the idct.c kernels evaluate; |M| <= 90 keeps every
  product inside 32 bits.
- `avm_gpu_inter_convolve_scaled()` matches `avm_highbd_convolve8_c()` for
  any `x_step_q4` / `y_step_q4` in 1..64 - unscaled (16) and scaled reference
  prediction - with per-pixel fractional phases: 6 subpel positions x block
  sizes for 5 step pairs, whole-pel included (gate:
  `test/avm_gpu_inter_test.c`, 120 cases).
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
- `avm_gpu_inter_blend_a64()` matches `avm_highbd_blend_a64_mask_c()` for all
  four mask subsampling modes, and `avm_gpu_inter_cwp()` matches
  `avm_highbd_cwp_c()` for 5 offset pairs x 5 block sizes (gate:
  `test/avm_gpu_inter_test.c`, 165 cases in total).
- `avm_gpu_warp_affine()` matches `av2_highbd_warp_affine_c()` for the
  single-reference case, luma and chroma: 3 affine matrices x 3 shear settings
  x 7 block sizes (luma 8x8..32x16, chroma 4x4..8x8) x the chroma subsampling
  modes 420/422/440, whole prediction plane compared (gate:
  `test/avm_gpu_warp_test.c`, 117 cases). The chroma projection shifts the
  block centre to luma coordinates and back with a 64-bit arithmetic shift.
- `avm_gpu_warp_affine_ext()` matches `av2_ext_highbd_warp_affine_c()`
  (single-reference, 4x4 chunks, 6-tap ext filter): 3 matrices x 4 block sizes
  x 3 subsampling modes, 36 further cases.
  MoltenVK has no `shaderInt64`, so the affine projection uses
  `imulExtended` with explicit hi/lo carry.
- The runtime reaches a device on both Linux (native Vulkan) and macOS
  (MoltenVK), including the portability-driver handshake MoltenVK needs.

## Layout

| Path | Contents |
|---|---|
| `vulkan_gpu.{h,c}` | instance/device/queue/command pool/fence, host-visible buffers, sync submit |
| `pipeline.{h,c}` | shared SPIR-V/descriptor/push-constant/pipeline setup and dispatch for every kernel |
| `arena.{h,c}` | device-local arena: one large `DEVICE_LOCAL` buffer with a 64-byte-aligned first-fit suballocator, a staging buffer and copy commands; plus the `avm_gpu_slot_*` helpers the modules use |
| `vulkan_sad.{h,c}` | resident planes + batched SAD dispatch, pipeline and descriptor caching |
| `vulkan_var.{h,c}` | resident planes + batched variance/SSE, bitdepth scaling and the AVM variance formula on the host |
| `vulkan_txfm.{h,c}` | resident residual plane + batched forward DCT (4x4/8x8) |
| `vulkan_intra.{h,c}` | intra DC/V/H/DC_LEFT/DC_TOP/DC_128, availability rule, clamped edge builder |
| `vulkan_itxfm.{h,c}` | batched inverse DCT 4..64 with reconstruction add (`avm_gpu_itxfm_add_tx()` for the 64-wide/tall sizes) |
| `vulkan_inter.{h,c}` | 8-tap subpel inter prediction incl. scaled reference, and compound A64 blending |
| `vulkan_warp.{h,c}` | affine (warped) prediction, one invocation per 8x8 block |
| `vulkan_deblock.{h,c}` | generic asymmetric deblock, horizontal and vertical |
| `vulkan_cdef.{h,c}` | CDEF block filter (the four 16-bit wrapper variants) |
| `vulkan_ccso.{h,c}` | CCSO block pass with the 64-band LUT |
| `pcwiener.{h,c}` | PC-Wiener restoration, both stages: `avm_gpu_pcwiener_classify()` and `avm_gpu_pcwiener_filter()`, host-derived tap geometry |
| `kernels/txfm.comp` | DCT butterflies with 64-bit products emulated via `imulExtended` |
| `kernels/intra.comp` | intra predictors, one invocation per row |
| `kernels/intra_dir.comp` | directional zones 1-3, one invocation per pixel, with the edge guard window |
| `kernels/itxfm.comp` | inverse DCT for 4/8/16/32 driven by AVM's inverse kernel matrix, plus the 2x replicate of the 64-wide/tall sizes |
| `kernels/inter.comp` | 8-tap separable subpel filter, one invocation per pixel |
| `kernels/compound.comp` | A64 mask blend of two predictors (mask subsampling included) |
| `kernels/cwp.comp` | weighted (CWP) compound average, `avm_highbd_cwp_c` |
| `kernels/warp.comp` | affine projection (64-bit products via `imulExtended`) + 8-tap warp filter |
| `kernels/warp_ext.comp` | extended-range warp: 4x4 chunks + 6-tap `av2_ext_warped_filter` |
| `kernels/deblock.comp` | `filt_choice` + `filt_generic_asym`, one invocation per 4-sample call |
| `kernels/cdef.comp` | CDEF `constrain` + pri/sec taps, one invocation per pixel |
| `kernels/ccso.comp` | `cal_filter_support` + band/class LUT lookup, one invocation per sample |
| `kernels/cwpconv.comp` | compound prediction, the CWP family: `av2_highbd_cwp_convolve_2d_c()` two passes over the high-precision buffer |
| `kernels/pcwiener.comp` | PC-Wiener loop restoration: the symmetric 25-tap non-separable filter per 4x4 class block, one invocation per pixel |
| `kernels/pcwcls.comp` | PC-Wiener classification: 6x6 directional-feature and tx-skip box sums, qval/tskip LUT, thresholds and class LUT, one invocation per 4x4 block |
| `kernels/sad.comp` | GLSL compute shader, one workgroup per candidate |
| `kernels/variance.comp` | same shape; carries 64-bit SSE as an (hi, lo) u32 pair |
| `test/avm_gpu_sad_test.c` | parity gate against `avm_highbd_sad*_c` (exit 77 = no device) |
| `test/avm_gpu_var_test.c` | parity gate against `avm_highbd_{8,10,12}_variance*_c` |
| `test/avm_gpu_txfm_test.c` | parity gate against `avm_fdct4x4_c` / `avm_highbd_fdct8x8_c` |
| `test/avm_gpu_intra_test.c` | parity gate against the `avm_highbd_*_predictor_*` set, plus rule/edge checks |
| `test/avm_gpu_itxfm_test.c` | parity gate against `inv_txfm_c()` |
| `test/avm_gpu_inter_test.c` | parity gate against `avm_highbd_convolve8_c()` and the A64 blend |
| `test/avm_gpu_warp_test.c` | parity gate against `av2_highbd_warp_affine_c()` |
| `test/avm_gpu_deblock_test.c` | parity gate against the generic lpf pair |
| `test/avm_gpu_cdef_test.c` | parity gate against `av2_cdef_filter_16_{0..3}_c` |
| `test/avm_gpu_ccso_test.c` | parity gate against `ccso_filter_block_hbd_wo_buf_c()` |
| `test/avm_gpu_pcwiener_test.c` | end-to-end gate against `apply_pc_wiener_highbd()`: class ids, then the restored plane |
| `test/avm_gpu_pcwiener_filter_test.c` | filter stage alone against `av2_convolve_symmetric_highbd_c()`, AVM's own filter banks and tap config |
| `test/avm_gpu_lr_hook_test.c` | seam gate: procunit-sized stripes (64/32/partial) against `apply_pc_wiener_highbd()` |
| `test/avm_gpu_seam_repro.c` | inter seam at block level: 400 hook-vs-C comparisons over freshly filled planes (unscaled and scaled, 4..64), and a replay of the first failing case to say whether it is deterministic or stale data |
| `vulkan_hook.{h,c}` | encoder seam: `avm_gpu_hook_sad()`, `avm_gpu_hook_sad_multi()`, the residency scope, lazy device init, env-gated |
| `vulkan_hook_var.{h,c}` | encoder seam for variance/SSE: content-keyed memo, block-area threshold, lazy device init |
| `vulkan_hook_inter.{h,c}` | inter prediction seam: single reference, unscaled, 8-tap subpel, `avm_gpu_hook_inter()` |
| `vulkan_hook_lr.{h,c}` | loop-restoration seam: `avm_gpu_hook_pcwiener()` runs one procunit through the classifier + filter kernels, with the lossless per-block copy mask |
| `test/avm_gpu_var_hook_test.c` | gate for the variance seam: memo hit path + every block size against the reference loop |
| `test/run_encoder_parity.sh` | end-to-end gate: CPU vs Vulkan encode, md5 must match |
| `test/make_test_clip.py` | deterministic 3-frame 64x64 4:2:0 clip for that gate |
| `test/make_motion_clip.py` | 5-frame clip whose content moves by half a pixel per frame, so the inter gate exercises the subpel filter rather than the integer copy path |

## Data model

- **Two layouts, one kernel.** With `AVM_GPU_PACK16=1` planes are uploaded as
  **two little-endian 16-bit samples per 32-bit word** (M1): the host `memcpy`s
  the plane straight into the SSBO and the shader unpacks. Without it the M0
  layout (one sample per word) is used. SAD, variance/SSE and the forward DCT
  all support it; `avm_gpu_var_test_packed` and `avm_gpu_txfm_test_packed`
  gate the packed path in ctest, and the encoder parity gate passes with
  `AVM_GPU_PACK16=1` exported. Packed mode requires even strides; odd strides
  fall back to the M0 layout.
- **Two placements.** By default buffers are host-visible and coherent: planes
  are uploaded by `memcpy` and results read back after the fence. With
  `AV2_GPU_ARENA=1` the SAD and variance modules instead suballocate their
  buffers from a single device-local `VkBuffer` (`arena.{h,c}`), upload through
  a reusable host-visible staging buffer with `vkCmdCopyBuffer`, and read back
  the same way. The descriptor offset of a suballocation travels in
  `avm_gpu_buf.offset`, and a suballocation is recognisable by
  `buffer != NULL, memory == NULL`, so `avm_gpu_buf_destroy()` never frees
  memory it does not own. If the arena cannot satisfy a request (or is full)
  the slot silently falls back to a host-visible buffer, and
  `AV2_GPU_ARENA_VERBOSE=1` reports it.
  `AV2_GPU_ARENA_MB` sets the arena size (default 64 MiB).
  On Apple silicon the arena is *slower* than plain host-visible buffers
  (unified memory, so the staging copy is pure overhead): the three-encode
  parity gate takes 126 s with the default placement and 543 s with
  `AV2_GPU_ARENA=1`, with identical md5s. It is the right placement for a
  discrete GPU (VRAM), which is what it exists for; it is off by default.
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

| env | seam | shape |
|---|---|---|
| `AV2_GPU_SAD=1` (or `batch`) | `av2/encoder/mcomp.c` diamond search | each 4-candidate group in one dispatch |
| `AV2_GPU_SAD=single` | `avm_dsp/sad.c::highbd_sad()` | one block per call (debugging) |
| `AV2_GPU_SAD=all` | both | per-call fallback plus batched groups |
| `AV2_GPU_VAR=1` | `avm_dsp/variance.c::highbd_variance64()` | variance/SSE blocks of at least `AV2_GPU_VAR_MIN_AREA` samples (default 1024), memoised by block content |
| `AV2_GPU_VAR_MIN_AREA=N` | `avm_gpu/vulkan_hook_var.c` | block area below which the call stays on the CPU (`0` = every block) |
| `AV2_GPU_VAR_MEMO=0` | `avm_gpu/vulkan_hook_var.c` | disable the content-keyed memo (A/B switch) |
| `AV2_GPU_SAD_RESIDENT=0` | `avm_gpu/vulkan_hook.c` | drop source-block residency inside a search (A/B switch) |
| `AV2_GPU_SAD_LOOKAHEAD=0` | `av2/encoder/mcomp.c` | one dispatch per search step instead of one per two steps (A/B switch, also a ctest case) |

Both entries fall through to the exact CPU SAD when no device initialises, so a
machine without Vulkan still encodes. `AV2_GPU_SAD_STATS=1` prints blocks and
dispatches at exit.

```sh
AV2_GPU_SAD=1 AV2_GPU_SAD_STATS=1 ./avmenc --limit=3 --good --cpu-used=8 \
    --threads=1 -o out.ivf clip.y4m
```

### Variance seam

The variance seam sits in `highbd_variance64()`, so it sees every variance and
SSE request of the encoder (4x4..256x256). Two things make it pay:

- **Content-keyed memo.** `(sse, sum)` is a pure function of the two blocks and
  their shape, so the key can be the content itself: a 64-bit FNV-1a over both
  blocks plus a 32-bit checksum and the shape. A hit cannot go stale, and no
  scope or generation token is needed. In the gate clip 44 % of the calls are
  exact repeats of an earlier pair (the same source block against the same
  predictor in several RD trials); with a 256 K-entry direct-mapped table the
  hit rate is 51 %.
- **Block-area threshold.** A dispatch costs a fence round trip (~0.2 ms through
  MoltenVK) while a 16x16 variance is well under a microsecond, so calls below
  `AV2_GPU_VAR_MIN_AREA` samples (default 1024, i.e. 32x32) stay on the CPU.
  77 % of the clip's calls are 4x4..8x8; they are cheaper on the CPU than the
  descriptor update alone. The remaining 5,261 calls still go through the
  kernel, so the seam stays live and gated.

`AV2_GPU_VAR_STATS=1` prints the per-area call histogram, the memo hit rate and
the number of blocks the kernel priced.

### Search lookahead

Each search step prices its candidates in one batch, and every other step also
prices the *next* step's sites around every position that step could elect
(the current centre plus each valid candidate), in the same dispatch. The
winner of a step is one of those centres, so the following step's candidate
list is already priced and needs no dispatch at all - the search costs one
dispatch per two steps instead of two. Speculative positions never change a
decision, they are only extra SADs, so the search trajectory is untouched.
The lookahead is limited to blocks of at most 32x32 samples, where the extra
packing is far cheaper than a dispatch; larger blocks keep one dispatch per
step. `AV2_GPU_SAD_LOOKAHEAD=0` restores the previous shape and is covered by
`avm_gpu_encoder_parity_no_lookahead`.

Buffer placement used by both seams:

| env | effect |
|---|---|
| unset | host-visible, coherent per-call buffers |
| `AV2_GPU_ARENA=1` | device-local arena + staging copies (`AV2_GPU_ARENA_MB`, default 64) |
| `AV2_GPU_ARENA_VERBOSE=1` | report arena size and any slot fallback |
| `AVM_GPU_PACK16=1` | packed 16-bit upload layout (SAD, variance, forward DCT) |

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
| `AV2_GPU_SAD=1` (4-candidate groups) | 712,996 | 178,249 | 73.9 s | md5 `cd833e6f...` |
| same + `AVM_GPU_PACK16=1` | 712,996 | 178,249 | 35.7 s | md5 `cd833e6f...` |
| whole-step batching + packed storage | 714,775 | 80,895 | 17.7 s | md5 `cd833e6f...` |
| + source residency (`AV2_GPU_SAD=1`, `AVM_GPU_PACK16=1`) | 714,775 | 80,895 | 17.0 s | md5 `cd833e6f...` |
| + search lookahead (one dispatch per two steps) | 2,674,812 | 51,403 | **12.3 s** | md5 `cd833e6f...` |
| + search lookahead, `AV2_GPU_ARENA=1` | 2,674,812 | 51,403 | 12.8 s | md5 `cd833e6f...` |

Variance seam, same clip (`AV2_GPU_VAR=1`, whole encode):

| run | kernel blocks | wall time | bitstream |
|---|---:|---:|---|
| CPU only | - | 0.87 s | md5 `cd833e6f...` |
| every block, no memo (`AV2_GPU_VAR_MIN_AREA=0 AV2_GPU_VAR_MEMO=0`) | 457,023 | 92.8 s | md5 `cd833e6f...` |
| every block, memo on | 457,023 (231,476 hits) | 45.9 s | md5 `cd833e6f...` |
| default (`MIN_AREA=1024`, memo on) | 5,261 (1,367 hits) | **1.9 s** | md5 `cd833e6f...` |

Batching, the reusable expansion scratch and packed storage took the GPU encode
from 440.5 s to 35.7 s (12.3x) with the same bitstream. It is still ~19x slower
than the CPU on this machine: each dispatch costs ~0.2 ms through MoltenVK and
the block is re-packed per call. The remaining fixes are structural, not micro:

1. ~~Price the whole search step instead of 4-point groups.~~ Done: the
   diamond search now prices every candidate of a step in one dispatch
   (8.8 blocks/dispatch, 80,895 dispatches, 17.7 s vs 35.7 s). Note that
   validity is now decided per candidate instead of per group; on the gate clip
   the bitstream is unchanged.
2. ~~Keep the source block resident for the duration of a search.~~ Done:
   `avm_gpu_hook_sad_scope_begin()/end()` wrap one diamond search, inside which
   the source block is immutable, so the hook uploads it once and re-uploads
   only the candidate reference blocks (59,025 of 80,895 source uploads skipped
   on the gate clip). A scope is what makes the pointer identity meaningful:
   nothing stays resident across scopes.
3. ~~Add a device-local arena.~~ Done (`AV2_GPU_ARENA=1`, gated by
   `avm_gpu_sad_test_arena` / `avm_gpu_var_test_arena`): it is correct but a
   loss on unified memory; the next win is to keep the planes resident across
   calls (item 2) rather than to move them to VRAM.

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
- [x] inverse DCT + ADST + IDTX + FLIPADST + DDT, 4..32, and all 9 64-wide/tall
      sizes + parity gate
- [x] inter 8-tap subpel convolve + parity gate (6 subpel positions x 5 sizes)
- [x] inter scaled reference (steps 1..64), compound A64 blend, weighted (CWP)
      compound, warp affine (luma + chroma, regular + extended-range filters)
- [x] generic deblock (horizontal + vertical) + parity gate (378 cases)
- [x] CDEF block filter + parity gate (5760 cases)
- [x] CCSO block pass + parity gate (1008 cases)
- [x] PC-Wiener restoration filter stage + parity gate (2 geometries x 4 filter
      sets x 2 bitdepths against `av2_convolve_symmetric_highbd_c()`)
- [x] PC-Wiener classification stage + end-to-end gate (class ids and the
      restored plane against `apply_pc_wiener_highbd()`, 2 bitdepths x
      tx-skip on/off)
- [x] compound inter (CWP 2D) kernel + parity gate (2 bitdepths x even/uneven
      weights x 6 subpel positions) and, in the seam, compound prediction itself
- [x] inter prediction call site: `av2_make_inter_predictor()` runs the subpel
      kernel when `AV2_GPU_INTER=1` (single reference, unscaled or scaled,
      8-tap), with the encoder bitstream gate on a half-pel motion clip
- [x] loop-restoration call site: `pc_wiener_stripe_highbd()` runs the kernels
      when `AV2_GPU_LR=1`, gated by `avm_gpu_lr_hook_test` (6 procunit shapes x
      2 bitdepths x tx-skip on/off)
- [x] encoder call sites (batched SAD and variance) with a byte-identical
      bitstream gate
- [x] source-block residency per search and one-step search lookahead
      (80,895 -> 51,403 dispatches, 18.1 s -> 12.3 s, same md5)
- [x] variance seam: content-keyed memo + block-area threshold
      (457,023 -> 5,261 kernel blocks, 92.8 s -> 1.9 s, same md5)
- [x] device-local arena + staging copies, opt-in, gated for SAD and variance
- [x] docs, CMake option, ctest wiring (`ctest -R avm_gpu_`)

## Not done (M1/M2)

- Entropy coding, CDF state, headers, rate control: CPU by design.
- Data model: packed 16-bit storage covers SAD, variance and the forward DCT;
  the inverse transform, intra, inter, warp and the filters still widen to
  32-bit words. A device-local arena exists for SAD and variance, but the
  remaining modules use host-visible buffers and re-upload per call.
- Kernel families still missing: the forward DCT only covers 4x4/8x8. The inter
  seam covers single-reference **unscaled** 8-tap prediction; compound, warped,
  scaled and >8-tap encoder-side filters still run on the CPU.
- The inter seam's upload window is `last_base_y + 7` rows, which is the exact
  minimum the entry point enforces (`sy + last_base + 4 <= rows`, `sy == 3`),
  not slack. Measured: `last_base_y + 7` agrees over 800 comparisons, one row
  less makes the seam decline every case. The seam test treats a decline as a
  failure - an earlier revision of it counted declines as passes, which is why
  an under-sized window looked clean.
- **Compound prediction** now has its own kernel and call site: for a 2D
  compound call (a non-zero phase on both axes, unscaled, single reference
  pass) the seam runs `kernels/cwpconv.comp` and leaves the CWP x-only/y-only/
  copy variants - a different rounding - on the CPU. The seam gate covers both
  passes (even and uneven weights) and the encoder now prices 2,530 compound
  blocks with an unchanged bitstream.
- **Compound prediction is not a seam extension.** AVM 1.0 does not produce two
  sr predictors and blend them: with `conv_params->is_compound`,
  `av2_highbd_convolve_2d_facade()` dispatches to the CWP family
  (`av2_highbd_cwp_convolve_2d/x/y/copy`), which writes the *high-precision*
  `CONV_BUF_TYPE` buffer on the first pass and does the (optionally uneven
  weighted, `DIST_PRECISION_BITS`) average in the second, subtracting the two
  pass biases before the final rounding. Bridging it needs a new kernel for that
  two-pass path; `compound.comp` (`avm_highbd_blend_a64_mask_c`) is a different
  operation and cannot be substituted.
- Loop restoration is wired (`AV2_GPU_LR=1` in `pc_wiener_stripe_highbd()`,
  which serves both the encoder and the decoder) and the seam is gated, but no
  end-to-end stream gate exists yet: with the settings in this repo the encoder
  exercises the path (24 procunits for the gate clip) yet the chosen
  restoration type does not change the bitstream, and the streams we produce do
  not signal `RESTORE_PC_WIENER` units to the decoder. An end-to-end LR gate
  needs a stream whose RD actually picks PC-Wiener units.
- The encoder uses the **single-candidate** path. Replacing a search loop with
  one batched dispatch per step (and keeping frames resident) is what can turn
  this into a speed win; the current 236x slowdown is per-call overhead, not
  kernel cost.
