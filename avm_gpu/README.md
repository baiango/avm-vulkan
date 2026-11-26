# avm_gpu — Vulkan compute backend for AVM

Status: **M0 complete.** Two kernels (`highbd SAD`, `highbd variance/SSE`),
three gates, and an encoder call site that produces **byte-identical
bitstreams**. Everything here is C11 and talks to Vulkan directly; there is no
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
  through `avm_gpu_hook_sad()` and produces a bitstream identical to the CPU
  run (gate: `test/run_encoder_parity.sh`, 1.2M blocks, md5 compared).
- The runtime reaches a device on both Linux (native Vulkan) and macOS
  (MoltenVK), including the portability-driver handshake MoltenVK needs.

## Layout

| Path | Contents |
|---|---|
| `vulkan_gpu.{h,c}` | instance/device/queue/command pool/fence, host-visible buffers, sync submit |
| `pipeline.{h,c}` | shared SPIR-V/descriptor/push-constant/pipeline setup and dispatch for every kernel |
| `vulkan_sad.{h,c}` | resident planes + batched SAD dispatch, pipeline and descriptor caching |
| `vulkan_var.{h,c}` | resident planes + batched variance/SSE, bitdepth scaling and the AVM variance formula on the host |
| `kernels/sad.comp` | GLSL compute shader, one workgroup per candidate |
| `kernels/variance.comp` | same shape; carries 64-bit SSE as an (hi, lo) u32 pair |
| `test/avm_gpu_sad_test.c` | parity gate against `avm_highbd_sad*_c` (exit 77 = no device) |
| `test/avm_gpu_var_test.c` | parity gate against `avm_highbd_{8,10,12}_variance*_c` |
| `vulkan_hook.{h,c}` | encoder seam: `avm_gpu_hook_sad()`, lazy device init, env-gated |
| `test/run_encoder_parity.sh` | end-to-end gate: CPU vs Vulkan encode, md5 must match |
| `test/make_test_clip.py` | deterministic 3-frame 64x64 4:2:0 clip for that gate |

## Data model (M0)

- **Planes are uploaded as one 16-bit sample per 32-bit word.** This avoids
  `VK_KHR_16bit_storage` and keeps the shader to plain `u32` arithmetic. The
  cost is 2× upload bandwidth and 2× SSBO footprint; M1 replaces this with
  `ushort` storage when the extension/feature is present, keeping the same
  kernel signature.
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

`avm_dsp/sad.c::highbd_sad()` calls `avm_gpu_hook_sad()` when the library is
built with `AVM_GPU_SAD_HOOK` (the `AV2_ENABLE_VULKAN` build defines it on
`avm_dsp_common`). The hook:

- is **inert unless `AV2_GPU_SAD=1`**, so default builds and runs are unchanged;
- falls through to the CPU loop if no device initialises, so a machine without
  Vulkan still encodes;
- packs each block into a contiguous staging plane and calls
  `avm_gpu_sad_run()` for one candidate, under a mutex (encoder threads may
  call SAD concurrently);
- prints `avm_gpu: N SAD blocks priced on the GPU` at exit when
  `AV2_GPU_SAD_STATS=1`.

```sh
AV2_GPU_SAD=1 AV2_GPU_SAD_STATS=1 ./avmenc --limit=3 --good --cpu-used=8 \
    --threads=1 -o out.ivf clip.y4m
```

### End-to-end gate

`ctest -R avm_gpu_encoder_parity` (or `test/run_encoder_parity.sh /path/to/avmenc`)
encodes the same clip twice and requires identical md5s. It fails if the hook
never initialises or prices zero blocks, so a silent CPU fallback cannot pass.

Measured on Apple M4 via MoltenVK, 64x64, 3 frames, `--cpu-used=8 --threads=1`:

| run | wall time | bitstream |
|---|---:|---|
| CPU SAD | 1.86 s | 1020 B, md5 `cd833e6f7d078f45480bf1166fecb673` |
| Vulkan SAD (`AV2_GPU_SAD=1`, 1,211,438 blocks) | 440.5 s | 1020 B, md5 `cd833e6f7d078f45480bf1166fecb673` |

That 236x slowdown is the honest result: this hook issues **one dispatch plus
one fence wait per SAD call**, which is the worst possible shape for a GPU.
It exists to prove the encoder can consume the Vulkan kernels bit-exactly, not
to make encoding faster on this machine. The throughput case needs the batched
API (`avm_gpu_sad_run()` with N candidates) wired into a search loop that
already has N candidates in hand, plus resident frames instead of per-call
staging.

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
- [x] encoder call site with byte-identical bitstream gate
- [x] docs, CMake option, ctest wiring (`ctest -R avm_gpu_`)

## Not done (M1/M2)

- Entropy coding, CDF state, headers, rate control: CPU by design.
- Data model: 16-bit samples still occupy 32-bit words, and buffers are
  host-visible; the `ushort` storage and device-local arena are M1.
- The encoder uses the **single-candidate** path. Replacing a search loop with
  one batched dispatch per step (and keeping frames resident) is what can turn
  this into a speed win; the current 236x slowdown is per-call overhead, not
  kernel cost.
