# avm_gpu — Vulkan compute backend for AVM

Status: **M0 / proof of parity.** Two kernels (`highbd SAD`,
`highbd variance/SSE`), one gate each, no encoder wiring. Everything here is
C11 and talks to Vulkan directly; there is no CUDA, no wgpu and no C++ in the
module.

## What M0 proves

- `avm_gpu_sad_*` prices a batch of SAD candidates on the GPU and returns values
  **bit-identical** to AVM's own `avm_highbd_sadWxH_c` reference (gate:
  `avm_gpu/test/avm_gpu_sad_test.c`, 9 block sizes, 128 random candidates each,
  modulo 16-bit planes).
- `avm_gpu_var_*` returns `variance` and `sse` bit-identical to
  `avm_highbd_{8,10,12}_varianceWxH_c` for 8/10/12-bit data (gate:
  `test/avm_gpu_var_test.c`, 3 bitdepths x 9 block sizes x 128 candidates).
- The runtime reaches a device on both Linux (native Vulkan) and macOS
  (MoltenVK), including the portability-driver handshake MoltenVK needs.

## Layout

| Path | Contents |
|---|---|
| `vulkan_gpu.{h,c}` | instance/device/queue/command pool/fence, host-visible buffers, sync submit |
| `vulkan_sad.{h,c}` | resident planes + batched SAD dispatch, pipeline and descriptor caching |
| `vulkan_var.{h,c}` | resident planes + batched variance/SSE, bitdepth scaling and the AVM variance formula on the host |
| `kernels/sad.comp` | GLSL compute shader, one workgroup per candidate |
| `kernels/variance.comp` | same shape; carries 64-bit SSE as an (hi, lo) u32 pair |
| `test/avm_gpu_sad_test.c` | parity gate against `avm_highbd_sad*_c` (exit 77 = no device) |
| `test/avm_gpu_var_test.c` | parity gate against `avm_highbd_{8,10,12}_variance*_c` |

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

## Build and run

```sh
cmake -S . -B build-vk -G Ninja \
      -DAV2_ENABLE_VULKAN=1 -DENABLE_TESTS=1 -DENABLE_EXAMPLES=0
cmake --build build-vk --target avm_gpu_sad_test
ctest --test-dir build-vk -R avm_gpu_sad_test
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

## Non-goals / not wired

- Entropy coding, CDF state, headers, rate control: CPU by design.
- No encoder call site yet. The next step is to route the motion-search SAD
  candidates through `avm_gpu_sad_run()` behind a runtime flag and compare
  bitstreams with the CPU path.
- `vulkan_sad.c` and `vulkan_var.c` duplicate the pipeline/descriptor setup;
  that should become one `avm_gpu_pipeline` helper before a third kernel lands.
