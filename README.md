# AVM-Vulkan

> **Fork notice.** This is [`baiango/avm-vulkan`](https://github.com/baiango/avm-vulkan),
> a fork of [AOMediaCodec/avm](https://github.com/AOMediaCodec/avm). Upstream
> AVM is unchanged except for an optional Vulkan compute backend in `avm_gpu/`
> (off by default, branch `vulkan`). See
> [Vulkan compute backend](#vulkan-compute-backend-fork) below.

## Building

### Prerequisites

 1. [CMake](https://cmake.org) version 3.16 or higher.
 2. [Git](https://git-scm.com/).
 3. [Perl](https://www.perl.org/).
 4. For x86 targets, [yasm](https://www.tortall.net/projects/yasm/releases/), which is preferred, or a
    recent version of [nasm](http://www.nasm.us/). If you download yasm with
    the intention to work with Visual Studio, please download win32.exe or
    win64.exe and rename it into yasm.exe. DO NOT download or use vsyasm.exe.
 5. Building the documentation requires
   [doxygen version 1.8.10 or newer](http://doxygen.org).
 6. Building the unit tests requires [Python](https://www.python.org/).

### Get the code

The AVM project source code is stored in the Alliance of Open Media’s Github [repository](https://github.com/AOMediaCodec/avm.git). To get the code,
~~~
    $ git clone https://github.com/AOMediaCodec/avm.git
    # By default, the above command stores the source in the avm directory:
    $ cd avm
~~~

### Basic build

CMake replaces the configure step typical of many projects. Running CMake will
produce configuration and build files for the currently selected CMake
generator. For most systems the default generator is Unix Makefiles. The basic
form of a makefile build is the following:

~~~
    $ cmake path/to/avm
    $ make
~~~

The above will generate a makefile build that produces the AVM library and
applications for the current host system after the make step completes
successfully. The compiler chosen varies by host platform, but a general rule
applies: On systems where cc and c++ are present in $PATH at the time CMake is
run the generated build will use cc and c++ by default.

### Configuration options

The AVM codec library has a great many configuration options. These come in two
varieties:

 1. Build system configuration options. These have the form `ENABLE_FEATURE`.
 2. AVM codec configuration options. These have the form `CONFIG_FEATURE`.

Both types of options are set at the time CMake is run. The following example
enables ccache and disables the AVM encoder:

~~~
    $ cmake path/to/avm -DENABLE_CCACHE=1 -DCONFIG_MULTITHREAD=0
    $ make
~~~

The available configuration options are too numerous to list here. Build system
configuration options can be found at the top of the CMakeLists.txt file found
in the root of the AVM repository, and AVM codec configuration options can
currently be found in the file `cmake/avm_config_defaults.cmake`.

### Dylib builds

A dylib (shared object) build of the AVM codec library can be enabled via the
CMake built in variable `BUILD_SHARED_LIBS`:

~~~
    $ cmake path/to/avm -DBUILD_SHARED_LIBS=1
    $ make
~~~

This is currently only supported on non-Windows targets.

### Debugging

Depending on the generator used there are multiple ways of going about
debugging AVM components. For single configuration generators like the Unix
Makefiles generator, setting `CMAKE_BUILD_TYPE` to Debug is sufficient:

~~~
    $ cmake path/to/avm -DCMAKE_BUILD_TYPE=Debug
~~~

For Xcode, mainly because configuration controls for Xcode builds are buried two
configuration windows deep and must be set for each subproject within the Xcode
IDE individually, `CMAKE_CONFIGURATION_TYPES` should be set to Debug:

~~~
    $ cmake path/to/avm -G Xcode -DCMAKE_CONFIGURATION_TYPES=Debug
~~~

For Visual Studio the in-IDE configuration controls should be used. Simply set
the IDE project configuration to Debug to allow for stepping through the code.

In addition to the above it can sometimes be useful to debug only C and C++
code. To disable all assembly code and intrinsics set `AVM_TARGET_CPU` to
generic at generation time:

~~~
    $ cmake path/to/avm -DAVM_TARGET_CPU=generic
~~~

### Cross compiling

For the purposes of building the AVM codec and applications, relative to the scope of this guide,
all builds for architectures differing from the native host architecture will be considered cross compiles.
The AVM CMake build handles cross compiling via the use of toolchain files included in the AVM repository.
The available toolchain files can be found at cmake folder in the AVM repository.
The following example demonstrates use of the x86-linux.cmake toolchain file on a x86_64 linux host:

~~~
    $ cmake path/to/avm \
      -DCMAKE_TOOLCHAIN_FILE=path/to/avm/cmake/toolchains/x86-linux.cmake
    $ make
~~~

### Sanitizers

Sanitizer integration is built-in to the CMake build system. To enable a
sanitizer, add `-DSANITIZE=<type>` to the CMake command line. For example, to
enable address sanitizer:

~~~
    $ cmake path/to/avm -DSANITIZE=address
    $ make
~~~

Sanitizers available vary by platform, target, and compiler. Consult your
compiler documentation to determine which, if any, are available.

### Microsoft Visual Studio builds

Building the AVM codec library in Microsoft Visual Studio is supported. Visual
Studio 2019 (16.7) or later is required. The following example demonstrates
generating projects and a solution for the Microsoft IDE:

~~~
    # This does not require a bash shell; Command Prompt (cmd.exe) is fine.
    # This assumes the build host is a Windows x64 computer.

    # To build with Visual Studio 2019 for the x64 target:
    $ cmake path/to/avm -G "Visual Studio 16 2019"
    $ cmake --build .

    # To build with Visual Studio 2019 for the 32-bit x86 target:
    $ cmake path/to/avm -G "Visual Studio 16 2019" -A Win32
    $ cmake --build .
~~~

NOTE: The build system targets Windows 7 or later by compiling files with
`-D_WIN32_WINNT=0x0601`.

### Xcode builds

Building the AVM codec library in Xcode is supported. The following example
demonstrates generating an Xcode project:

~~~
    $ cmake path/to/avm -G Xcode
~~~

### Build with VMAF support

After installing
[libvmaf.a](https://github.com/Netflix/vmaf/blob/master/libvmaf/README.md),
you can use it with the encoder:

~~~
    $ cmake path/to/avm -DCONFIG_TUNE_VMAF=1
~~~

Please note that the default VMAF model
will be used unless you set the following flag when running the encoder:

~~~
    # --vmaf-model-path=path/to/model
~~~

## Vulkan compute backend (fork)

An optional, self-contained compute backend for the encoder, written in **C11
against the Vulkan API directly** — no CUDA, no wgpu, no C++ in the module. It
is **off by default** and **not wired into the encoder** yet; what exists is a
small set of pixel kernels that are already proven bit-identical to AVM's C
reference code.

Everything lives in [`avm_gpu/`](avm_gpu/README.md); that file is the module
documentation. This section is the short version.

### Status

| Check | Contract | Gate |
|---|---|---|
| high-bitdepth SAD (`sad.comp`) | matches `avm_highbd_sadWxH_c`, 9 block sizes 4×4…64×64, host-visible and device-local arena placement | `avm_gpu_sad_test`, `avm_gpu_sad_test_arena` |
| variance / SSE (`variance.comp`) | matches `avm_highbd_{8,10,12}_varianceWxH_c`, 9 block sizes × 3 bitdepths, both upload layouts and both buffer placements | `avm_gpu_var_test`, `avm_gpu_var_test_packed`, `avm_gpu_var_test_arena` |
| variance seam (memo + block-area threshold) | the encoder's `highbd_variance64()` seam returns the reference `(sse, sum)` for every block size, memo hits included | `avm_gpu_var_hook_test` |
| intra DC/V/H/DC_LEFT/DC_TOP/DC_128/SMOOTH/SMOOTH_V/SMOOTH_H/PAETH (`intra.comp`) | matches `avm_highbd_*_predictor_*_c`, 10 modes × 4x4…64x64, plus availability rule and clamped edges; the call site in `av2_build_intra_predictors_high_default()` (`AV2_GPU_INTRA=1`) predicts SMOOTH*/PAETH and square DC with the CPU md5 (491,431 blocks in the gate) | `avm_gpu_intra_test`, `avm_gpu_encoder_parity` |
| directional intra zones 1-3 (`intra_dir.comp`) | matches `av2_highbd_dr_prediction_z{1,2,3}_c`, 6 angle configs × 5 sizes × mrl 0-2 | `avm_gpu_intra_test` |
| inverse DCT + ADST + IDTX + FLIPADST + DDT, 4…64 (`itxfm.comp`) | matches `inv_txfm_c()` for DCT on 16 sizes, all 9 64-wide/tall sizes (32-sample 1D pass + 2× replicate) and ADST/IDTX/FLIPADST/DDT type and size combinations, 100 cases; the call site in `av2_highbd_inv_txfm_add_master()` (`AV2_GPU_ITXFM=1`) runs 97,726 / 200,239 / 367,124 blocks at cpu-used 8 / 4 / 2 with the CPU md5s | `avm_gpu_itxfm_test`, `avm_gpu_encoder_parity` |
| compound inter over a scaled reference (`inter.comp`, `vulkan_hook_inter.c`) | `av2_highbd_convolve_2d_scale_c()` with `is_compound` (conv buffer, weighted/plain average, round offset) matches the reference over the seam reproducer's scaled compound cases; a key-frame-resized stream encodes to the CPU md5 with 12,166 compound blocks over a scaled reference | `avm_gpu_inter_seam_test`, `avm_gpu_encoder_parity` |
| compound inter, CWP family (`cwpconv.comp`, `vulkan_hook_inter.c`) | all four shapes (`2d`, `x`, `y`, `copy`) of `av2_highbd_cwp_convolve_*_c()`: high-precision buffer plus even/uneven weighted average, 24 kernel cases (2 bitdepths × 2 weight modes × 6 subpel, reference dispatched by phase) and seam cases; the seam takes every shape, scaled or not, and the encoder prices 50,160 compound blocks with the CPU run's md5 | `avm_gpu_inter_test`, `avm_gpu_inter_seam_test`, `avm_gpu_encoder_parity` |
| inter seam at block level (`vulkan_hook_inter.c`) | 400 comparisons of `avm_gpu_hook_inter()` against `av2_highbd_convolve_2d_{sr,scale}_c()` over freshly filled planes, blocks 4×4…64×64, scaled and unscaled; a decline counts as a failure and a failing case is replayed to distinguish a window bug from stale data | `avm_gpu_inter_seam_test` |
| inter subpel (unscaled + scaled), sr rounding, scaled 1/1024 units, compound blend and CWP (`inter.comp`, `compound.comp`, `cwp.comp`) | matches `avm_highbd_convolve8_c()` any step 1…64, `av2_highbd_convolve_2d_sr_c()`, `av2_highbd_convolve_2d_scale_c()` (per-pixel 1/1024 phase), `avm_highbd_blend_a64_mask_c()` and `avm_highbd_cwp_c()`, 213 cases | `avm_gpu_inter_test` |
| warp affine luma + chroma, regular + extended + scaled, single and compound (`warp.comp`, `warp_ext.comp`, `warp_scaled.comp`) | matches `av2_highbd_warp_affine_c()`, `av2_ext_highbd_warp_affine_c()` and `av2_ext_highbd_warp_affine_scaled_c()` (3 reference/current size pairs × 3 matrices × 3 subsamplings × 4 block sizes), 36 compound cases (two different references, conv buffer and pixels, plain and weighted average) and 48 boundary-box cases (clipped per-8x8 limits, single and compound), 347 cases | `avm_gpu_warp_test` |
| generic deblock (`deblock.comp`) | matches `avm_highbd_lpf_{horizontal,vertical}_generic_c()` over 122,880 combinations of width (4/4…18/18), q/side thresholds (1…16 plus random pairs), lossless flags, 8 and 10 bit, and orientation, plus the windowed entry point the seam uploads; the gate fails if no case actually filtered | `avm_gpu_deblock_test` |
| generic deblock call site (`vulkan_hook_deblock.c`, `AV2_GPU_DEBLOCK=1`) | `av2_filter_block_plane_{vert,horz}()` and the TIP pass run one windowed dispatch per edge (only the 4×21 window the filter can read or write is uploaded, so a caller's plane or band extent cannot be disturbed); with `AV2_GPU_DEBLOCK_VERIFY=1` the C re-runs on the same window and reports zero differing samples; the noisy clip at cpu-used 8 (53,344 edges) and the motion clip at cpu-used 8/4/2 (4,934 / 3,492 / 4,300 edges) reproduce the CPU md5s, and `avmdec` with the GPU deblock reproduces the CPU decoder's md5 | `avm_gpu_encoder_parity` |
| CDEF block filter (`cdef.comp`) | matches `av2_cdef_filter_16_{0,1,2,3}_c`, 5760 combinations of direction/strength/damping/shift/variant | `avm_gpu_cdef_test` |
| CDEF call site (`vulkan_hook_cdef.c`, `AV2_GPU_CDEF=1`) | `av2_cdef_filter_fb()` runs one dispatch per filter block, uploading only the block's 2/8-sample bordered window; with `AV2_GPU_CDEF_VERIFY=1` the C wrapper re-runs on the same input and reports zero differing samples (214,272 blocks on the noisy clip); the noisy clip at cpu-used 8 and the motion clip at cpu-used 4/2 reproduce the CPU md5s, and `avmdec` with the GPU CDEF reproduces the CPU decoder's md5 | `avm_gpu_encoder_parity` |
| CCSO block pass (`ccso.comp`) | matches `ccso_filter_block_hbd_wo_buf_c()`, 1008 combinations of support/classifier/thresholds/bands/block size | `avm_gpu_ccso_test` |
| CCSO call site (`vulkan_hook_ccso.c`, `AV2_GPU_CCSO=1`) | `av2_apply_ccso_filter_for_row()` and the tiled path of `apply_ccso_filter()` run one staged window per processing unit; the seam gate drives the hook against the reference over the encoder's geometry (192-wide padded luma reference, 32×32 units including frame edges, chroma reading the subsampled reference), 2,268 cases, and the noisy clip encodes to the CPU md5 with 66 blocks and the CPU decoder md5 | `avm_gpu_ccso_seam_test`, `avm_gpu_encoder_parity` |
| loop-restoration call site (`vulkan_hook_lr.c`, `AV2_GPU_LR=1`) | one restoration procunit through the kernels matches `apply_pc_wiener_highbd()`, 6 procunit shapes (64/32/partial) × 2 bitdepths × tx-skip on/off, with the caller's MI-based class/tskip strides; end to end, the noisy clip that selects PC-Wiener encodes to the CPU run's md5 with 150 procunits on the GPU | `avm_gpu_lr_hook_test`, `avm_gpu_encoder_parity` |
| PC-Wiener restoration, classification + filtering (`pcwcls.comp`, `pcwiener.comp`) | end to end against `apply_pc_wiener_highbd()`: class ids and the restored plane, 2 bitdepths × tx-skip on/off | `avm_gpu_pcwiener_test` |
| PC-Wiener restoration filter (`pcwiener.comp`) | matches `av2_convolve_symmetric_highbd_c()` with AVM's own filter banks, class→filter selector and `wienerns_simd_config_y` tap geometry; 2 geometries (incl. partial 4×4 edge blocks) × 4 filter sets × 2 bitdepths, plane borders untouched | `avm_gpu_pcwiener_test` |
| forward transform call site (`av2_fwd_txfm()`, `AV2_GPU_FWD_TXFM=1`) | every primary transform type at every size up to 64x64 (DCT2 folded butterflies, identity scaling, DST7/DCT8/DDTX/FDDT matrices, DDT substitution, 64-axis box downsample); 16 types × 15 sizes × DDT on/off match `fwd_txfm_c()`, and the gate encodes to the CPU run's md5 with 109,325 transforms on the GPU | `avm_gpu_txfm_test`, `avm_gpu_encoder_parity` |
| masked inter-inter compound call site (`build_masked_compound_no_round()`, `AV2_GPU_INTER=1`) | d16 mode of `kernels/compound.comp` mirrors `avm_highbd_blend_a64_d16_mask_c`; 4,433 / 9,828 / 29,740 blocks on the GPU at cpu-used 8 / 4 / 2 with the CPU md5s | `avm_gpu_inter_seam_test`, `avm_gpu_encoder_parity` |
| warped (affine) prediction call site (`vulkan_hook_inter.c`, `AV2_GPU_INTER=1`) | `av2_make_inter_predictor()`'s WARP_PRED branch runs the affine / extended-range (with its boundary box) / scaled kernel, single or compound; no warp is left to the CPU ("warped C" 0 in every gate stream); the half-pel clip encodes to the CPU run's md5 with 4,701 warped blocks (77,403 at cpu-used 2), and the resized-reference streams with 15,575 / 12,479 scaled warped blocks | `avm_gpu_encoder_parity` |
| masked (inter-intra/wedge) blend call site (`vulkan_hook_inter.c`, `AV2_GPU_INTER=1`) | `combine_interintra_highbd()` runs `compound.comp`; the cpu-used 2 gate encodes to the CPU run's md5 with 21,359 masked blends on the GPU | `avm_gpu_encoder_parity` |
| inter prediction call site (`vulkan_hook_inter.c`, `AV2_GPU_INTER=1`) | `av2_make_inter_predictor()` runs the sr kernel for single-reference 8-tap subpel, scaled and unscaled; the 5-frame half-pel clip encodes to the CPU run's md5 with 66,020 blocks on the GPU, and `--resize-mode=1` streams at 16:9 and 4:3 reproduce the CPU md5 (187k / 102k blocks) | `avm_gpu_encoder_parity`, `avm_gpu_inter_seam_test` |
| encoder integration (batched motion search + variance) | `avmenc` with `AV2_GPU_SAD=1` and with `AV2_GPU_VAR=1` both produce the CPU run's **md5**, and must price blocks; the same gate also runs with the search lookahead disabled | `avm_gpu_encoder_parity`, `avm_gpu_encoder_parity_no_lookahead` |

Both gates compare **every block** against the reference on random planes with
random strides and candidate positions — no tolerance, no PSNR fallback. A
machine with no Vulkan device returns exit code 77, so the tests skip instead of
failing.

Branch: `vulkan`. Base: upstream `main` at `1cfda4fa`.

### Build

The backend is gated by the build option `AV2_ENABLE_VULKAN`, which defaults to
`OFF`. It needs the Vulkan loader plus `glslangValidator` (package `glslang`) to
turn the GLSL kernels into SPIR-V:

~~~
    $ cmake -S . -B build-vk -G Ninja \
          -DAV2_ENABLE_VULKAN=1 -DENABLE_TESTS=1 -DENABLE_EXAMPLES=0
    $ cmake --build build-vk --target avmenc avm_gpu_sad_test avm_gpu_var_test
    $ ctest --test-dir build-vk -R avm_gpu_
~~~

To encode through the Vulkan SAD path (off by default, CPU fallback if no
device is found):

~~~
    $ AV2_GPU_SAD=1 AV2_GPU_SAD_STATS=1 ./build-vk/avmenc --limit=3 --good \
          --cpu-used=8 -o out.ivf clip.y4m
~~~

On macOS also install MoltenVK and point the loader at its ICD:

~~~
    $ brew install molten-vk vulkan-loader glslang
    $ export DYLD_LIBRARY_PATH=/opt/homebrew/lib
    $ export VK_DRIVER_FILES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json
~~~

`CONFIG_ML_PART_SPLIT=0 -DCONFIG_DIP_EXT_PRUNING=0` drops the vendored
TensorFlow-lite dependency (it is only reachable through those two encoder
experiments) and makes a gate build take seconds instead of minutes. It does
not change any kernel result.

### Design notes

- **Portable subset.** Kernels use shared memory, plain u32 arithmetic and a
  256-thread workgroup: one workgroup prices one candidate block. Buffers are
  `std430` storage buffers; planes are uploaded once with
  `avm_gpu_*_set_planes()` and stay resident.
- **No 64-bit shader integers.** SSE is carried as an (hi, lo) u32 pair and
  reduced with explicit carry; the bitdepth scaling and the
  `sse - sum²/(w·h)` formula run on the host in 64-bit C, copied verbatim from
  `avm_dsp/variance.c`.
- **M0 data model.** Each 16-bit sample occupies one 32-bit word, which avoids
  `VK_KHR_16bit_storage`; with `AVM_GPU_PACK16=1` SAD, variance and the forward
  DCT instead pack two samples per word. Buffers are host-visible and coherent
  by default; `AV2_GPU_ARENA=1` puts SAD and variance in a device-local arena
  with staging copies, while the other modules still use the host-visible
  path.

### Not claimed

These are **correctness** results, not speed. On an Apple M4 through MoltenVK
the integration gate encodes a 3-frame 64×64 clip:

| configuration | wall time | bitstream |
|---|---:|---|
| CPU SAD | 1.86 s | md5 `cd833e6f...` |
| `AV2_GPU_SAD=single`, one dispatch per call | 440.5 s | md5 `cd833e6f...` |
| `AV2_GPU_SAD=1`, whole-step batching + residency + search lookahead (51,403 dispatches, 52 blocks/dispatch) | **12.3 s** | md5 `cd833e6f...` |
| same + `AVM_GPU_PACK16=1` (packed 16-bit storage) | 35.7 s | md5 `cd833e6f...` |

Batching plus packed 16-bit storage bought 12.3x, but the GPU path is still
~19x slower than the CPU here: each dispatch costs ~0.2 ms through MoltenVK and
the block is re-packed per call. The remaining work is structural — price whole
search steps, then keep frames resident in a device-local arena. Entropy coding, CDF state, headers
and rate control stay on the CPU by design.

## Testing

### Testing basics

There are several methods of testing the AVM codec. All of these methods require
the presence of the AVM source code and a working build of the AVM library and
applications.

#### 1. Unit tests:

The unit tests can be run at build time:

~~~
    # Before running the make command the LIBAVM_TEST_DATA_PATH environment
    # variable should be set to avoid downloading the test files to the
    # cmake build configuration directory.
    $ cmake path/to/avm
    # Note: The AVM CMake build creates many test targets. Running make
    # with multiple jobs will speed up the test run significantly.
    $ make runtests
~~~

#### 2. Example tests:

The example tests require a bash shell and can be run in the following manner:

~~~
    # See the note above about LIBAVM_TEST_DATA_PATH above.
    $ cmake path/to/avm
    $ make
    # It's best to build the testdata target using many make jobs.
    # Running it like this will verify and download (if necessary)
    # one at a time, which takes a while.
    $ make testdata
    $ path/to/avm/test/examples.sh --bin-path examples
~~~

### Downloading the test data

The fastest and easiest way to obtain the test data is to use CMake to generate
a build using the Unix Makefiles generator, and then to build only the testdata
rule:

~~~
    $ cmake path/to/avm -G "Unix Makefiles"
    # 28 is used because there are 28 test files as of this writing.
    $ make -j28 testdata
~~~

The above make command will only download and verify the test data.

### Sharded testing

The AVM codec library unit tests are built upon gtest which supports sharding of test jobs.
Sharded tests can be achieved as follows for example:

~~~
   # Set the environment variable GTEST_TOTAL_SHARDS to control the number of
   # shards.
   $ export GTEST_TOTAL_SHARDS=10
   # (GTEST shard indexing is 0 based).
   $ seq 0 $(( $GTEST_TOTAL_SHARDS - 1 )) \
       | xargs -n 1 -P 0 -I{} env GTEST_SHARD_INDEX={} ./test_libavm
~~~

To create a test shard for each CPU core available on the current system set
`GTEST_TOTAL_SHARDS` to the number of CPU cores on your system minus one.
The maximum number of test targets that can run concurrently is determined by
the number of CPUs on the system where the build is configured as detected by
CMake. A system with 24 cores can run 24 test shards using a value of 24 with
the `-j` parameter. When CMake is unable to detect the number of cores 10 shards
is the default maximum value.

### Common Test Conditions (CTC)

CTC document can be found [here](https://aomedia.org/docs/CWG-G082_AV2_CTC_v9.pdf) and
CTC sequences can be obtained from [here](https://media.xiph.org/video/aomctc/test_set/).

## Coding style

We are using the Google C Coding Style defined by the
[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

The coding style used by this project is enforced with `clang-format` and
`cmake-format`.

- `clang-format` can be installed with your system's package manager, or directly
from [llvm.org](http://llvm.org/releases/download.html). For best results, your
version should match the one used by Github Actions, noted as a comment in
[.clang-format](https://github.com/AOMediaCodec/avm/blob/main/.clang-format).

- `cmake-format` can be obtained by installing
[cmakelang](https://pypi.org/project/cmakelang/). Again, for best results, your
version should match the one used by Github Actions, noted as a comment in
[.cmake-format.py](https://github.com/AOMediaCodec/avm/blob/main/.cmake-format.py)


Before pushing changes for review, format your code using the tools above.

We recommend automating the formatting by adding a
[pre-commit hook](https://git-scm.com/docs/githooks#_pre_commit) at
`.git/hooks/pre-commit`. An example pre-commit hook is provided below:

~~~
#!/bin/bash
# Replace 'VERSION' below to match clang-format version in `.clang-format` file.
CLANG_FORMAT_PATH=/usr/lib/llvm-VERSION/bin
CMAKE_FORMAT_PATH=${HOME}/.local/bin/

echo "Applying clang-format ..."
for file in $(git diff-index --cached --name-only HEAD -- "*.[hc]pp" "*.cc" "*.[ch]") ; do
  ${CLANG_FORMAT_PATH}/clang-format -i --style=file ${file}
  git add ${file}
  echo "Formatted file: $file"
done
echo "Done."

echo "Applying cmake-format ..."
for file in $(git diff-index --cached --name-only HEAD -- '*.cmake' CMakeLists.txt) ; do
  ${CMAKE_FORMAT_PATH}/cmake-format -i ${file}
  git add ${file}
  echo "Formatted file: $file"
done
echo "Done."
~~~

## Submitting patches

We manage the submission of patches using Github's
[pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) process.
This tool implements a workflow on top of the Git version control system to ensure that
all changes get peer reviewed and tested prior to their distribution.
If you are not able to submit an PR, please contact SW coordinators to make sure necessary
contributor agreements are signed for the AOMedia Project.

- Follow the one-time set-up steps as detailed [here](https://github.com/AOMediaCodec/avm/wiki/AVM:-Software-Development-Workflow#1-one-time-setup).
- For pushing your code modifications, follow the steps detailed [here](https://github.com/AOMediaCodec/avm/wiki/AVM:-Software-Development-Workflow#2-develop-a-tool--feature--bugfix-in-your-fork).
- Once the code is pushed into a branch, create a pull request as detailed [here](https://github.com/AOMediaCodec/avm/wiki/AVM:-Software-Development-Workflow#3-create-a-pull-request-pr).
    - The code review, approval and Github Actions will be initiated after the PR is created.
    - Once the PR is approved, the software co-ordinators will merge the PR to the main branch.

Follow the pull request page to check the status of the changes, review comments etc.

## Branches

- `main` branch is the default and stable branch.
- `av2-enc` is an active development branch for encoder optimization work. It
  is generally ahead of main branch containing more recent optimizations but
  maybe less stable.

Changes are brought in from `av2-enc` branch to `main` branch periodicaly after
more thorough testing.

## Support

This library is an open source project supported by its community.
Please email https://aomedia.org/contact/ for help.

## Bug reports

Bug reports can be filed in the Alliance for Open Media
[Github issue tracker](https://github.com/AOMediaCodec/avm/issues).
