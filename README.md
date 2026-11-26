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
| high-bitdepth SAD (`sad.comp`) | matches `avm_highbd_sadWxH_c`, 9 block sizes 4×4…64×64 | `avm_gpu_sad_test` |
| variance / SSE (`variance.comp`) | matches `avm_highbd_{8,10,12}_varianceWxH_c`, 9 block sizes × 3 bitdepths | `avm_gpu_var_test` |
| forward DCT 4x4 / 8x8 (`txfm.comp`) | matches `avm_fdct4x4_c` and `avm_highbd_fdct8x8_c` bit-for-bit (64-bit products emulated) | `avm_gpu_txfm_test` |
| intra DC/V/H/DC_LEFT/DC_TOP/DC_128/SMOOTH/SMOOTH_V/SMOOTH_H/PAETH (`intra.comp`) | matches `avm_highbd_*_predictor_*_c`, 10 modes × 4x4…64x64, plus availability rule and clamped edges | `avm_gpu_intra_test` |
| directional intra zones 1-3 (`intra_dir.comp`) | matches `av2_highbd_dr_prediction_z{1,2,3}_c`, 6 angle configs × 5 sizes × mrl 0-2 | `avm_gpu_intra_test` |
| inverse DCT + ADST + IDTX + FLIPADST, 4…32 (`itxfm.comp`) | matches `inv_txfm_c()` for DCT on 16 sizes plus 62 ADST/IDTX/FLIPADST type/size combinations | `avm_gpu_itxfm_test` |
| inter 8-tap subpel (`inter.comp`) | matches `avm_highbd_convolve8_c()`, 6 subpel positions × 5 block sizes, whole-pel included | `avm_gpu_inter_test` |
| generic deblock (`deblock.comp`) | matches `avm_highbd_lpf_{horizontal,vertical}_generic_c()`, 378 combinations of width/threshold/flags/orientation | `avm_gpu_deblock_test` |
| CDEF block filter (`cdef.comp`) | matches `av2_cdef_filter_16_{0,1,2,3}_c`, 5760 combinations of direction/strength/damping/shift/variant | `avm_gpu_cdef_test` |
| CCSO block pass (`ccso.comp`) | matches `ccso_filter_block_hbd_wo_buf_c()`, 1008 combinations of support/classifier/thresholds/bands/block size | `avm_gpu_ccso_test` |
| encoder integration (`vulkan_hook.c`, batched motion search) | `avmenc` with `AV2_GPU_SAD=1` produces a bitstream with the **same md5** as the CPU run and must price >1 block per dispatch | `avm_gpu_encoder_parity` |

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
  `VK_KHR_16bit_storage`; M1 replaces this with `ushort` storage behind the same
  signatures. Buffers are host-visible and coherent; the arena/device-local
  staging step is not implemented yet.

### Not claimed

These are **correctness** results, not speed. On an Apple M4 through MoltenVK
the integration gate encodes a 3-frame 64×64 clip:

| configuration | wall time | bitstream |
|---|---:|---|
| CPU SAD | 1.86 s | md5 `cd833e6f...` |
| `AV2_GPU_SAD=single`, one dispatch per call | 440.5 s | md5 `cd833e6f...` |
| `AV2_GPU_SAD=1`, batched, 4 candidates per dispatch (178,249 dispatches) | 73.9 s | md5 `cd833e6f...` |
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
