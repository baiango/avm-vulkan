#
# Copyright (c) 2021, Alliance for Open Media. All rights reserved
#
# This source code is subject to the terms of the BSD 3-Clause Clear License and
# the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
# License was not distributed with this source code in the LICENSE file, you
# can obtain it at aomedia.org/license/software-license/bsd-3-c-c/.  If the
# Alliance for Open Media Patent License 1.0 was not distributed with this
# source code in the PATENTS file, you can obtain it at
# aomedia.org/license/patent-license/.
#
if(NOT AV2_ENABLE_VULKAN)
  return()
endif()

find_package(Vulkan REQUIRED)

find_program(AVM_GPU_GLSLANG NAMES glslangValidator glslang)
if(NOT AVM_GPU_GLSLANG)
  message(
    FATAL_ERROR
      "AV2_ENABLE_VULKAN needs glslangValidator (glslang) to build SPIR-V.")
endif()

set(AVM_GPU_SPV_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(AVM_GPU_SAD_SPV "${AVM_GPU_SPV_DIR}/sad.spv")

add_custom_command(
  OUTPUT "${AVM_GPU_SAD_SPV}"
  COMMAND "${AVM_GPU_GLSLANG}" -V
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels/sad.comp" -o "${AVM_GPU_SAD_SPV}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/sad.comp"
  COMMENT "avm_gpu: sad.comp -> sad.spv"
  VERBATIM)
add_custom_target(avm_gpu_shaders DEPENDS "${AVM_GPU_SAD_SPV}")

add_library(
  avm_gpu STATIC
  "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_gpu.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_sad.c")
target_include_directories(avm_gpu PUBLIC "${AVM_ROOT}")
target_compile_definitions(avm_gpu
                           PRIVATE AVM_GPU_SAD_SPV_PATH="${AVM_GPU_SAD_SPV}")
target_link_libraries(avm_gpu PUBLIC Vulkan::Vulkan)
add_dependencies(avm_gpu avm_gpu_shaders)

if(ENABLE_TESTS)
  add_executable(avm_gpu_sad_test
                 "${CMAKE_CURRENT_SOURCE_DIR}/test/avm_gpu_sad_test.c")
  target_link_libraries(avm_gpu_sad_test PRIVATE avm_gpu avm)
  add_test(NAME avm_gpu_sad_test COMMAND avm_gpu_sad_test)
  # 77 = "no Vulkan device", so machines without a GPU skip instead of failing.
  set_tests_properties(avm_gpu_sad_test PROPERTIES SKIP_RETURN_CODE 77)
endif()
