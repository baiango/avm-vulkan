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
set(AVM_GPU_VAR_SPV "${AVM_GPU_SPV_DIR}/variance.spv")
set(AVM_GPU_TXFM_SPV "${AVM_GPU_SPV_DIR}/txfm.spv")
set(AVM_GPU_INTRA_SPV "${AVM_GPU_SPV_DIR}/intra.spv")

add_custom_command(
  OUTPUT "${AVM_GPU_SAD_SPV}"
  COMMAND "${AVM_GPU_GLSLANG}" -V
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels/sad.comp" -o "${AVM_GPU_SAD_SPV}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/sad.comp"
  COMMENT "avm_gpu: sad.comp -> sad.spv"
  VERBATIM)
add_custom_command(
  OUTPUT "${AVM_GPU_VAR_SPV}"
  COMMAND "${AVM_GPU_GLSLANG}" -V
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels/variance.comp" -o
          "${AVM_GPU_VAR_SPV}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/variance.comp"
  COMMENT "avm_gpu: variance.comp -> variance.spv"
  VERBATIM)
add_custom_command(
  OUTPUT "${AVM_GPU_TXFM_SPV}"
  COMMAND "${AVM_GPU_GLSLANG}" -V
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels/txfm.comp" -o
          "${AVM_GPU_TXFM_SPV}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/txfm.comp"
  COMMENT "avm_gpu: txfm.comp -> txfm.spv"
  VERBATIM)
add_custom_command(
  OUTPUT "${AVM_GPU_INTRA_SPV}"
  COMMAND "${AVM_GPU_GLSLANG}" -V
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels/intra.comp" -o
          "${AVM_GPU_INTRA_SPV}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/intra.comp"
  COMMENT "avm_gpu: intra.comp -> intra.spv"
  VERBATIM)
add_custom_target(avm_gpu_shaders DEPENDS "${AVM_GPU_SAD_SPV}"
                                          "${AVM_GPU_VAR_SPV}"
                                          "${AVM_GPU_TXFM_SPV}"
                                          "${AVM_GPU_INTRA_SPV}")

add_library(
  avm_gpu STATIC
  "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_gpu.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/pipeline.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_sad.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_var.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_txfm.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_intra.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_hook.c")
target_include_directories(avm_gpu PUBLIC "${AVM_ROOT}")
target_compile_definitions(avm_gpu
                           PRIVATE AVM_GPU_SAD_SPV_PATH="${AVM_GPU_SAD_SPV}"
                                   AVM_GPU_VAR_SPV_PATH="${AVM_GPU_VAR_SPV}"
                                   AVM_GPU_TXFM_SPV_PATH="${AVM_GPU_TXFM_SPV}"
                                   AVM_GPU_INTRA_SPV_PATH="${AVM_GPU_INTRA_SPV}")
# Vulkan structs are initialised by sType; the remaining fields are required
# to be zero, so the missing-field diagnostic is noise here.
if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(avm_gpu PRIVATE -Wno-missing-field-initializers)
endif()
target_link_libraries(avm_gpu PUBLIC Vulkan::Vulkan)
add_dependencies(avm_gpu avm_gpu_shaders)

# Encoder integration seam. avm_dsp/sad.c calls avm_gpu_hook_sad() when it is
# built with AVM_GPU_SAD_HOOK; the hook is inert unless AV2_GPU_SAD=1, so the
# default encode path is byte-identical.
foreach (avm_gpu_hook_target avm_dsp_common avm_dsp_encoder avm_av2_common
                             avm_av2_encoder avm_av2_decoder)
  if(TARGET ${avm_gpu_hook_target})
    target_compile_definitions(${avm_gpu_hook_target} PRIVATE AVM_GPU_SAD_HOOK=1)
  endif()
endforeach()
if(TARGET avm)
  target_link_libraries(avm PUBLIC avm_gpu)
endif()

if(ENABLE_TESTS)
  add_executable(avm_gpu_sad_test
                 "${CMAKE_CURRENT_SOURCE_DIR}/test/avm_gpu_sad_test.c")
  target_link_libraries(avm_gpu_sad_test PRIVATE avm)
  add_test(NAME avm_gpu_sad_test COMMAND avm_gpu_sad_test)
  # 77 = "no Vulkan device", so machines without a GPU skip instead of failing.
  set_tests_properties(avm_gpu_sad_test PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(avm_gpu_var_test
                 "${CMAKE_CURRENT_SOURCE_DIR}/test/avm_gpu_var_test.c")
  target_link_libraries(avm_gpu_var_test PRIVATE avm)
  add_test(NAME avm_gpu_var_test COMMAND avm_gpu_var_test)
  set_tests_properties(avm_gpu_var_test PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(avm_gpu_txfm_test
                 "${CMAKE_CURRENT_SOURCE_DIR}/test/avm_gpu_txfm_test.c")
  target_link_libraries(avm_gpu_txfm_test PRIVATE avm)
  add_test(NAME avm_gpu_txfm_test COMMAND avm_gpu_txfm_test)
  set_tests_properties(avm_gpu_txfm_test PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(avm_gpu_intra_test
                 "${CMAKE_CURRENT_SOURCE_DIR}/test/avm_gpu_intra_test.c")
  target_link_libraries(avm_gpu_intra_test PRIVATE avm)
  add_test(NAME avm_gpu_intra_test COMMAND avm_gpu_intra_test)
  set_tests_properties(avm_gpu_intra_test PROPERTIES SKIP_RETURN_CODE 77)

  # End-to-end: encode the same clip with the CPU and the Vulkan SAD path and
  # require identical bitstreams.
  if(TARGET avmenc)
    add_test(
      NAME avm_gpu_encoder_parity
      COMMAND sh "${CMAKE_CURRENT_SOURCE_DIR}/test/run_encoder_parity.sh"
              "$<TARGET_FILE:avmenc>")
    set_tests_properties(avm_gpu_encoder_parity PROPERTIES SKIP_RETURN_CODE 77)
  endif()
endif()
