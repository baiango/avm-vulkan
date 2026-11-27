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
set(AVM_GPU_INTRA_DIR_SPV "${AVM_GPU_SPV_DIR}/intra_dir.spv")
set(AVM_GPU_ITXFM_SPV "${AVM_GPU_SPV_DIR}/itxfm.spv")
set(AVM_GPU_INTER_SPV "${AVM_GPU_SPV_DIR}/inter.spv")
set(AVM_GPU_DEBLOCK_SPV "${AVM_GPU_SPV_DIR}/deblock.spv")
set(AVM_GPU_CDEF_SPV "${AVM_GPU_SPV_DIR}/cdef.spv")
set(AVM_GPU_CCSO_SPV "${AVM_GPU_SPV_DIR}/ccso.spv")
set(AVM_GPU_COMPOUND_SPV "${AVM_GPU_SPV_DIR}/compound.spv")
set(AVM_GPU_WARP_SPV "${AVM_GPU_SPV_DIR}/warp.spv")
set(AVM_GPU_WARP_EXT_SPV "${AVM_GPU_SPV_DIR}/warp_ext.spv")
set(AVM_GPU_CWP_SPV "${AVM_GPU_SPV_DIR}/cwp.spv")
set(AVM_GPU_PCWIENER_SPV "${AVM_GPU_SPV_DIR}/pcwiener.spv")
set(AVM_GPU_PCWCLS_SPV "${AVM_GPU_SPV_DIR}/pcwcls.spv")

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
add_custom_command(
  OUTPUT "${AVM_GPU_INTRA_DIR_SPV}"
  COMMAND "${AVM_GPU_GLSLANG}" -V
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels/intra_dir.comp" -o
          "${AVM_GPU_INTRA_DIR_SPV}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/intra_dir.comp"
  COMMENT "avm_gpu: intra_dir.comp -> intra_dir.spv"
  VERBATIM)
add_custom_command(
  OUTPUT "${AVM_GPU_ITXFM_SPV}"
  COMMAND "${AVM_GPU_GLSLANG}" -V
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels/itxfm.comp" -o
          "${AVM_GPU_ITXFM_SPV}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/itxfm.comp"
  COMMENT "avm_gpu: itxfm.comp -> itxfm.spv"
  VERBATIM)
add_custom_command(
  OUTPUT "${AVM_GPU_PCWIENER_SPV}"
  COMMAND "${AVM_GPU_GLSLANG}" -V
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels/pcwiener.comp" -o
          "${AVM_GPU_PCWIENER_SPV}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/pcwiener.comp"
  COMMENT "avm_gpu: pcwiener.comp -> pcwiener.spv"
  VERBATIM)
add_custom_command(
  OUTPUT "${AVM_GPU_PCWCLS_SPV}"
  COMMAND "${AVM_GPU_GLSLANG}" -V
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels/pcwcls.comp" -o
          "${AVM_GPU_PCWCLS_SPV}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/pcwcls.comp"
  COMMENT "avm_gpu: pcwcls.comp -> pcwcls.spv"
  VERBATIM)
add_custom_command(
  OUTPUT "${AVM_GPU_INTER_SPV}"
  COMMAND "${AVM_GPU_GLSLANG}" -V
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels/inter.comp" -o
          "${AVM_GPU_INTER_SPV}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/inter.comp"
  COMMENT "avm_gpu: inter.comp -> inter.spv"
  VERBATIM)
add_custom_command(
  OUTPUT "${AVM_GPU_DEBLOCK_SPV}"
  COMMAND "${AVM_GPU_GLSLANG}" -V
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels/deblock.comp" -o
          "${AVM_GPU_DEBLOCK_SPV}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/deblock.comp"
  COMMENT "avm_gpu: deblock.comp -> deblock.spv"
  VERBATIM)
add_custom_command(
  OUTPUT "${AVM_GPU_CDEF_SPV}"
  COMMAND "${AVM_GPU_GLSLANG}" -V
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels/cdef.comp" -o "${AVM_GPU_CDEF_SPV}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/cdef.comp"
  COMMENT "avm_gpu: cdef.comp -> cdef.spv"
  VERBATIM)
add_custom_command(
  OUTPUT "${AVM_GPU_CCSO_SPV}"
  COMMAND "${AVM_GPU_GLSLANG}" -V
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ccso.comp" -o "${AVM_GPU_CCSO_SPV}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ccso.comp"
  COMMENT "avm_gpu: ccso.comp -> ccso.spv"
  VERBATIM)
add_custom_command(
  OUTPUT "${AVM_GPU_COMPOUND_SPV}"
  COMMAND "${AVM_GPU_GLSLANG}" -V
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels/compound.comp" -o
          "${AVM_GPU_COMPOUND_SPV}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/compound.comp"
  COMMENT "avm_gpu: compound.comp -> compound.spv"
  VERBATIM)
add_custom_command(
  OUTPUT "${AVM_GPU_WARP_SPV}"
  COMMAND "${AVM_GPU_GLSLANG}" -V
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels/warp.comp" -o "${AVM_GPU_WARP_SPV}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/warp.comp"
  COMMENT "avm_gpu: warp.comp -> warp.spv"
  VERBATIM)
add_custom_command(
  OUTPUT "${AVM_GPU_WARP_EXT_SPV}"
  COMMAND "${AVM_GPU_GLSLANG}" -V
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels/warp_ext.comp" -o
          "${AVM_GPU_WARP_EXT_SPV}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/warp_ext.comp"
  COMMENT "avm_gpu: warp_ext.comp -> warp_ext.spv"
  VERBATIM)
add_custom_command(
  OUTPUT "${AVM_GPU_CWP_SPV}"
  COMMAND "${AVM_GPU_GLSLANG}" -V
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels/cwp.comp" -o "${AVM_GPU_CWP_SPV}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/cwp.comp"
  COMMENT "avm_gpu: cwp.comp -> cwp.spv"
  VERBATIM)
add_custom_target(avm_gpu_shaders DEPENDS "${AVM_GPU_SAD_SPV}"
                                          "${AVM_GPU_VAR_SPV}"
                                          "${AVM_GPU_TXFM_SPV}"
                                          "${AVM_GPU_INTRA_SPV}"
                                          "${AVM_GPU_INTRA_DIR_SPV}"
                                          "${AVM_GPU_ITXFM_SPV}"
                                          "${AVM_GPU_INTER_SPV}"
                                          "${AVM_GPU_DEBLOCK_SPV}"
                                          "${AVM_GPU_CDEF_SPV}"
                                          "${AVM_GPU_CCSO_SPV}"
                                          "${AVM_GPU_COMPOUND_SPV}"
                                          "${AVM_GPU_WARP_SPV}"
                                          "${AVM_GPU_WARP_EXT_SPV}"
                                          "${AVM_GPU_CWP_SPV}"
                                          "${AVM_GPU_PCWIENER_SPV}"
                                          "${AVM_GPU_PCWCLS_SPV}")

add_library(
  avm_gpu STATIC
  "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_gpu.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/pipeline.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/arena.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_sad.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_var.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_txfm.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_intra.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_itxfm.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_inter.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_warp.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_deblock.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_cdef.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_ccso.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/pcwiener.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_hook.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_hook_var.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_hook_lr.c")
target_include_directories(avm_gpu PUBLIC "${AVM_ROOT}")
target_compile_definitions(avm_gpu
                           PRIVATE AVM_GPU_SAD_SPV_PATH="${AVM_GPU_SAD_SPV}"
                                   AVM_GPU_VAR_SPV_PATH="${AVM_GPU_VAR_SPV}"
                                   AVM_GPU_TXFM_SPV_PATH="${AVM_GPU_TXFM_SPV}"
                                   AVM_GPU_INTRA_SPV_PATH="${AVM_GPU_INTRA_SPV}"
                                   AVM_GPU_INTRA_DIR_SPV_PATH="${AVM_GPU_INTRA_DIR_SPV}"
                                   AVM_GPU_ITXFM_SPV_PATH="${AVM_GPU_ITXFM_SPV}"
                                   AVM_GPU_INTER_SPV_PATH="${AVM_GPU_INTER_SPV}"
                                   AVM_GPU_DEBLOCK_SPV_PATH="${AVM_GPU_DEBLOCK_SPV}"
                                   AVM_GPU_CDEF_SPV_PATH="${AVM_GPU_CDEF_SPV}"
                                   AVM_GPU_CCSO_SPV_PATH="${AVM_GPU_CCSO_SPV}"
                                   AVM_GPU_COMPOUND_SPV_PATH="${AVM_GPU_COMPOUND_SPV}"
                                   AVM_GPU_WARP_SPV_PATH="${AVM_GPU_WARP_SPV}"
                                   AVM_GPU_WARP_EXT_SPV_PATH="${AVM_GPU_WARP_EXT_SPV}"
                                   AVM_GPU_CWP_SPV_PATH="${AVM_GPU_CWP_SPV}"
                                   AVM_GPU_PCWIENER_SPV_PATH="${AVM_GPU_PCWIENER_SPV}"
                                   AVM_GPU_PCWCLS_SPV_PATH="${AVM_GPU_PCWCLS_SPV}")
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
    target_compile_definitions(${avm_gpu_hook_target} PRIVATE AVM_GPU_HOOKS=1)
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
  # Same gate with a device-local arena for the buffers.
  add_test(NAME avm_gpu_sad_test_arena COMMAND avm_gpu_sad_test)
  set_tests_properties(
    avm_gpu_sad_test_arena
    PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "AV2_GPU_ARENA=1")

  # The encoder's variance seam: memo + block-area threshold paths.
  add_executable(avm_gpu_var_hook_test
                 "${CMAKE_CURRENT_SOURCE_DIR}/test/avm_gpu_var_hook_test.c")
  target_link_libraries(avm_gpu_var_hook_test PRIVATE avm)
  add_test(NAME avm_gpu_var_hook_test COMMAND avm_gpu_var_hook_test)
  set_tests_properties(avm_gpu_var_hook_test PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(avm_gpu_var_test
                 "${CMAKE_CURRENT_SOURCE_DIR}/test/avm_gpu_var_test.c")
  target_link_libraries(avm_gpu_var_test PRIVATE avm)
  add_test(NAME avm_gpu_var_test COMMAND avm_gpu_var_test)
  set_tests_properties(avm_gpu_var_test PROPERTIES SKIP_RETURN_CODE 77)
  # Same gate with the packed 16-bit upload layout.
  add_test(NAME avm_gpu_var_test_packed COMMAND avm_gpu_var_test)
  set_tests_properties(
    avm_gpu_var_test_packed
    PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "AVM_GPU_PACK16=1")
  add_test(NAME avm_gpu_var_test_arena COMMAND avm_gpu_var_test)
  set_tests_properties(
    avm_gpu_var_test_arena
    PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "AV2_GPU_ARENA=1;AVM_GPU_PACK16=1")

  add_executable(avm_gpu_txfm_test
                 "${CMAKE_CURRENT_SOURCE_DIR}/test/avm_gpu_txfm_test.c")
  target_link_libraries(avm_gpu_txfm_test PRIVATE avm)
  add_test(NAME avm_gpu_txfm_test COMMAND avm_gpu_txfm_test)
  set_tests_properties(avm_gpu_txfm_test PROPERTIES SKIP_RETURN_CODE 77)
  add_test(NAME avm_gpu_txfm_test_packed COMMAND avm_gpu_txfm_test)
  set_tests_properties(
    avm_gpu_txfm_test_packed
    PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "AVM_GPU_PACK16=1")

  add_executable(avm_gpu_intra_test
                 "${CMAKE_CURRENT_SOURCE_DIR}/test/avm_gpu_intra_test.c")
  target_link_libraries(avm_gpu_intra_test PRIVATE avm)
  add_test(NAME avm_gpu_intra_test COMMAND avm_gpu_intra_test)
  set_tests_properties(avm_gpu_intra_test PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(avm_gpu_pcwiener_test
                 "${CMAKE_CURRENT_SOURCE_DIR}/test/avm_gpu_pcwiener_test.c")
  target_link_libraries(avm_gpu_pcwiener_test PRIVATE avm)
  add_test(NAME avm_gpu_pcwiener_test COMMAND avm_gpu_pcwiener_test)
  set_tests_properties(avm_gpu_pcwiener_test PROPERTIES SKIP_RETURN_CODE 77)

  # The loop-restoration seam as pc_wiener_stripe_highbd() calls it.
  add_executable(avm_gpu_lr_hook_test
                 "${CMAKE_CURRENT_SOURCE_DIR}/test/avm_gpu_lr_hook_test.c")
  target_link_libraries(avm_gpu_lr_hook_test PRIVATE avm)
  add_test(NAME avm_gpu_lr_hook_test COMMAND avm_gpu_lr_hook_test)
  set_tests_properties(avm_gpu_lr_hook_test PROPERTIES SKIP_RETURN_CODE 77)

  # The filter stage on its own: all four filter sets, two geometries.
  add_executable(avm_gpu_pcwiener_filter_test
                 "${CMAKE_CURRENT_SOURCE_DIR}/test/avm_gpu_pcwiener_filter_test.c")
  target_link_libraries(avm_gpu_pcwiener_filter_test PRIVATE avm)
  add_test(NAME avm_gpu_pcwiener_filter_test
           COMMAND avm_gpu_pcwiener_filter_test)
  set_tests_properties(avm_gpu_pcwiener_filter_test
                       PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(avm_gpu_itxfm_test
                 "${CMAKE_CURRENT_SOURCE_DIR}/test/avm_gpu_itxfm_test.c")
  target_link_libraries(avm_gpu_itxfm_test PRIVATE avm)
  add_test(NAME avm_gpu_itxfm_test COMMAND avm_gpu_itxfm_test)
  set_tests_properties(avm_gpu_itxfm_test PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(avm_gpu_inter_test
                 "${CMAKE_CURRENT_SOURCE_DIR}/test/avm_gpu_inter_test.c")
  target_link_libraries(avm_gpu_inter_test PRIVATE avm)
  add_test(NAME avm_gpu_inter_test COMMAND avm_gpu_inter_test)
  set_tests_properties(avm_gpu_inter_test PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(avm_gpu_deblock_test
                 "${CMAKE_CURRENT_SOURCE_DIR}/test/avm_gpu_deblock_test.c")
  target_link_libraries(avm_gpu_deblock_test PRIVATE avm)
  add_test(NAME avm_gpu_deblock_test COMMAND avm_gpu_deblock_test)
  set_tests_properties(avm_gpu_deblock_test PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(avm_gpu_cdef_test
                 "${CMAKE_CURRENT_SOURCE_DIR}/test/avm_gpu_cdef_test.c")
  target_link_libraries(avm_gpu_cdef_test PRIVATE avm)
  add_test(NAME avm_gpu_cdef_test COMMAND avm_gpu_cdef_test)
  set_tests_properties(avm_gpu_cdef_test PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(avm_gpu_ccso_test
                 "${CMAKE_CURRENT_SOURCE_DIR}/test/avm_gpu_ccso_test.c")
  target_link_libraries(avm_gpu_ccso_test PRIVATE avm)
  add_test(NAME avm_gpu_ccso_test COMMAND avm_gpu_ccso_test)
  set_tests_properties(avm_gpu_ccso_test PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(avm_gpu_warp_test
                 "${CMAKE_CURRENT_SOURCE_DIR}/test/avm_gpu_warp_test.c")
  target_link_libraries(avm_gpu_warp_test PRIVATE avm)
  add_test(NAME avm_gpu_warp_test COMMAND avm_gpu_warp_test)
  set_tests_properties(avm_gpu_warp_test PROPERTIES SKIP_RETURN_CODE 77)

  # End-to-end: encode the same clip with the CPU and the Vulkan SAD path and
  # require identical bitstreams.
  if(TARGET avmenc)
    add_test(
      NAME avm_gpu_encoder_parity
      COMMAND sh "${CMAKE_CURRENT_SOURCE_DIR}/test/run_encoder_parity.sh"
              "$<TARGET_FILE:avmenc>")
    set_tests_properties(avm_gpu_encoder_parity PROPERTIES SKIP_RETURN_CODE 77)
    # Same gate with the speculative search lookahead turned off, so both
    # search shapes (one dispatch per step and one per two steps) are covered.
    add_test(
      NAME avm_gpu_encoder_parity_no_lookahead
      COMMAND
        sh "${CMAKE_CURRENT_SOURCE_DIR}/test/run_encoder_parity.sh"
        "$<TARGET_FILE:avmenc>")
    set_tests_properties(
      avm_gpu_encoder_parity_no_lookahead
      PROPERTIES SKIP_RETURN_CODE 77
                 ENVIRONMENT "AV2_GPU_SAD_LOOKAHEAD=0")
  endif()
endif()
