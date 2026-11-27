#!/bin/sh
#
# Copyright (c) 2021, Alliance for Open Media. All rights reserved
#
# This source code is subject to the terms of the BSD 3-Clause Clear License
# and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
# License was not distributed with this source code in the LICENSE file, you
# can obtain it at aomedia.org/license/software-license/bsd-3-c-c/.  If the
# Alliance for Open Media Patent License 1.0 was not distributed with this
# source code in the PATENTS file, you can obtain it at
# aomedia.org/license/patent-license/.
#
# Encoder parity gate for the Vulkan SAD hook.
#
# Encodes the same clip twice -- CPU SAD, then Vulkan SAD (AV2_GPU_SAD=1) --
# and requires byte-identical bitstreams. It also fails if the GPU path never
# ran, so a silent CPU fallback cannot pass this test.
#
# Usage: run_encoder_parity.sh /path/to/avmenc
# Exit codes: 0 pass, 1 mismatch/failure, 77 no Vulkan device (test skipped).

set -eu

AVMENC=${1:?usage: run_encoder_parity.sh /path/to/avmenc}

here=$(cd "$(dirname "$0")" && pwd)

# macOS: point the loader at MoltenVK if the caller has not already done so.
if [ -z "${VK_DRIVER_FILES:-}" ] && [ -f /opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json ]; then
  export VK_DRIVER_FILES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json
  DYLD_LIBRARY_PATH=/opt/homebrew/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}
  export DYLD_LIBRARY_PATH
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/avm_gpu_enc.XXXXXX")
trap 'rm -rf "$work"' EXIT

python3 "$here/make_test_clip.py" "$work/clip.y4m"

ARGS="--limit=3 --good --cpu-used=8 --passes=1 --threads=1"

"$AVMENC" $ARGS -o "$work/cpu.ivf" "$work/clip.y4m" >"$work/cpu.log" 2>&1 ||
  { echo "FAIL: CPU encode failed"; tail -20 "$work/cpu.log"; exit 1; }

if ! AV2_GPU_SAD=1 AV2_GPU_SAD_STATS=1 "$AVMENC" $ARGS -o "$work/gpu.ivf" \
    "$work/clip.y4m" >"$work/gpu.log" 2>&1; then
  if grep -q "no Vulkan device\|vkCreateInstance" "$work/gpu.log"; then
    echo "SKIP: no Vulkan device"
    exit 77
  fi
  echo "FAIL: GPU encode failed"
  tail -20 "$work/gpu.log"
  exit 1
fi

if ! grep -q "SAD hook enabled" "$work/gpu.log"; then
  echo "FAIL: the Vulkan SAD hook did not initialise (see gpu.log)"
  tail -20 "$work/gpu.log"
  exit 1
fi

stats=$(sed -n 's/.*avm_gpu: \([0-9][0-9]*\) SAD blocks in \([0-9][0-9]*\) dispatches.*/\1 \2/p' "$work/gpu.log")
calls=$(echo "$stats" | cut -d' ' -f1)
dispatches=$(echo "$stats" | cut -d' ' -f2)
if [ -z "$calls" ] || [ "$calls" -le 0 ]; then
  echo "FAIL: the GPU path priced no blocks (silent CPU fallback)"
  tail -20 "$work/gpu.log"
  exit 1
fi
if [ -z "$dispatches" ] || [ "$dispatches" -le 0 ]; then
  echo "FAIL: no dispatch statistics in the GPU log"
  exit 1
fi
if [ "$calls" -le "$dispatches" ]; then
  echo "FAIL: $calls blocks in $dispatches dispatches - the batched path did not run"
  exit 1
fi

# Second call site: the variance/SSE kernel feeding partition and mode RD.
if ! AV2_GPU_VAR=1 AV2_GPU_VAR_STATS=1 "$AVMENC" $ARGS -o "$work/var.ivf" \
    "$work/clip.y4m" >"$work/var.log" 2>&1; then
  if grep -q "no Vulkan device\|vkCreateInstance" "$work/txfm.log"; then
    echo "SKIP: no Vulkan device"
    exit 77
  fi
  echo "FAIL: GPU variance encode failed"
  tail -20 "$work/var.log"
  exit 1
fi
if ! grep -q "variance hook enabled" "$work/var.log"; then
  echo "FAIL: the Vulkan variance hook did not initialise"
  tail -20 "$work/var.log"
  exit 1
fi
tcalls=$(sed -n 's/.*avm_gpu: \([0-9][0-9]*\) variance blocks.*/\1/p' "$work/var.log")
if [ -z "$tcalls" ] || [ "$tcalls" -le 0 ]; then
  echo "FAIL: the GPU variance path priced no blocks"
  tail -20 "$work/var.log"
  exit 1
fi

# Third call site: single-reference subpel inter prediction, on a clip whose
# content moves by a fractional pixel so the subpel filter (not the integer
# copy path) is what the encoder actually uses.
python3 "$here/make_motion_clip.py" "$work/motion.y4m"
INTER_ARGS="--limit=5 --good --cpu-used=8 --passes=1 --threads=1"
if ! "$AVMENC" $INTER_ARGS -o "$work/motion_cpu.ivf" "$work/motion.y4m" \
    >"$work/motion_cpu.log" 2>&1; then
  echo "FAIL: CPU motion-clip encode failed"
  tail -20 "$work/motion_cpu.log"
  exit 1
fi
if ! AV2_GPU_INTER=1 AV2_GPU_FWD_TXFM=1 AV2_GPU_INTER_STATS=1 "$AVMENC" $INTER_ARGS \
    -o "$work/motion_gpu.ivf" "$work/motion.y4m" >"$work/inter.log" 2>&1; then
  if grep -q "no Vulkan device\|vkCreateInstance" "$work/inter.log"; then
    echo "SKIP: no Vulkan device"
    exit 77
  fi
  echo "FAIL: GPU inter encode failed"
  tail -20 "$work/inter.log"
  exit 1
fi
if ! grep -q "inter hook enabled" "$work/inter.log"; then
  echo "FAIL: the Vulkan inter hook did not initialise"
  tail -20 "$work/inter.log"
  exit 1
fi
icalls=$(sed -n 's/.*avm_gpu: \([0-9][0-9]*\) inter prediction blocks.*/\1/p' "$work/inter.log")
wcalls=$(sed -n 's/.* \([0-9][0-9]*\) warped.*/\1/p' "$work/inter.log")
fcalls=$(sed -n 's/.* \([0-9][0-9]*\) forward transforms.*/\1/p' "$work/inter.log")
dcalls=$(sed -n 's/.* \([0-9][0-9]*\) masked compound.*/\1/p' "$work/inter.log")
if [ -z "$icalls" ] || [ "$icalls" -le 0 ]; then
  echo "FAIL: the GPU inter path predicted no blocks"
  tail -20 "$work/inter.log"
  exit 1
fi
if [ -z "$wcalls" ] || [ "$wcalls" -le 0 ]; then
  echo "FAIL: the GPU warp path predicted no blocks"
  tail -20 "$work/inter.log"
  exit 1
fi
if [ -z "$fcalls" ] || [ "$fcalls" -le 0 ]; then
  echo "FAIL: the GPU forward transform ran on no blocks"
  tail -20 "$work/inter.log"
  exit 1
fi
if [ -z "$dcalls" ] || [ "$dcalls" -le 0 ]; then
  echo "FAIL: the GPU ran no masked inter-inter compound"
  tail -20 "$work/inter.log"
  exit 1
fi

# Fourth call site: masked (inter-intra / wedge) blending. It needs the slower
# preset - cpu-used 8 turns inter-intra compound off, so no mask is ever built.
MASK_ARGS="--limit=5 --good --cpu-used=2 --passes=1 --threads=1"
if ! "$AVMENC" $MASK_ARGS -o "$work/mask_cpu.ivf" "$work/motion.y4m" \
    >"$work/mask_cpu.log" 2>&1; then
  echo "FAIL: CPU masked-compound encode failed"
  tail -20 "$work/mask_cpu.log"
  exit 1
fi
if ! AV2_GPU_INTER=1 AV2_GPU_INTER_STATS=1 "$AVMENC" $MASK_ARGS \
    -o "$work/mask_gpu.ivf" "$work/motion.y4m" >"$work/mask.log" 2>&1; then
  if grep -q "no Vulkan device\|vkCreateInstance" "$work/mask.log"; then
    echo "SKIP: no Vulkan device"
    exit 77
  fi
  echo "FAIL: GPU masked-compound encode failed"
  tail -20 "$work/mask.log"
  exit 1
fi
mcalls=$(sed -n 's/.* \([0-9][0-9]*\) masked blends.*/\1/p' "$work/mask.log")
if [ -z "$mcalls" ] || [ "$mcalls" -le 0 ]; then
  echo "FAIL: the GPU blended no masked blocks"
  tail -20 "$work/mask.log"
  exit 1
fi

hash_of() {
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | cut -d' ' -f1
  else
    md5 -q "$1"
  fi
}

cpu_hash=$(hash_of "$work/cpu.ivf")
gpu_hash=$(hash_of "$work/gpu.ivf")
cpu_size=$(wc -c <"$work/cpu.ivf" | tr -d ' ')
gpu_size=$(wc -c <"$work/gpu.ivf" | tr -d ' ')

echo "gpu SAD blocks : $calls in $dispatches dispatches"
echo "cpu bitstream  : $cpu_size bytes, md5 $cpu_hash"
echo "gpu bitstream  : $gpu_size bytes, md5 $gpu_hash"

inter_cpu_hash=$(hash_of "$work/motion_cpu.ivf")
inter_hash=$(hash_of "$work/motion_gpu.ivf")
echo "inter bitstream: $(wc -c <"$work/motion_gpu.ivf" | tr -d ' ') bytes, md5 $inter_hash ($icalls inter blocks, $wcalls warped, $fcalls forward transforms, $dcalls masked compound)"

mask_cpu_hash=$(hash_of "$work/mask_cpu.ivf")
mask_hash=$(hash_of "$work/mask_gpu.ivf")
echo "mask bitstream : $(wc -c <"$work/mask_gpu.ivf" | tr -d ' ') bytes, md5 $mask_hash ($mcalls masked blends)"

var_hash=$(hash_of "$work/var.ivf")
echo "var bitstream  : $(wc -c <"$work/var.ivf" | tr -d ' ') bytes, md5 $var_hash ($tcalls variance blocks)"

if [ "$cpu_hash" != "$gpu_hash" ]; then
  echo "FAIL: SAD bitstreams differ"
  exit 1
fi
if [ "$cpu_hash" != "$var_hash" ]; then
  echo "FAIL: variance bitstreams differ"
  exit 1
fi
if [ "$inter_cpu_hash" != "$inter_hash" ]; then
  echo "FAIL: inter bitstreams differ ($inter_cpu_hash vs $inter_hash)"
  exit 1
fi
if [ "$mask_cpu_hash" != "$mask_hash" ]; then
  echo "FAIL: masked-compound bitstreams differ ($mask_cpu_hash vs $mask_hash)"
  exit 1
fi

echo "PASS: bit-identical bitstreams with the Vulkan SAD, variance, inter and masked-compound paths"
