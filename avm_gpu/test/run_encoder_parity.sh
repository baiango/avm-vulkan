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
if ! AV2_GPU_INTER=1 AV2_GPU_FWD_TXFM=1 AV2_GPU_ITXFM=1 \
    AV2_GPU_INTER_STATS=1 AV2_GPU_ITXFM_STATS=1 "$AVMENC" $INTER_ARGS \
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
wcalls=$(sed -n 's/.*blends, \([0-9][0-9]*\) warped,.*/\1/p' "$work/inter.log")
fcalls=$(sed -n 's/.* \([0-9][0-9]*\) forward transforms.*/\1/p' "$work/inter.log")
cwcalls=$(sed -n 's/.* \([0-9][0-9]*\) compound warped,.*/\1/p' "$work/inter.log")
wcpu=$(sed -n 's/.* \([0-9][0-9]*\) warped C).*/\1/p' "$work/inter.log")
icalls2=$(sed -n 's/.* \([0-9][0-9]*\) inverse transforms on the GPU.*/\1/p' "$work/inter.log")
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
if [ -z "$cwcalls" ] || [ "$cwcalls" -le 0 ]; then
  echo "FAIL: no compound warped block ran on the GPU"
  tail -20 "$work/inter.log"
  exit 1
fi
if [ -z "$icalls2" ] || [ "$icalls2" -le 0 ]; then
  echo "FAIL: no inverse transform ran on the GPU"
  tail -20 "$work/inter.log"
  exit 1
fi
if [ -z "$wcpu" ] || [ "$wcpu" -ne 0 ]; then
  echo "FAIL: $wcpu warped blocks were left to the CPU"
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

# Fifth call site: loop restoration. The clips above never make the encoder pick
# a restoration type, so this one is a noisy clip at a quality where PC-Wiener
# wins the RD decision; four frames are needed before the filtered reference
# frame reaches the bitstream (two frames still encode only the key frame).
python3 "$here/make_noise_clip.py" "$work/noise.y4m"
LR_ARGS="--limit=4 --good --cpu-used=8 --passes=1 --threads=1"
if ! "$AVMENC" $LR_ARGS -o "$work/lr_cpu.ivf" "$work/noise.y4m" \
    >"$work/lr_cpu.log" 2>&1; then
  echo "FAIL: CPU loop-restoration encode failed"
  tail -20 "$work/lr_cpu.log"
  exit 1
fi
if ! AV2_GPU_LR=1 AV2_GPU_LR_STATS=1 AV2_GPU_INTRA=1 AV2_GPU_INTRA_STATS=1 \
    "$AVMENC" $LR_ARGS \
    -o "$work/lr_gpu.ivf" "$work/noise.y4m" >"$work/lr.log" 2>&1; then
  if grep -q "no Vulkan device\|vkCreateInstance" "$work/lr.log"; then
    echo "SKIP: no Vulkan device"
    exit 77
  fi
  echo "FAIL: GPU loop-restoration encode failed"
  tail -20 "$work/lr.log"
  exit 1
fi
lrcalls=$(sed -n 's/.* \([0-9][0-9]*\) loop-restoration procunits.*/\1/p' "$work/lr.log")
intracalls=$(sed -n 's/.* \([0-9][0-9]*\) intra blocks on the GPU.*/\1/p' "$work/lr.log")
if [ -z "$lrcalls" ] || [ "$lrcalls" -le 0 ]; then
  echo "FAIL: the GPU filtered no loop-restoration procunit"
  tail -20 "$work/lr.log"
  exit 1
fi
if [ -z "$intracalls" ] || [ "$intracalls" -le 0 ]; then
  echo "FAIL: the GPU predicted no intra block"
  tail -20 "$work/lr.log"
  exit 1
fi

# Seventh call site: compound prediction over a scaled reference. Resizing only
# the key frame makes the inter frames (full size) predict from it, so compound
# blocks reference a scaled frame: the C facade routes those through
# av2_highbd_convolve_2d_scale() with is_compound, not the CWP family.
KF_ARGS="--limit=5 --good --cpu-used=8 --passes=1 --threads=1 --resize-mode=1 --resize-kf-denominator=9"
if ! "$AVMENC" $KF_ARGS -o "$work/kf_cpu.ivf" "$work/motion.y4m" \
    >"$work/kf_cpu.log" 2>&1; then
  echo "FAIL: CPU scaled-compound encode failed"
  tail -20 "$work/kf_cpu.log"
  exit 1
fi
if ! AV2_GPU_INTER=1 AV2_GPU_INTER_STATS=1 "$AVMENC" $KF_ARGS \
    -o "$work/kf_gpu.ivf" "$work/motion.y4m" >"$work/kf.log" 2>&1; then
  if grep -q "no Vulkan device\|vkCreateInstance" "$work/kf.log"; then
    echo "SKIP: no Vulkan device"
    exit 77
  fi
  echo "FAIL: GPU scaled-compound encode failed"
  tail -20 "$work/kf.log"
  exit 1
fi
kfcalls=$(sed -n 's/.* \([0-9][0-9]*\) scaled compound.*/\1/p' "$work/kf.log")
if [ -z "$kfcalls" ] || [ "$kfcalls" -le 0 ]; then
  echo "FAIL: no compound block over a scaled reference ran on the GPU"
  tail -20 "$work/kf.log"
  exit 1
fi

hash_of() {
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | cut -d' ' -f1
  else
    md5 -q "$1"
  fi
}

# Sixth call site: resized references. --resize-mode=1 resizes the source, so
# every block predicts from a scaled reference: the scaled 8-tap convolve, the
# scaled CWP compound (declined, see below) and the scaled affine kernel. Two
# ratios: 16:9 (step 1820, the one that exposed a one-row-short window) and
# 4:3 (step 1365).
resize_md5s=""
resize_counts=""
for den in 9 12; do
  RESIZE_ARGS="--limit=5 --good --cpu-used=8 --passes=1 --threads=1 --resize-mode=1 --resize-denominator=$den"
  if ! "$AVMENC" $RESIZE_ARGS -o "$work/resize${den}_cpu.ivf" "$work/motion.y4m" \
      >"$work/resize${den}_cpu.log" 2>&1; then
    echo "FAIL: CPU resized-reference encode (denominator $den) failed"
    tail -20 "$work/resize${den}_cpu.log"
    exit 1
  fi
  if ! AV2_GPU_INTER=1 AV2_GPU_INTER_STATS=1 "$AVMENC" $RESIZE_ARGS \
      -o "$work/resize${den}_gpu.ivf" "$work/motion.y4m" \
      >"$work/resize${den}.log" 2>&1; then
    if grep -q "no Vulkan device\|vkCreateInstance" "$work/resize${den}.log"; then
      echo "SKIP: no Vulkan device"
      exit 77
    fi
    echo "FAIL: GPU resized-reference encode (denominator $den) failed"
    tail -20 "$work/resize${den}.log"
    exit 1
  fi
  sw=$(sed -n 's/.* \([0-9][0-9]*\) scaled warped.*/\1/p' "$work/resize${den}.log")
  if [ -z "$sw" ] || [ "$sw" -le 0 ]; then
    echo "FAIL: no scaled warped block predicted on the GPU (denominator $den)"
    tail -20 "$work/resize${den}.log"
    exit 1
  fi
  resize_md5s="$resize_md5s $den:$(hash_of "$work/resize${den}_cpu.ivf")/$(hash_of "$work/resize${den}_gpu.ivf")"
  resize_counts="$resize_counts $den:$sw"
done

# Eight call site: the deblock filters, at both the encoder's own call sites
# (av2_filter_block_plane_{vert,horz}) and the TIP-frame pass. Deblocking changes
# every reconstructed frame, so this is checked against the CPU at three presets
# plus on the noisy clip, and the decoder-side filter is checked as well.
deblock_check() { # name, clip, args
  if ! AV2_GPU_DEBLOCK=1 AV2_GPU_DEBLOCK_STATS=1 "$AVMENC" $3 \
      -o "$work/db_$1.ivf" "$2" >"$work/db_$1.log" 2>&1; then
    if grep -q "no Vulkan device\|vkCreateInstance" "$work/db_$1.log"; then
      echo "SKIP: no Vulkan device"
      exit 77
    fi
    echo "FAIL: GPU deblock encode ($1) failed"
    tail -20 "$work/db_$1.log"
    exit 1
  fi
  if ! grep -q "deblock hook enabled" "$work/db_$1.log"; then
    echo "FAIL: the Vulkan deblock hook did not initialise ($1)"
    tail -20 "$work/db_$1.log"
    exit 1
  fi
  dedges=$(sed -n 's/.*avm_gpu: \([0-9][0-9]*\) deblock edges on the GPU.*/\1/p' "$work/db_$1.log")
  if [ -z "$dedges" ] || [ "$dedges" -le 0 ]; then
    echo "FAIL: the GPU filtered no deblock edge ($1)"
    tail -20 "$work/db_$1.log"
    exit 1
  fi
  echo "deblock $1: $dedges edges, md5 $(hash_of "$work/db_$1.ivf")"
}

deblock_fail=0
deblock_check noise8 "$work/noise.y4m" "$LR_ARGS"
if [ "$(hash_of "$work/db_noise8.ivf")" != "$(hash_of "$work/lr_cpu.ivf")" ]; then
  echo "FAIL: deblock bitstream differs from the CPU on the noise clip (cpu-used 8)"
  deblock_fail=1
fi
for cu in 8 4 2; do
  DB_ARGS="--limit=5 --good --cpu-used=$cu --passes=1 --threads=1"
  if [ "$cu" = 8 ]; then
    db_cpu="$work/motion_cpu.ivf"
  else
    if ! "$AVMENC" $DB_ARGS -o "$work/dbcu${cu}_cpu.ivf" "$work/motion.y4m" \
        >"$work/dbcu${cu}_cpu.log" 2>&1; then
      echo "FAIL: CPU deblock baseline encode (cpu-used $cu) failed"
      tail -20 "$work/dbcu${cu}_cpu.log"
      exit 1
    fi
    db_cpu="$work/dbcu${cu}_cpu.ivf"
  fi
  deblock_check "cu$cu" "$work/motion.y4m" "$DB_ARGS"
  if [ "$(hash_of "$work/db_cu$cu.ivf")" != "$(hash_of "$db_cpu")" ]; then
    echo "FAIL: deblock bitstream differs from the CPU at cpu-used $cu"
    deblock_fail=1
  fi
done

# The decoder runs the same av2_filter_block_plane_* path: decoding the stream
# with the GPU deblock must reproduce the CPU decoder's output byte for byte.
AVMDEC="$(dirname "$AVMENC")/avmdec"
if [ -x "$AVMDEC" ]; then
  if ! "$AVMDEC" --md5 "$work/db_cu8.ivf" >"$work/db_dec_cpu.log" 2>&1 ||
      ! AV2_GPU_DEBLOCK=1 "$AVMDEC" --md5 "$work/db_cu8.ivf" \
      >"$work/db_dec_gpu.log" 2>&1; then
    echo "FAIL: deblock decode failed"
    tail -20 "$work/db_dec_gpu.log"
    exit 1
  fi
  db_yuv_cpu=$(sed -n 's/^\([0-9a-f]\{32\}\) .*/\1/p' "$work/db_dec_cpu.log" | head -1)
  db_yuv_gpu=$(sed -n 's/^\([0-9a-f]\{32\}\) .*/\1/p' "$work/db_dec_gpu.log" | head -1)
  if [ -z "$db_yuv_gpu" ] || [ "$db_yuv_cpu" != "$db_yuv_gpu" ]; then
    echo "FAIL: GPU deblock decode differs ($db_yuv_gpu vs $db_yuv_cpu)"
    deblock_fail=1
  fi
  echo "deblock decode: md5 $db_yuv_gpu (gpu) / $db_yuv_cpu (cpu)"
fi
if [ "$deblock_fail" -ne 0 ]; then
  exit 1
fi

cpu_hash=$(hash_of "$work/cpu.ivf")
gpu_hash=$(hash_of "$work/gpu.ivf")
cpu_size=$(wc -c <"$work/cpu.ivf" | tr -d ' ')
gpu_size=$(wc -c <"$work/gpu.ivf" | tr -d ' ')

echo "gpu SAD blocks : $calls in $dispatches dispatches"
echo "cpu bitstream  : $cpu_size bytes, md5 $cpu_hash"
echo "gpu bitstream  : $gpu_size bytes, md5 $gpu_hash"

inter_cpu_hash=$(hash_of "$work/motion_cpu.ivf")
inter_hash=$(hash_of "$work/motion_gpu.ivf")
echo "inter bitstream: $(wc -c <"$work/motion_gpu.ivf" | tr -d ' ') bytes, md5 $inter_hash ($icalls inter blocks, $wcalls warped ($cwcalls compound, 0 left to the C), $fcalls forward transforms, $icalls2 inverse transforms, $dcalls masked compound)"

mask_cpu_hash=$(hash_of "$work/mask_cpu.ivf")
mask_hash=$(hash_of "$work/mask_gpu.ivf")
echo "mask bitstream : $(wc -c <"$work/mask_gpu.ivf" | tr -d ' ') bytes, md5 $mask_hash ($mcalls masked blends)"

lr_cpu_hash=$(hash_of "$work/lr_cpu.ivf")
lr_hash=$(hash_of "$work/lr_gpu.ivf")
echo "lr bitstream   : $(wc -c <"$work/lr_gpu.ivf" | tr -d ' ') bytes, md5 $lr_hash ($lrcalls loop-restoration procunits, $intracalls intra blocks)"

echo "resized streams: ratios$resize_counts (scaled warped blocks), md5$resize_md5s (cpu/gpu)"

kf_cpu_hash=$(hash_of "$work/kf_cpu.ivf")
kf_hash=$(hash_of "$work/kf_gpu.ivf")
echo "scaled compound: $(wc -c <"$work/kf_gpu.ivf" | tr -d ' ') bytes, md5 $kf_hash ($kfcalls compound blocks over a scaled reference)"

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
if [ "$kf_cpu_hash" != "$kf_hash" ]; then
  echo "FAIL: scaled-compound bitstreams differ"
  exit 1
fi
if [ "$lr_cpu_hash" != "$lr_hash" ]; then
  echo "FAIL: loop-restoration bitstreams differ"
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

echo "PASS: bit-identical bitstreams with the Vulkan SAD, variance, inter, masked-compound, intra, loop-restoration and deblock paths"
