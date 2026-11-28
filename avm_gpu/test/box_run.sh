#!/bin/sh
# Runs the avm_gpu battery on a Linux box with a real Vulkan 1.4 driver:
# the gates, the three-way arena bisect, the encoder streams with the arena on,
# and the arena-vs-host-visible timing. Usage: sh /root/box_run.sh [build-dir]
set -u
B=${1:-/root/avm-vulkan/build}
cd "$(dirname "$B")" || exit 1
ENC=$B/avmenc

hash_of() { md5sum "$1" | cut -d' ' -f1; }

echo "== device"
"$B/avm_gpu/avm_gpu_sad_test" 2>&1 | head -1

echo "== gates (ctest, no parity)"
ctest --test-dir "$B" -R "avm_gpu_" -E "encoder_parity" 2>&1 | tail -3

echo "== clips"
python3 avm_gpu/test/make_noise_clip.py /tmp/box_noise.y4m
python3 avm_gpu/test/make_motion_clip.py /tmp/box_motion.y4m
python3 avm_gpu/test/make_rotate_clip.py /tmp/box_rotate.y4m

echo "== three-way arena bisect (2 frames of the noisy clip)"
ARGS="--limit=2 --good --cpu-used=8 --passes=1 --threads=1"
"$ENC" $ARGS -o /tmp/box_cpu.ivf /tmp/box_noise.y4m >/dev/null 2>&1
CPU=$(hash_of /tmp/box_cpu.ivf)
echo "cpu reference: $CPU"
for ex in none intra txfm itxfm; do
  if [ "$ex" = none ]; then EXCL=""; else EXCL="$ex"; fi
  AV2_GPU_ARENA=1 AV2_GPU_ARENA_EXCLUDE="$EXCL" AV2_GPU_INTRA=1 \
    AV2_GPU_FWD_TXFM=1 AV2_GPU_ITXFM=1 "$ENC" $ARGS -o "/tmp/box_ex_$ex.ivf" \
    /tmp/box_noise.y4m >/dev/null 2>&1
  rc=$?
  printf "exclude=%-6s rc=%d md5=%s %s\n" "${EXCL:-none}" "$rc" \
    "$(hash_of "/tmp/box_ex_$ex.ivf" 2>/dev/null)" \
    "$([ "$(hash_of "/tmp/box_ex_$ex.ivf" 2>/dev/null)" = "$CPU" ] && echo OK || echo DIFF)"
done

echo "== streams with the arena on (all hooks)"
ALL="AV2_GPU_INTER=1 AV2_GPU_INTRA=1 AV2_GPU_FWD_TXFM=1 AV2_GPU_ITXFM=1 AV2_GPU_DEBLOCK=1 AV2_GPU_CDEF=1 AV2_GPU_CCSO=1 AV2_GPU_LR=1"
"$ENC" --limit=5 --good --cpu-used=8 --passes=1 --threads=1 -o /tmp/box_m_cpu.ivf \
  /tmp/box_motion.y4m >/dev/null 2>&1
env AV2_GPU_ARENA=1 $ALL "$ENC" --limit=5 --good --cpu-used=8 --passes=1 \
  --threads=1 -o /tmp/box_m_gpu.ivf /tmp/box_motion.y4m >/dev/null 2>&1
printf "motion: cpu %s gpu %s\n" "$(hash_of /tmp/box_m_cpu.ivf)" \
  "$(hash_of /tmp/box_m_gpu.ivf)"
"$ENC" --limit=6 --good --cpu-used=8 --passes=1 --threads=1 -o /tmp/box_r_cpu.ivf \
  /tmp/box_rotate.y4m >/dev/null 2>&1
env AV2_GPU_ARENA=1 $ALL "$ENC" --limit=6 --good --cpu-used=8 --passes=1 \
  --threads=1 -o /tmp/box_r_gpu.ivf /tmp/box_rotate.y4m >/dev/null 2>&1
printf "rotate: cpu %s gpu %s\n" "$(hash_of /tmp/box_r_cpu.ivf)" \
  "$(hash_of /tmp/box_r_gpu.ivf)"

echo "== arena vs host-visible timing (rotate clip, inter)"
for i in 1 2; do
  t0=$(date +%s%N)
  AV2_GPU_INTER=1 "$ENC" --limit=6 --good --cpu-used=8 --passes=1 --threads=1 \
    -o /tmp/box_t1.ivf /tmp/box_rotate.y4m >/dev/null 2>&1
  t1=$(date +%s%N)
  AV2_GPU_INTER=1 AV2_GPU_ARENA=1 "$ENC" --limit=6 --good --cpu-used=8 \
    --passes=1 --threads=1 -o /tmp/box_t2.ivf /tmp/box_rotate.y4m >/dev/null 2>&1
  t2=$(date +%s%N)
  printf "run %d: host-visible %d ms (%s)   arena %d ms (%s)\n" "$i" \
    "$(( (t1 - t0) / 1000000 ))" "$(hash_of /tmp/box_t1.ivf)" \
    "$(( (t2 - t1) / 1000000 ))" "$(hash_of /tmp/box_t2.ivf)"
done
