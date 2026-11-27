#!/usr/bin/env python3
"""Writes the noisy clip the loop-restoration gate encodes.

The tiny clean clips the other gates use never make the encoder pick a
restoration type: RESTORE_NONE wins the RD decision and the loop-restoration
filter never runs. This one is textured 192x192 noise with a moving gradient, at
a quality where PC-Wiener is chosen, and it is deterministic so the expected
bitstreams stay stable.
"""

import random
import sys

W, H, FRAMES = 192, 192, 4
SEED = 7


def plane(w, h, t, base):
    out = bytearray(w * h)
    for y in range(h):
        for x in range(w):
            v = base + (x * 2 + y * 3 + t * 11) % 60 + random.randint(-24, 24)
            v += 40 if ((x // 16) ^ (y // 16)) & 1 else 0
            out[y * w + x] = max(0, min(255, v))
    return out


def main(path):
    random.seed(SEED)
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{W} H{H} F30:1 Ip A1:1 C420jpeg\n".encode())
        for t in range(FRAMES):
            f.write(b"FRAME\n")
            f.write(plane(W, H, t, 80))
            f.write(plane(W // 2, H // 2, t, 120))
            f.write(plane(W // 2, H // 2, t, 140))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "noise.y4m")
