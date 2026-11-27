#!/usr/bin/env python3
"""Writes a small 4:2:0 y4m clip whose content moves by a fractional pixel.

The gate clip slides a gradient by a whole pixel per frame, which the encoder
answers with integer motion vectors - those take the copy path, not the subpel
filter. Here the content is a fixed textured image sampled at x + 0.5 * t, so
the best motion vectors land on half-pel positions and the subpel convolution
(and the Vulkan seam over it) is actually exercised.
"""

import math
import sys

W, H, FRAMES = 64, 64, 5


def base(x, y):
    # Structured, non-repeating-ish field with plenty of gradient energy.
    return (0.5
            + 0.25 * math.sin(x * 0.7 + y * 0.3)
            + 0.15 * math.sin(x * 0.23 - y * 0.51)
            + 0.10 * math.cos((x + 2 * y) * 0.11))


def plane(w, h, t):
    shift = 0.5 * t
    out = bytearray(w * h)
    for y in range(h):
        for x in range(w):
            sx = x + shift
            x0 = int(math.floor(sx))
            fx = sx - x0
            v = (base(x0, y) * (1.0 - fx) + base(x0 + 1, y) * fx)
            out[y * w + x] = max(0, min(255, int(round(v * 255.0))))
    return out


def main(path):
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{W} H{H} F30:1 Ip A1:1 C420jpeg\n".encode())
        for t in range(FRAMES):
            f.write(b"FRAME\n")
            f.write(plane(W, H, t))
            f.write(plane(W // 2, H // 2, t))
            f.write(plane(W // 2, H // 2, t))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "clip_motion.y4m")
