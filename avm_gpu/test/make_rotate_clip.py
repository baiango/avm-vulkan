#!/usr/bin/env python3
"""Writes a small 4:2:0 y4m clip whose content rotates and zooms.

Compound prediction with a warped (affine) model wins when two references are
both related to the current frame by an affine motion, which is what a rotating
and slightly zooming pattern gives: the warp model can express it exactly, and
the two references (previous and golden) bracket the motion. The clip exists to
give `run_encoder_parity.sh` a stream in which the encoder actually selects
compound warped prediction, so a change in that kernel has to show up in the
bitstream.
"""

import math
import sys

W, H, FRAMES = 64, 64, 9


def base(x, y):
    return (0.5
            + 0.22 * math.sin(x * 0.37 + y * 0.21)
            + 0.16 * math.sin(x * 0.13 - y * 0.29)
            + 0.10 * math.cos((x - 2 * y) * 0.17)
            + 0.06 * math.cos((3 * x + y) * 0.41))


def sample(x, y, t):
    """Content rotating by ~1.2 deg per frame and zooming out slightly."""
    cx, cy = (W - 1) * 0.5, (H - 1) * 0.5
    ang = math.radians(1.2) * t
    zoom = 1.0 + 0.01 * t
    dx, dy = x - cx, y - cy
    sx = cx + (dx * math.cos(ang) - dy * math.sin(ang)) * zoom
    sy = cy + (dx * math.sin(ang) + dy * math.cos(ang)) * zoom
    x0, y0 = math.floor(sx), math.floor(sy)
    fx, fy = sx - x0, sy - y0
    v = (base(x0, y0) * (1 - fx) * (1 - fy) + base(x0 + 1, y0) * fx * (1 - fy) +
         base(x0, y0 + 1) * (1 - fx) * fy + base(x0 + 1, y0 + 1) * fx * fy)
    return max(0, min(255, int(round(v * 255.0))))


def plane(w, h, t):
    out = bytearray(w * h)
    for y in range(h):
        for x in range(w):
            out[y * w + x] = sample(x, y, t)
    return out


def main(path):
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{W} H{H} F30:1 Ip A1:1 C420jpeg\n".encode())
        for t in range(FRAMES):
            luma = plane(W, H, t)
            chroma_w, chroma_h = W // 2, H // 2
            chroma = bytearray(chroma_w * chroma_h * 2)
            for y in range(chroma_h):
                for x in range(chroma_w):
                    u = v = 0
                    for dy in range(2):
                        for dx in range(2):
                            s = sample(2 * x + dx, 2 * y + dy, t)
                            u += s - 128
                            v += s - 128
                    chroma[y * chroma_w + x] = 128 + int(u // 4) // 2
                    chroma[chroma_w * chroma_h + y * chroma_w + x] = (
                        128 + int(v // 4) // 2)
            f.write(b"FRAME\n")
            f.write(luma)
            f.write(chroma)


if __name__ == "__main__":
    main(sys.argv[1])
