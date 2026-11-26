#!/usr/bin/env python3
"""Writes a small deterministic 4:2:0 y4m clip for the encoder parity gate."""

import sys

W, H, FRAMES = 64, 64, 3


def plane(w, h, t):
    out = bytearray(w * h)
    for y in range(h):
        for x in range(w):
            # Moving gradient plus a blocky pattern: enough structure that the
            # encoder actually runs motion search, small enough to encode fast.
            v = (x * 3 + y * 2 + t * 17 + (16 if ((x // 8) ^ (y // 8)) & 1 else 0)) & 0xFF
            out[y * w + x] = v
    return out


def main(path):
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{W} H{H} F30:1 Ip A1:1 C420jpeg\n".encode())
        for t in range(FRAMES):
            f.write(b"FRAME\n")
            f.write(plane(W, H, t))
            f.write(plane(W // 2, H // 2, t + 40))
            f.write(plane(W // 2, H // 2, t + 80))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "clip64.y4m")
