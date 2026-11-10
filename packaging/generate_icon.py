"""Generate a crisp multi-resolution Windows .ico file (bridge arch + analytics bars) in pure Python."""
from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path
from typing import List, Tuple


def _make_png_rgba(width: int, height: int, pixels: bytes) -> bytes:
    """Encode raw RGBA bytes (width * height * 4) into a valid PNG image."""
    raw_rows = bytearray()
    stride = width * 4
    for y in range(height):
        raw_rows.append(0)  # Filter type 0 (None)
        raw_rows.extend(pixels[y * stride : (y + 1) * stride])

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    idat = zlib.compress(bytes(raw_rows), level=9)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )


def _render_bridge_icon_rgba(size: int) -> bytes:
    """Render anti-aliased rounded square badge with bridge arch & traffic analytics bars."""
    buf = bytearray(size * size * 4)
    radius = size * 0.22
    pad = size * 0.04

    for y in range(size):
        ny = y / float(size - 1 if size > 1 else 1)
        for x in range(size):
            nx = x / float(size - 1 if size > 1 else 1)
            idx = (y * size + x) * 4

            # Signed distance to rounded rectangle
            cx = max(pad + radius, min(size - 1 - pad - radius, float(x)))
            cy = max(pad + radius, min(size - 1 - pad - radius, float(y)))
            dist = math.hypot(x - cx, y - cy) - radius
            if dist > 0.75:
                continue
            alpha = 1.0 if dist < -0.75 else max(0.0, min(1.0, 0.5 - dist / 1.5))

            # Rich teal gradient background (#0B4F4A -> #0F766E -> #0284C7 accent)
            r = int(11 + 10 * (1.0 - ny) + 8 * nx)
            g = int(85 + 42 * (1.0 - ny))
            b = int(82 + 48 * nx)

            # Subtle inner border highlight
            if -2.2 < dist < -0.6 and size >= 32:
                r = min(255, r + 28)
                g = min(255, g + 38)
                b = min(255, b + 38)

            # Analytics rising bars in upper half (gold/amber & sky highlights)
            bar_specs = [
                (0.23, 0.33, 0.36, 0.54, (125, 211, 252)),  # Sky bar 1
                (0.38, 0.48, 0.25, 0.54, (56, 189, 248)),   # Sky bar 2
                (0.53, 0.63, 0.17, 0.54, (251, 191, 36)),   # Gold peak bar 3
                (0.68, 0.78, 0.29, 0.54, (245, 158, 11)),   # Amber bar 4
            ]
            for x0, x1, y0, y1, col in bar_specs:
                if x0 <= nx <= x1 and y0 <= ny <= y1:
                    r, g, b = col

            # Bridge deck horizontal beam (white)
            if 0.12 <= nx <= 0.88 and 0.57 <= ny <= 0.64:
                r, g, b = (248, 250, 252)

            # Bridge main arch curve underneath/supporting deck
            arch_y = 0.84 - 0.25 * math.sin(max(0.0, min(1.0, (nx - 0.12) / 0.76)) * math.pi)
            if 0.12 <= nx <= 0.88 and abs(ny - arch_y) <= max(0.038, 1.3 / size):
                r, g, b = (255, 255, 255)

            # Vertical suspension/pier lines between arch and deck
            for px in (0.26, 0.38, 0.50, 0.62, 0.74):
                if abs(nx - px) <= max(0.018, 0.8 / size):
                    top_y = min(0.61, arch_y)
                    bot_y = max(0.61, arch_y)
                    if top_y <= ny <= bot_y:
                        r, g, b = (226, 232, 240)

            buf[idx] = r
            buf[idx + 1] = g
            buf[idx + 2] = b
            buf[idx + 3] = int(round(alpha * 255))

    return bytes(buf)


def generate_bridge_ico(output_path: Path) -> Path:
    """Create a multi-size Windows .ico file containing 16, 32, 48, 64, and 256 px PNG frames."""
    sizes: List[int] = [16, 32, 48, 64, 256]
    frames: List[Tuple[int, bytes]] = []
    for sz in sizes:
        rgba = _render_bridge_icon_rgba(sz)
        png_bytes = _make_png_rgba(sz, sz, rgba)
        frames.append((sz, png_bytes))

    # ICONDIR header: Reserved=0, Type=1 (ICO), Count=len(frames)
    header = struct.pack("<HHH", 0, 1, len(frames))
    entries = bytearray()
    payload = bytearray()
    offset = 6 + 16 * len(frames)

    for sz, data in frames:
        w_byte = 0 if sz >= 256 else sz
        h_byte = 0 if sz >= 256 else sz
        entry = struct.pack(
            "<BBBBHHII",
            w_byte,       # width (0 means 256)
            h_byte,       # height (0 means 256)
            0,            # color palette count
            0,            # reserved
            1,            # color planes
            32,           # bits per pixel
            len(data),    # image byte size
            offset,       # offset from start of file
        )
        entries.extend(entry)
        payload.extend(data)
        offset += len(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(header + bytes(entries) + bytes(payload))
    return output_path


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "assets" / "bridge_app.ico"
    generate_bridge_ico(target)
    print(f"Generated icon: {target} ({target.stat().st_size} bytes)")
