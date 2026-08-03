"""Photo-math intake: PNG/PGM decoding and a deterministic printed-equation OCR.

The synthetic recognizer is a real image pipeline (threshold, segmentation,
nearest-neighbour glyph matching) for clean renders. ``HttpVlmOcr`` is the
adapter for real photos through dots.mocr or a PaliGemma-compatible endpoint.
"""

from __future__ import annotations

import base64
import json
import re
import struct
import zlib
from collections.abc import Callable
from dataclasses import dataclass

from .safety import ProviderError


class ImageError(ValueError):
    """Raised when image bytes cannot be decoded."""


@dataclass(frozen=True)
class Image:
    width: int
    height: int
    pixels: tuple[tuple[int, ...], ...]

    def pixel(self, x: int, y: int) -> int:
        return self.pixels[y][x]


@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: float


_FONT: dict[str, tuple[str, ...]] = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "x": ("00000", "00000", "10001", "01010", "00100", "01010", "10001"),
    "y": ("00000", "00000", "10001", "01010", "00100", "00100", "01000"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "=": ("00000", "11111", "00000", "11111", "00000", "00000", "00000"),
    "(": ("00010", "00100", "01000", "01000", "01000", "00100", "00010"),
    ")": ("01000", "00100", "00010", "00010", "00010", "00100", "01000"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "00110", "00110"),
    "^": ("00100", "01010", "10001", "00000", "00000", "00000", "00000"),
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
}


def _font_rows(char: str, scale: int) -> list[list[int]]:
    glyph = _FONT.get(char.lower(), _FONT[" "])
    rows: list[list[int]] = []
    for row in glyph:
        expanded = []
        for cell in row:
            expanded.extend([0 if cell == "1" else 255] * scale)
        for _ in range(scale):
            rows.append(expanded)
    return rows


def render_image(text: str, scale: int = 3, padding: int = 8) -> Image:
    glyphs = [_font_rows(char, scale) for char in text]
    if not glyphs:
        glyphs = [_font_rows(" ", scale)]
    gap = [255] * scale
    width = padding * 2 + sum(len(g[0]) for g in glyphs) + gap.__len__() * (len(glyphs) - 1)
    height = padding * 2 + 7 * scale
    canvas = [[255] * width for _ in range(height)]
    x = padding
    for rows in glyphs:
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row):
                canvas[padding + row_index][x + column_index] = value
        x += len(rows[0]) + len(gap)
    return Image(width=width, height=height, pixels=tuple(tuple(row) for row in canvas))


def encode_png(image: Image) -> bytes:
    height = len(image.pixels)
    width = len(image.pixels[0]) if height else 0
    raw = bytearray()
    for row in image.pixels:
        raw.append(0)
        for value in row:
            grey = max(0, min(255, int(value)))
            raw.extend((grey, grey, grey))
    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return (
            struct.pack(">I", len(payload))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(
        b"IDAT", zlib.compress(bytes(raw), 9)
    ) + chunk(b"IEND", b"")


def render_pgm(image: Image) -> bytes:
    height = len(image.pixels)
    width = len(image.pixels[0]) if height else 0
    body = bytearray()
    for row in image.pixels:
        body.extend(max(0, min(255, int(value))) for value in row)
    return f"P5\n{width} {height}\n255\n".encode("ascii") + bytes(body)


def decode_image(data: bytes, media_type: str = "") -> Image:
    if len(data) > 5 * 1024 * 1024:
        raise ImageError("image exceeds the 5 MiB limit")
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return _decode_png(data)
    if data.startswith(b"P5"):
        return _decode_pgm(data)
    raise ImageError("only non-interlaced 8-bit PNG and PGM P5 images are supported")


def _decode_pgm(data: bytes) -> Image:
    header_match = re.match(rb"P5\s+(?:#[^\n]*\n\s*)*(\d+)\s+(\d+)\s+(\d+)\s", data)
    if not header_match:
        raise ImageError("invalid PGM header")
    width, height, maxval = map(int, header_match.groups())
    if maxval != 255 or width <= 0 or height <= 0 or width > 4000 or height > 4000:
        raise ImageError("unsupported PGM dimensions or max value")
    offset = header_match.end()
    payload = data[offset : offset + width * height]
    if len(payload) != width * height:
        raise ImageError("truncated PGM payload")
    rows = tuple(
        tuple(payload[y * width + x] for x in range(width)) for y in range(height)
    )
    return Image(width, height, rows)


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _decode_png(data: bytes) -> Image:
    offset = 8
    width = height = bit_depth = color_type = interlace = 0
    idat = bytearray()
    while offset + 8 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            if len(payload) != 13:
                raise ImageError("invalid PNG header")
            width, height, bit_depth, color_type, _compression, _filter, interlace = (
                struct.unpack(">IIBBBBB", payload)
            )
        elif kind == b"IDAT":
            idat.extend(payload)
        elif kind == b"IEND":
            break
    if width <= 0 or height <= 0 or width > 4000 or height > 4000:
        raise ImageError("unsupported PNG dimensions")
    if bit_depth != 8 or interlace != 0 or color_type not in {0, 2, 6}:
        raise ImageError("only non-interlaced 8-bit grayscale/RGB/RGBA PNG is supported")
    channels = {0: 1, 2: 3, 6: 4}[color_type]
    try:
        raw = zlib.decompress(bytes(idat))
    except zlib.error as exc:
        raise ImageError("invalid PNG compressed data") from exc
    stride = width * channels
    if len(raw) != (stride + 1) * height:
        raise ImageError("truncated PNG pixel data")
    rows: list[list[int]] = []
    previous = bytearray(stride)
    cursor = 0
    for _ in range(height):
        filter_type = raw[cursor]
        cursor += 1
        current = bytearray(raw[cursor : cursor + stride])
        cursor += stride
        for index in range(stride):
            left = current[index - channels] if index >= channels else 0
            up = previous[index]
            up_left = previous[index - channels] if index >= channels else 0
            if filter_type == 1:
                current[index] = (current[index] + left) & 0xFF
            elif filter_type == 2:
                current[index] = (current[index] + up) & 0xFF
            elif filter_type == 3:
                current[index] = (current[index] + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                current[index] = (current[index] + _paeth(left, up, up_left)) & 0xFF
            elif filter_type != 0:
                raise ImageError("unsupported PNG filter")
        grey = []
        for x in range(width):
            position = x * channels
            if channels == 1:
                grey.append(current[position])
            else:
                red, green, blue = current[position : position + 3]
                grey.append((red * 299 + green * 587 + blue * 114) // 1000)
        rows.append(grey)
        previous = current
    return Image(width, height, tuple(tuple(row) for row in rows))


class SyntheticOcr:
    def recognize(self, image: Image) -> OcrResult:
        pixels = image.pixels
        values = [value for row in pixels for value in row]
        if not values:
            raise ImageError("empty image")
        threshold = (min(values) + max(values)) // 2
        ink = [[value <= threshold for value in row] for row in pixels]
        columns = [any(ink[y][x] for y in range(image.height)) for x in range(image.width)]
        groups: list[tuple[int, int]] = []
        start: int | None = None
        for x, populated in enumerate(columns):
            if populated and start is None:
                start = x
            elif not populated and start is not None:
                groups.append((start, x))
                start = None
        if start is not None:
            groups.append((start, image.width))
        if not groups:
            raise ImageError("no glyphs detected")
        chars: list[str] = []
        matches = 0
        for left, right in groups:
            mask = [
                [1 if ink[y][x] else 0 for x in range(left, right)]
                for y in range(image.height)
            ]
            char = _match_glyph(mask)
            if char is None:
                chars.append("?")
            else:
                chars.append(char)
                matches += 1
        confidence = matches / max(1, len(groups))
        text = "".join(chars)
        if confidence < 0.8 or "=" not in text:
            raise ImageError("low-confidence equation recognition")
        return OcrResult(text=text, confidence=confidence)


def _canonical(mask: list[list[int]], width: int = 5, height: int = 7) -> list[list[int]]:
    """Crop to the ink bounding box, then normalize to a 5x7 comparison grid.

    Cropping makes recognition invariant to where a glyph sits inside its
    segmentation box, which matters for asymmetric shapes such as parentheses.
    """
    populated = [
        (y, x)
        for y, row in enumerate(mask)
        for x, value in enumerate(row)
        if value
    ]
    if not populated:
        return [[0] * width for _ in range(height)]
    top = min(y for y, _ in populated)
    bottom = max(y for y, _ in populated) + 1
    left = min(x for _, x in populated)
    right = max(x for _, x in populated) + 1
    source_height = bottom - top
    source_width = right - left
    result = []
    for y in range(height):
        sy = top + min(source_height - 1, int(y * source_height / height))
        row = []
        for x in range(width):
            sx = left + min(source_width - 1, int(x * source_width / width))
            row.append(mask[sy][sx])
        result.append(row)
    return result


def _match_glyph(mask: list[list[int]]) -> str | None:
    normalized = _canonical(mask)
    best_char: str | None = None
    best_distance = 10**9
    for char, rows in _FONT.items():
        if char == " ":
            continue
        template_mask = [[1 if cell == "1" else 0 for cell in row] for row in rows]
        template = _canonical(template_mask)
        distance = sum(
            abs(normalized[y][x] - template[y][x])
            for y in range(7)
            for x in range(5)
        )
        if distance < best_distance:
            best_distance = distance
            best_char = char
    return best_char if best_distance <= 9 else None


def render_png(text: str, scale: int = 3) -> bytes:
    return encode_png(render_image(text, scale))


class HttpVlmOcr:
    def __init__(
        self,
        url: str,
        model: str = "dots.mocr",
        api_key: str | None = None,
        transport: Callable[[str, dict[str, str], bytes, float], bytes] | None = None,
    ) -> None:
        self.url = url
        self.model = model
        self.api_key = api_key
        self.transport = transport or _urllib_post

    def recognize(self, image: Image) -> OcrResult:
        encoded = base64.b64encode(encode_png(image)).decode("ascii")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only the equation as ASCII, for example 3x+6=12.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read this math equation."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded}"},
                        },
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": 80,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            raw = self.transport(
                self.url, headers, json.dumps(payload).encode("utf-8"), 15.0
            )
            data = json.loads(raw.decode("utf-8"))
            content = str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError, ValueError, OSError) as exc:
            raise ProviderError("photo OCR request failed") from exc
        text = content.strip().strip("`").removeprefix("text").strip()
        text = re.sub(r"\s+", "", text)
        if not text or "=" not in text:
            raise ImageError("OCR did not return an equation")
        return OcrResult(text=text, confidence=0.95)


def _urllib_post(url: str, headers: dict[str, str], body: bytes, timeout: float) -> bytes:
    import urllib.request

    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return bytes(response.read())
