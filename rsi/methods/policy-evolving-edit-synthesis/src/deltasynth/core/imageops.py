from __future__ import annotations

import re
from io import BytesIO
from typing import Optional

from PIL import Image


_SIZE_RE = re.compile(r"^\s*(\d+)\s*[xX*]\s*(\d+)\s*$")

JPEG_QUALITY = 95


def parse_size(spec: Optional[str]) -> Optional[tuple[int, int]]:
    if not spec:
        return None
    match = _SIZE_RE.match(str(spec))
    if not match:
        return None
    width, height = int(match.group(1)), int(match.group(2))
    return (width, height) if width > 0 and height > 0 else None


def read_size(data: bytes) -> Optional[tuple[int, int]]:
    try:
        with Image.open(BytesIO(data)) as img:
            return (int(img.width), int(img.height))
    except Exception:  # noqa: BLE001 — caller decides what a bad image means
        return None


class UnexpectedImageSize(RuntimeError):

    def __init__(self, got: tuple[int, int], want: tuple[int, int]) -> None:
        super().__init__(
            f"unexpected_image_size: endpoint returned {got[0]}x{got[1]}, "
            f"asked for {want[0]}x{want[1]}"
        )
        self.got = got
        self.want = want


def require_size(data: bytes, target: Optional[tuple[int, int]]) -> tuple[int, int]:
    current = read_size(data)
    if current is None:
        raise ValueError("could not read the returned image")
    if target is None or current == target:
        return current
    if current[0] * target[1] != current[1] * target[0]:
        raise UnexpectedImageSize(current, target)
    return target


def fit_size(
    data: bytes,
    target: Optional[tuple[int, int]],
    *,
    quality: int = JPEG_QUALITY,
) -> tuple[bytes, tuple[int, int]]:
    current = read_size(data)
    if current is None:
        raise ValueError("could not read the image to resize it")
    if target is None or current == target:
        return data, current
    if current[0] * target[1] != current[1] * target[0]:
        raise UnexpectedImageSize(current, target)
    with Image.open(BytesIO(data)) as img:
        resized = img.convert("RGB").resize(target, Image.Resampling.LANCZOS)
        buf = BytesIO()
        resized.save(buf, "JPEG", quality=quality)
    return buf.getvalue(), target
