"""Image redaction and masking engine with Gaussian blur and pixelation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from docshield.core.types import BoundingBox, RedactMode


class ImageRedactor:
    """High-performance image redaction engine supporting Gaussian Blur and Pixelation."""

    def __init__(
        self,
        default_mode: RedactMode = RedactMode.BLUR,
        blur_intensity: int = 51,
        pixel_size: int = 14,
    ):
        """Initialize redactor.

        Args:
            default_mode: Default redaction mode (blur or pixelate).
            blur_intensity: Base kernel size for Gaussian blur (will be normalized to odd).
            pixel_size: Pixelation block size in pixels.
        """
        self.default_mode = default_mode
        self.blur_intensity = blur_intensity if blur_intensity % 2 != 0 else blur_intensity + 1
        self.pixel_size = max(2, pixel_size)

    def apply_gaussian_blur(
        self,
        roi: np.ndarray,
        intensity: Optional[int] = None,
    ) -> np.ndarray:
        """Apply heavy, irreversible Gaussian smoothing to the region of interest.

        Args:
            roi: Image region slice.
            intensity: Kernel size (must be odd). If None, calculated dynamically from ROI size.

        Returns:
            Blurred ROI numpy array.
        """
        if roi.size == 0:
            return roi

        h, w = roi.shape[:2]
        if intensity is None:
            # Dynamically size kernel based on ROI dimensions, clamped to odd integers
            kw = max(15, (w // 3) | 1)
            kh = max(15, (h // 3) | 1)
        else:
            k = intensity if intensity % 2 != 0 else intensity + 1
            kw, kh = min(w | 1, k), min(h | 1, k)
            kw = max(3, kw | 1)
            kh = max(3, kh | 1)

        # Apply multi-pass smoothing to ensure text/facial features cannot be inverted
        blurred = cv2.GaussianBlur(roi, (kw, kh), sigmaX=30, sigmaY=30)
        blurred = cv2.GaussianBlur(blurred, (kw, kh), sigmaX=30, sigmaY=30)
        return blurred

    def apply_pixelation(
        self,
        roi: np.ndarray,
        block_size: Optional[int] = None,
    ) -> np.ndarray:
        """Apply mosaic pixelation by downsampling and nearest-neighbor upsampling.

        Args:
            roi: Image region slice.
            block_size: Tile width/height in pixels.

        Returns:
            Pixelated ROI numpy array.
        """
        if roi.size == 0:
            return roi

        h, w = roi.shape[:2]
        bs = max(2, block_size or self.pixel_size)

        # Compute intermediate dimensions
        small_w = max(1, w // bs)
        small_h = max(1, h // bs)

        # Downscale with area interpolation
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        # Upscale back to original ROI size using nearest neighbor for crisp mosaic blocks
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        return pixelated

    def redact(
        self,
        image: np.ndarray,
        boxes: List[BoundingBox],
        mode: Optional[Union[RedactMode, str]] = None,
        intensity: Optional[int] = None,
        copy: bool = True,
    ) -> np.ndarray:
        """Redact sensitive regions on the provided image.

        Args:
            image: Original image as numpy ndarray.
            boxes: Bounding boxes to redact.
            mode: Redaction mode ('blur' or 'pixelate').
            intensity: Blur kernel size or pixel tile size.
            copy: If True, operates on a copy; if False, modifies in-place.

        Returns:
            Anonymized image preserving original resolution and channels.
        """
        if image is None or image.size == 0:
            return image

        target_mode = RedactMode(mode) if mode else self.default_mode
        output = image.copy() if copy else image
        img_h, img_w = output.shape[:2]

        for box in boxes:
            # Safe boundary clamping
            x1 = max(0, min(int(box.x), img_w - 1))
            y1 = max(0, min(int(box.y), img_h - 1))
            x2 = max(x1 + 1, min(int(box.x2), img_w))
            y2 = max(y1 + 1, min(int(box.y2), img_h))

            roi = output[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            if target_mode == RedactMode.BLUR:
                output[y1:y2, x1:x2] = self.apply_gaussian_blur(roi, intensity=intensity)
            elif target_mode == RedactMode.PIXELATE:
                output[y1:y2, x1:x2] = self.apply_pixelation(roi, block_size=intensity)
            else:
                output[y1:y2, x1:x2] = self.apply_gaussian_blur(roi, intensity=intensity)

        return output


def encode_image(
    image: np.ndarray,
    format: str = "jpeg",
    quality: int = 95,
) -> bytes:
    """Encode OpenCV image matrix into compressed image byte array.

    Args:
        image: Numpy ndarray image.
        format: Target format ('jpeg', 'jpg', 'png', 'webp').
        quality: Compression quality factor (1-100).

    Returns:
        Encoded image bytes.
    """
    ext = format.lower().lstrip(".")
    if ext in ("jpeg", "jpg"):
        ext_dot = ".jpg"
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    elif ext == "png":
        ext_dot = ".png"
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 4]
    elif ext == "webp":
        ext_dot = ".webp"
        params = [int(cv2.IMWRITE_WEBP_QUALITY), int(quality)]
    else:
        ext_dot = f".{ext}"
        params = []

    success, buffer = cv2.imencode(ext_dot, image, params)
    if not success:
        raise ValueError(f"Failed to encode image to format: {format}")

    return buffer.tobytes()


def decode_image(data: bytes) -> np.ndarray:
    """Decode raw bytes into OpenCV image matrix.

    Args:
        data: Image byte payload.

    Returns:
        Decoded image as numpy ndarray in BGR format.
    """
    if not data:
        raise ValueError("Received empty image byte payload.")

    nparr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image from provided byte stream.")

    return image


def read_image(path: Union[str, Path]) -> Optional[np.ndarray]:
    """Read an image from disk safely supporting non-ASCII / Unicode paths on Windows/Linux."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with open(p, "rb") as f:
            data = f.read()
        return decode_image(data)
    except Exception:
        return None


def write_image(path: Union[str, Path], image: np.ndarray, quality: int = 95) -> bool:
    """Write an image to disk safely supporting non-ASCII / Unicode paths on Windows/Linux."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fmt = p.suffix.lower().lstrip(".")
    if not fmt:
        fmt = "jpeg"
    try:
        data = encode_image(image, format=fmt, quality=quality)
        with open(p, "wb") as f:
            f.write(data)
        return True
    except Exception:
        return False
