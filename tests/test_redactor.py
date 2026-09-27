"""Tests for image redactor and spatial filtering operations."""

from __future__ import annotations

import numpy as np
import pytest

from docshield.core.redactor import ImageRedactor, decode_image, encode_image
from docshield.core.types import BoundingBox, RedactMode


def test_gaussian_blur_modifies_roi_only(synthetic_checkerboard_image: np.ndarray):
    """Test that Gaussian blur strictly changes pixels inside the ROI and leaves outside intact."""
    redactor = ImageRedactor(default_mode=RedactMode.BLUR)
    box = BoundingBox(x=100, y=100, width=100, height=100)

    original = synthetic_checkerboard_image.copy()
    redacted = redactor.redact(original, boxes=[box], mode=RedactMode.BLUR)

    # 1. Output dimensions and dtype must match original exactly
    assert redacted.shape == original.shape
    assert redacted.dtype == original.dtype

    # 2. Outside regions must be 100% identical
    # Check top margin (0 to 100 y)
    assert np.array_equal(redacted[0:100, :], original[0:100, :])
    # Check bottom margin (200 to 300 y)
    assert np.array_equal(redacted[200:300, :], original[200:300, :])
    # Check left margin (0 to 100 x)
    assert np.array_equal(redacted[:, 0:100], original[:, 0:100])
    # Check right margin (200 to 300 x)
    assert np.array_equal(redacted[:, 200:300], original[:, 200:300])

    # 3. Inside the ROI must be blurred (pixels changed and variance reduced)
    original_roi = original[100:200, 100:200]
    redacted_roi = redacted[100:200, 100:200]

    assert not np.array_equal(original_roi, redacted_roi)
    # Variance of blurred checkerboard should be significantly lower than sharp checkerboard
    assert np.var(redacted_roi) < np.var(original_roi)


def test_pixelation_modifies_roi_only(synthetic_checkerboard_image: np.ndarray):
    """Test that pixelation creates mosaic blocks inside ROI and preserves outside pixels."""
    redactor = ImageRedactor(default_mode=RedactMode.PIXELATE, pixel_size=16)
    box = BoundingBox(x=100, y=100, width=100, height=100)

    original = synthetic_checkerboard_image.copy()
    redacted = redactor.redact(original, boxes=[box], mode=RedactMode.PIXELATE)

    # Dimensions check
    assert redacted.shape == original.shape

    # Outside regions intact
    assert np.array_equal(redacted[0:100, :], original[0:100, :])
    assert np.array_equal(redacted[200:300, :], original[200:300, :])

    # Inside ROI is modified
    assert not np.array_equal(redacted[100:200, 100:200], original[100:200, 100:200])


def test_redactor_boundary_clamping():
    """Test that boxes exceeding image bounds or negative coordinates are safely clamped."""
    redactor = ImageRedactor()
    img = np.full((100, 100, 3), 200, dtype=np.uint8)

    # Box exceeding right/bottom edge
    box_overflow = BoundingBox(x=80, y=80, width=150, height=150)
    # Box with coordinates at upper limit
    box_edge = BoundingBox(x=99, y=99, width=10, height=10)

    # Should execute without throwing IndexError or shape mismatch
    result = redactor.redact(img, boxes=[box_overflow, box_edge], mode=RedactMode.BLUR)
    assert result.shape == (100, 100, 3)

    result_pix = redactor.redact(img, boxes=[box_overflow, box_edge], mode=RedactMode.PIXELATE)
    assert result_pix.shape == (100, 100, 3)


def test_redactor_empty_and_zero_boxes():
    """Test redaction with empty box lists or zero-sized images."""
    redactor = ImageRedactor()
    img = np.zeros((50, 50, 3), dtype=np.uint8)

    # Empty boxes
    clean = redactor.redact(img, boxes=[])
    assert np.array_equal(clean, img)

    # Empty image
    empty_img = np.array([], dtype=np.uint8)
    assert redactor.redact(empty_img, boxes=[]).size == 0


def test_encode_and_decode_roundtrip():
    """Test image encoding and decoding to/from byte streams."""
    img = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)

    # Test JPEG
    jpg_bytes = encode_image(img, format="jpeg", quality=90)
    assert isinstance(jpg_bytes, bytes)
    assert len(jpg_bytes) > 0
    decoded_jpg = decode_image(jpg_bytes)
    assert decoded_jpg.shape == img.shape

    # Test PNG (lossless)
    png_bytes = encode_image(img, format="png")
    assert isinstance(png_bytes, bytes)
    decoded_png = decode_image(png_bytes)
    assert np.array_equal(decoded_png, img)


def test_decode_invalid_bytes():
    """Test that decoding corrupted bytes raises ValueError."""
    with pytest.raises(ValueError, match="empty image byte payload"):
        decode_image(b"")

    with pytest.raises(ValueError, match="Failed to decode image"):
        decode_image(b"not-an-image-data-string")
