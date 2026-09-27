"""Shared pytest fixtures and synthetic image generators."""

from __future__ import annotations

import io
from typing import Tuple

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from docshield.api.app import app
from docshield.core.detector import PIIDetector, TextRegionDetector
from docshield.core.redactor import ImageRedactor, encode_image


@pytest.fixture
def test_client() -> TestClient:
    """FastAPI TestClient fixture."""
    return TestClient(app)


@pytest.fixture
def synthetic_document_image() -> np.ndarray:
    """Generate a high-contrast synthetic document image simulating lines of text."""
    # 600x800 white page
    img = np.full((600, 800, 3), 255, dtype=np.uint8)

    # Add header
    cv2.putText(img, "CONFIDENTIAL INTERNAL REPORT", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
    cv2.line(img, (50, 75), (750, 75), (80, 80, 80), 2)

    # Add paragraphs of text
    lines = [
        "Patient Name: Johnathan Doe, SSN: 000-12-3456",
        "Medical Record Number: #MRN-847291-B",
        "Clinical Diagnosis: Acute Rhinosinusitis with elevated inflammation markers",
        "Attending Physician: Dr. Sarah Vance, MD (License #92837)",
        "Address: 742 Evergreen Terrace, Springfield, OR 97477",
        "Emergency Contact: Mary Doe (555) 019-2834",
    ]

    y_pos = 130
    for line in lines:
        cv2.putText(img, line, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
        y_pos += 45

    # Add signature area
    cv2.putText(img, "Signature: __________________________", (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)

    return img


@pytest.fixture
def synthetic_checkerboard_image() -> np.ndarray:
    """Generate an image with a high-frequency checkerboard in the center."""
    img = np.full((300, 300, 3), 128, dtype=np.uint8)
    # Center 100x100 checkerboard pattern
    for r in range(100, 200, 10):
        for c in range(100, 200, 10):
            if (r // 10 + c // 10) % 2 == 0:
                img[r : r + 10, c : c + 10] = (255, 255, 255)
            else:
                img[r : r + 10, c : c + 10] = (0, 0, 0)
    return img


@pytest.fixture
def synthetic_jpeg_bytes(synthetic_document_image: np.ndarray) -> bytes:
    """Return synthetic document encoded as JPEG bytes."""
    return encode_image(synthetic_document_image, format="jpeg")


@pytest.fixture
def default_redactor() -> ImageRedactor:
    """ImageRedactor instance."""
    return ImageRedactor()


@pytest.fixture
def default_detector() -> PIIDetector:
    """PIIDetector instance."""
    return PIIDetector()
