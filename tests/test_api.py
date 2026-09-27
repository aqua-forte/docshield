"""Tests for FastAPI endpoints in DocShield."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from docshield.core.redactor import decode_image


def test_health_check(test_client: TestClient):
    """Test /health endpoint returns healthy status and service version."""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "docshield"
    assert "version" in data
    assert "face_detector_ready" in data
    assert "text_detector_ready" in data


def test_redact_endpoint_blur(test_client: TestClient, synthetic_jpeg_bytes: bytes):
    """Test /redact endpoint with Gaussian Blur mode."""
    response = test_client.post(
        "/redact",
        files={"file": ("report.jpg", synthetic_jpeg_bytes, "image/jpeg")},
        data={"mode": "blur", "detect_faces": "true", "detect_text": "true"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert "X-DocShield-Total-Redacted" in response.headers
    assert "X-DocShield-Processing-Time-Ms" in response.headers

    # Verify output is valid image
    anonymized = decode_image(response.content)
    assert anonymized.shape == (600, 800, 3)


def test_redact_endpoint_pixelate(test_client: TestClient, synthetic_jpeg_bytes: bytes):
    """Test /redact endpoint with Pixelation mode."""
    response = test_client.post(
        "/redact",
        files={"file": ("report.jpg", synthetic_jpeg_bytes, "image/jpeg")},
        data={"mode": "pixelate", "pixel_size": "12"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["X-DocShield-Mode"] == "pixelate"

    anonymized = decode_image(response.content)
    assert anonymized.shape == (600, 800, 3)


def test_redact_endpoint_png_output(test_client: TestClient, synthetic_jpeg_bytes: bytes):
    """Test /redact endpoint with PNG output format."""
    response = test_client.post(
        "/redact",
        files={"file": ("report.jpg", synthetic_jpeg_bytes, "image/jpeg")},
        data={"output_format": "png"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

    anonymized = decode_image(response.content)
    assert anonymized.shape == (600, 800, 3)


def test_redact_endpoint_corrupted_file(test_client: TestClient):
    """Test /redact returns 400 when sent corrupted binary data."""
    response = test_client.post(
        "/redact",
        files={"file": ("fake.jpg", b"invalid-binary-data", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Failed to decode image" in response.json()["detail"]


def test_detect_endpoint(test_client: TestClient, synthetic_jpeg_bytes: bytes):
    """Test /detect endpoint returns JSON bounding boxes without altering image."""
    response = test_client.post(
        "/detect",
        files={"file": ("report.jpg", synthetic_jpeg_bytes, "image/jpeg")},
        data={"detect_faces": "false", "detect_text": "true"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "boxes" in data
    assert "text_count" in data
    assert data["text_count"] > 0
    assert data["image_width"] == 800
    assert data["image_height"] == 600
