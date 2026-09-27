"""Tests for face and document text detectors."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from docshield.core.detector import FaceDetector, PIIDetector, TextRegionDetector
from docshield.core.types import BoundingBox, PIIType


def test_bounding_box_geometry():
    """Test geometric operations and IoU calculation for BoundingBox."""
    b1 = BoundingBox(x=10, y=10, width=50, height=50)
    assert b1.x2 == 60
    assert b1.y2 == 60
    assert b1.area == 2500
    assert b1.to_xyxy() == (10, 10, 60, 60)
    assert b1.to_xywh() == (10, 10, 50, 50)

    # Identical box should have IoU = 1.0
    b2 = BoundingBox(x=10, y=10, width=50, height=50)
    assert pytest.approx(b1.iou(b2), 0.001) == 1.0

    # Disjoint box should have IoU = 0.0
    b3 = BoundingBox(x=100, y=100, width=50, height=50)
    assert b1.iou(b3) == 0.0

    # Partially overlapping box:
    # b1: [10, 10, 60, 60] -> area 2500
    # b4: [35, 10, 85, 60] -> width 50, height 50, intersection: x=[35,60] (25) * y=[10,60] (50) = 1250
    # union: 2500 + 2500 - 1250 = 3750. IoU = 1250 / 3750 = 1/3
    b4 = BoundingBox(x=35, y=10, width=50, height=50)
    assert pytest.approx(b1.iou(b4), 0.01) == 1 / 3


def test_text_region_detection_on_synthetic_document(synthetic_document_image: np.ndarray):
    """Test that morphological detector detects text lines on synthetic document."""
    detector = TextRegionDetector()
    boxes = detector.detect(synthetic_document_image)

    # Synthetic document has several lines of text; detector must find multiple text regions
    assert len(boxes) >= 4

    h, w = synthetic_document_image.shape[:2]
    for b in boxes:
        assert b.label == PIIType.TEXT.value
        assert b.width > 0
        assert b.height > 0
        assert b.x >= 0 and b.x2 <= w
        assert b.y >= 0 and b.y2 <= h
        assert b.area >= detector.min_area


def test_text_region_detection_on_blank_image():
    """Test that pure white or black image produces 0 text boxes."""
    detector = TextRegionDetector()
    blank = np.full((300, 300, 3), 255, dtype=np.uint8)
    boxes = detector.detect(blank)
    assert len(boxes) == 0

    black = np.zeros((300, 300, 3), dtype=np.uint8)
    assert len(detector.detect(black)) == 0


def test_face_detector_initialization_and_blank():
    """Test face detector initialization and detection on blank images."""
    detector = FaceDetector()
    assert detector.backend in ("dnn_yunet", "haar_cascade", "unavailable")

    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    faces = detector.detect(blank)
    assert len(faces) == 0


def test_pii_detector_pipeline(synthetic_document_image: np.ndarray):
    """Test unified PIIDetector pipeline orchestration."""
    pii = PIIDetector()

    # Detect only text
    res_text = pii.detect(synthetic_document_image, detect_faces=False, detect_text=True)
    assert res_text.face_count == 0
    assert res_text.text_count > 0
    assert len(res_text.boxes) == res_text.text_count
    assert res_text.processing_time_ms > 0

    # Detect only faces (document has no human face)
    res_face = pii.detect(synthetic_document_image, detect_faces=True, detect_text=False)
    assert res_face.text_count == 0
    assert res_face.face_count == 0
    assert len(res_face.boxes) == 0

    # Both enabled
    res_both = pii.detect(synthetic_document_image, detect_faces=True, detect_text=True)
    assert res_both.text_count > 0
    assert res_both.image_width == 800
    assert res_both.image_height == 600
