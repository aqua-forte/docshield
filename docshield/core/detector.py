"""PII and sensitive region detection engine utilizing OpenCV and morphological analysis."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Suppress internal OpenCV DNN graph engine warnings
if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

from docshield.core.types import BoundingBox, DetectionResult, PIIType


class FaceDetector:
    """Multi-backend face detector supporting OpenCV DNN (YuNet) and Haar Cascades."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
    ):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.backend = None
        self.dnn_detector = None
        self.haar_classifier = None

        # 1. Try OpenCV DNN YuNet first (SOTA fast face detection)
        yunet_path = model_path or self._resolve_yunet_model()
        if hasattr(cv2, "FaceDetectorYN_create") and yunet_path and os.path.exists(yunet_path):
            try:
                self.dnn_detector = cv2.FaceDetectorYN_create(
                    model=yunet_path,
                    config="",
                    input_size=(320, 320),
                    score_threshold=score_threshold,
                    nms_threshold=nms_threshold,
                    top_k=5000,
                )
                self.backend = "dnn_yunet"
                self.model_path = yunet_path
                return
            except Exception:
                self.dnn_detector = None

        # 2. Try OpenCV Haar Cascade (OpenCV 4.x or legacy fallback)
        if hasattr(cv2, "CascadeClassifier"):
            cascade_xml = model_path or self._resolve_haar_cascade()
            if cascade_xml and os.path.exists(cascade_xml):
                try:
                    classifier = cv2.CascadeClassifier(cascade_xml)
                    if not classifier.empty():
                        self.haar_classifier = classifier
                        self.backend = "haar_cascade"
                        self.model_path = cascade_xml
                        return
                except Exception:
                    self.haar_classifier = None

        # If neither loaded
        self.backend = "unavailable"
        self.model_path = None

    @staticmethod
    def _resolve_yunet_model() -> Optional[str]:
        """Resolve bundled YuNet ONNX face detection model."""
        base_dir = Path(__file__).resolve().parent.parent / "data" / "models"
        onnx_file = base_dir / "face_detection_yunet_2023mar.onnx"
        if onnx_file.exists():
            return str(onnx_file)
        return None

    @staticmethod
    def _resolve_haar_cascade() -> Optional[str]:
        """Resolve bundled or OpenCV system Haar Cascade XML."""
        local_dir = Path(__file__).resolve().parent.parent / "data" / "haarcascades"
        bundled_xml = local_dir / "haarcascade_frontalface_default.xml"
        if bundled_xml.exists():
            return str(bundled_xml)

        if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            cv2_xml = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if cv2_xml.exists():
                return str(cv2_xml)

        return None

    def detect(
        self,
        image: np.ndarray,
        min_size: Tuple[int, int] = (20, 20),
    ) -> List[BoundingBox]:
        """Detect faces in the given image.

        Args:
            image: Input image as numpy ndarray.
            min_size: Minimum face bounding box size (w, h).

        Returns:
            List of detected face BoundingBox objects.
        """
        if image is None or image.size == 0 or self.backend == "unavailable":
            return []

        h, w = image.shape[:2]
        boxes: List[BoundingBox] = []

        # Backend 1: OpenCV DNN YuNet
        if self.backend == "dnn_yunet" and self.dnn_detector is not None:
            # Ensure 3-channel BGR for DNN input
            if len(image.shape) == 2:
                bgr_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                bgr_img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            else:
                bgr_img = image

            self.dnn_detector.setInputSize((w, h))
            _, raw_faces = self.dnn_detector.detect(bgr_img)

            if raw_faces is not None:
                for face in raw_faces:
                    fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    confidence = float(face[14]) if len(face) > 14 else 0.9

                    # Bounds clamping
                    fx = max(0, min(fx, w - 1))
                    fy = max(0, min(fy, h - 1))
                    fw = min(fw, w - fx)
                    fh = min(fh, h - fy)

                    if fw >= min_size[0] and fh >= min_size[1]:
                        boxes.append(
                            BoundingBox(
                                x=fx,
                                y=fy,
                                width=fw,
                                height=fh,
                                label=PIIType.FACE.value,
                                confidence=round(confidence, 3),
                            )
                        )
            return boxes

        # Backend 2: Haar Cascade Classifier (OpenCV 4 fallback)
        if self.backend == "haar_cascade" and self.haar_classifier is not None:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            equalized = cv2.equalizeHist(gray)
            raw_faces = self.haar_classifier.detectMultiScale(
                equalized,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=min_size,
                flags=cv2.CASCADE_SCALE_IMAGE if hasattr(cv2, "CASCADE_SCALE_IMAGE") else 0,
            )
            if len(raw_faces) > 0:
                for (x, y, bw, bh) in raw_faces:
                    boxes.append(
                        BoundingBox(
                            x=int(x),
                            y=int(y),
                            width=int(bw),
                            height=int(bh),
                            label=PIIType.FACE.value,
                            confidence=0.92,
                        )
                    )
            return boxes

        return []


class TextRegionDetector:
    """Document text region detector using morphological operations and spatial filters."""

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (19, 3),
        min_area: int = 60,
        min_width: int = 12,
        min_height: int = 6,
        padding: int = 2,
    ):
        """Initialize morphological text detector.

        Args:
            kernel_size: Rectangular structuring element size (width, height) for horizontal grouping.
            min_area: Minimum bounding box area in pixels to eliminate speckle noise.
            min_width: Minimum width in pixels.
            min_height: Minimum height in pixels.
            padding: Padding in pixels added around detected text regions.
        """
        self.kernel_size = kernel_size
        self.min_area = min_area
        self.min_width = min_width
        self.min_height = min_height
        self.padding = padding

    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        """Detect text regions on documents/images using gradient morphology.

        Pipeline:
        1. Grayscale conversion.
        2. High-pass filter via Sobel gradient along X axis (captures character vertical edges).
        3. Otsu automatic binarization.
        4. Morphological rectangular dilation to merge adjacent characters into lines.
        5. Contour extraction and geometric filtering.
        """
        if image is None or image.size == 0:
            return []

        h_img, w_img = image.shape[:2]
        total_pixels = h_img * w_img

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Slight Gaussian blur to suppress fine scanner/sensor noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Sobel gradient along X axis (vertical strokes of text characters)
        grad_x = cv2.Sobel(blurred, cv2.CV_8U, dx=1, dy=0, ksize=3)

        # Otsu thresholding to segment high-gradient text structures
        _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological horizontal closing/dilation to connect letters into continuous word/line blocks
        struct_elem = cv2.getStructuringElement(cv2.MORPH_RECT, self.kernel_size)
        connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, struct_elem)

        # Minor vertical dilation to merge tight stacked accents/lines
        vert_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        connected = cv2.dilate(connected, vert_elem, iterations=1)

        # Find external contours
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_boxes: List[BoundingBox] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h

            # Filter out minuscule noise or gigantic artifacts covering the full page
            if (
                w >= self.min_width
                and h >= self.min_height
                and area >= self.min_area
                and area < (total_pixels * 0.90)
            ):
                # Apply padding with edge clamping
                px = max(0, x - self.padding)
                py = max(0, y - self.padding)
                pw = min(w_img - px, w + 2 * self.padding)
                ph = min(h_img - py, h + 2 * self.padding)

                raw_boxes.append(
                    BoundingBox(
                        x=int(px),
                        y=int(py),
                        width=int(pw),
                        height=int(ph),
                        label=PIIType.TEXT.value,
                        confidence=0.85,
                    )
                )

        # Merge overlapping/adjacent text boxes
        return self._merge_overlapping_boxes(raw_boxes)

    @staticmethod
    def _merge_overlapping_boxes(boxes: List[BoundingBox], iou_thresh: float = 0.1) -> List[BoundingBox]:
        """Merge overlapping bounding boxes to create clean redaction zones."""
        if not boxes:
            return []

        # Sort boxes by top-left coordinate (y, then x)
        sorted_boxes = sorted(boxes, key=lambda b: (b.y, b.x))
        merged: List[BoundingBox] = []

        for box in sorted_boxes:
            matched = False
            for i, existing in enumerate(merged):
                # Check for overlap or containment
                if existing.iou(box) > iou_thresh or TextRegionDetector._boxes_intersect(existing, box):
                    # Combine coordinates
                    nx = min(existing.x, box.x)
                    ny = min(existing.y, box.y)
                    nx2 = max(existing.x2, box.x2)
                    ny2 = max(existing.y2, box.y2)

                    merged[i] = BoundingBox(
                        x=nx,
                        y=ny,
                        width=nx2 - nx,
                        height=ny2 - ny,
                        label=PIIType.TEXT.value,
                        confidence=max(existing.confidence, box.confidence),
                    )
                    matched = True
                    break

            if not matched:
                merged.append(box)

        return merged

    @staticmethod
    def _boxes_intersect(b1: BoundingBox, b2: BoundingBox, margin: int = 4) -> bool:
        """Check if two boxes intersect or are in close proximity within margin."""
        return not (
            b1.x2 + margin < b2.x
            or b2.x2 + margin < b1.x
            or b1.y2 + margin < b2.y
            or b2.y2 + margin < b1.y
        )


class PIIDetector:
    """Unified detector pipeline combining face detection and document text redaction."""

    def __init__(
        self,
        face_model_path: Optional[str] = None,
        text_kernel: Tuple[int, int] = (19, 3),
        min_text_area: int = 60,
    ):
        self.face_detector = FaceDetector(model_path=face_model_path)
        self.text_detector = TextRegionDetector(
            kernel_size=text_kernel,
            min_area=min_text_area,
        )

    def detect(
        self,
        image: np.ndarray,
        detect_faces: bool = True,
        detect_text: bool = True,
    ) -> DetectionResult:
        """Execute detection pipeline on input image.

        Args:
            image: Image as numpy ndarray.
            detect_faces: Whether to run face detection.
            detect_text: Whether to run text region detection.

        Returns:
            DetectionResult containing identified bounding boxes and metrics.
        """
        start_time = time.perf_counter()

        if image is None or image.size == 0:
            return DetectionResult(
                boxes=[],
                image_width=0,
                image_height=0,
                processing_time_ms=0.0,
                face_count=0,
                text_count=0,
            )

        h, w = image.shape[:2]
        detected_boxes: List[BoundingBox] = []
        face_count = 0
        text_count = 0

        if detect_faces:
            face_boxes = self.face_detector.detect(image)
            face_count = len(face_boxes)
            detected_boxes.extend(face_boxes)

        if detect_text:
            text_boxes = self.text_detector.detect(image)
            text_count = len(text_boxes)
            detected_boxes.extend(text_boxes)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return DetectionResult(
            boxes=detected_boxes,
            image_width=w,
            image_height=h,
            processing_time_ms=round(elapsed_ms, 2),
            face_count=face_count,
            text_count=text_count,
        )
