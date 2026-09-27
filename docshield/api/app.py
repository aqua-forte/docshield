"""FastAPI microservice for DocShield PII redaction and privacy compliance."""

from __future__ import annotations

import time
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from docshield.core.detector import PIIDetector
from docshield.core.redactor import ImageRedactor, decode_image, encode_image
from docshield.core.types import DetectionResult, RedactMode

app = FastAPI(
    title="DocShield API",
    description="High-Performance Microservice for Automated PII Detection and Redaction on Images & Documents.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for cross-origin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton detector and redactor instances
_detector: Optional[PIIDetector] = None
_redactor: Optional[ImageRedactor] = None


def get_detector() -> PIIDetector:
    """Lazy-load or reuse detector instance."""
    global _detector
    if _detector is None:
        _detector = PIIDetector()
    return _detector


def get_redactor() -> ImageRedactor:
    """Lazy-load or reuse redactor instance."""
    global _redactor
    if _redactor is None:
        _redactor = ImageRedactor()
    return _redactor


@app.get("/health", tags=["System"])
async def health_check():
    """Health status and detector readiness probe."""
    try:
        det = get_detector()
        face_ready = det.face_detector.backend != "unavailable"
        backend = det.face_detector.backend
    except Exception:
        face_ready = False
        backend = "unavailable"

    return {
        "status": "healthy",
        "service": "docshield",
        "version": "0.1.0",
        "face_detector_ready": face_ready,
        "face_detector_backend": backend,
        "text_detector_ready": True,
    }


@app.post(
    "/redact",
    tags=["Anonymization"],
    summary="Detect and redact sensitive PII regions on uploaded image",
    response_class=Response,
    responses={
        200: {
            "content": {"image/jpeg": {}, "image/png": {}},
            "description": "Returns sanitized binary image stream with PII obfuscated.",
        },
        400: {"description": "Invalid image payload or corrupted file."},
    },
)
async def redact_endpoint(
    file: UploadFile = File(..., description="Target image file (JPEG, PNG, WEBP, BMP)"),
    mode: RedactMode = Form(RedactMode.BLUR, description="Redaction mode: 'blur' or 'pixelate'"),
    detect_faces: bool = Form(True, description="Enable automated face detection"),
    detect_text: bool = Form(True, description="Enable document text detection"),
    blur_intensity: Optional[int] = Form(None, description="Custom blur kernel size (odd integer)"),
    pixel_size: Optional[int] = Form(None, description="Custom pixel mosaic block size"),
    output_format: str = Form("jpeg", description="Output image encoding: 'jpeg' or 'png'"),
):
    """Process uploaded image and return anonymized result."""
    start_total = time.perf_counter()

    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    try:
        image = decode_image(contents)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decode image: {str(e)}",
        )

    detector = get_detector()
    redactor = get_redactor()

    # Run detection
    result = detector.detect(
        image,
        detect_faces=detect_faces,
        detect_text=detect_text,
    )

    # Determine intensity parameter
    intensity = pixel_size if mode == RedactMode.PIXELATE else blur_intensity

    # Apply redaction
    redacted_image = redactor.redact(
        image,
        boxes=result.boxes,
        mode=mode,
        intensity=intensity,
    )

    # Format encoding
    fmt = output_format.lower()
    media_type = "image/png" if fmt == "png" else "image/jpeg"
    encoded_bytes = encode_image(redacted_image, format=fmt)

    total_time_ms = (time.perf_counter() - start_total) * 1000.0

    headers = {
        "X-DocShield-Faces-Redacted": str(result.face_count),
        "X-DocShield-Text-Redacted": str(result.text_count),
        "X-DocShield-Total-Redacted": str(len(result.boxes)),
        "X-DocShield-Mode": mode.value,
        "X-DocShield-Processing-Time-Ms": f"{total_time_ms:.2f}",
    }

    return Response(content=encoded_bytes, media_type=media_type, headers=headers)


@app.post(
    "/detect",
    response_model=DetectionResult,
    tags=["Inspection"],
    summary="Inspect and return bounding box coordinates of sensitive regions without modifying image",
)
async def detect_endpoint(
    file: UploadFile = File(..., description="Target image file"),
    detect_faces: bool = Form(True, description="Enable face detection"),
    detect_text: bool = Form(True, description="Enable text region detection"),
):
    """Inspect image and return detected bounding boxes with coordinates and confidence."""
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    try:
        image = decode_image(contents)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decode image: {str(e)}",
        )

    detector = get_detector()
    result = detector.detect(image, detect_faces=detect_faces, detect_text=detect_text)
    return result
