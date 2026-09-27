"""DocShield: High-Performance Image Anonymization & PII Redactor.

Enterprise-grade microservice and CLI for automated detection and redaction
of sensitive text, faces, and confidential regions on documents and images.
"""

__version__ = "0.1.0"
__author__ = "DocShield Team"
__license__ = "MIT"

from docshield.core.detector import FaceDetector, PIIDetector, TextRegionDetector
from docshield.core.redactor import ImageRedactor, decode_image, encode_image
from docshield.core.types import BoundingBox, DetectionResult, PIIType, RedactMode

__all__ = [
    "__version__",
    "BoundingBox",
    "DetectionResult",
    "PIIType",
    "RedactMode",
    "FaceDetector",
    "TextRegionDetector",
    "PIIDetector",
    "ImageRedactor",
    "encode_image",
    "decode_image",
]
