"""Type definitions and data transfer objects for DocShield."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class RedactMode(str, Enum):
    """Supported masking modes for redacted areas."""
    BLUR = "blur"
    PIXELATE = "pixelate"


class PIIType(str, Enum):
    """Types of detected sensitive information."""
    FACE = "face"
    TEXT = "text"
    CUSTOM = "custom"


class BoundingBox(BaseModel):
    """Represents a 2D bounding box for detected PII."""
    x: int = Field(ge=0, description="Top-left X coordinate")
    y: int = Field(ge=0, description="Top-left Y coordinate")
    width: int = Field(gt=0, description="Box width in pixels")
    height: int = Field(gt=0, description="Box height in pixels")
    label: str = Field(default=PIIType.TEXT.value, description="Classification label")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence score")

    @property
    def x2(self) -> int:
        """Bottom-right X coordinate."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Bottom-right Y coordinate."""
        return self.y + self.height

    @property
    def area(self) -> int:
        """Area of the bounding box in square pixels."""
        return self.width * self.height

    def to_xyxy(self) -> Tuple[int, int, int, int]:
        """Return box coordinates as (x1, y1, x2, y2)."""
        return (self.x, self.y, self.x2, self.y2)

    def to_xywh(self) -> Tuple[int, int, int, int]:
        """Return box coordinates as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)

    def iou(self, other: BoundingBox) -> float:
        """Calculate Intersection over Union (IoU) with another bounding box."""
        ix1 = max(self.x, other.x)
        iy1 = max(self.y, other.y)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        intersection = iw * ih

        if intersection == 0:
            return 0.0

        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


class DetectionResult(BaseModel):
    """Encapsulates detection analysis results."""
    boxes: List[BoundingBox] = Field(default_factory=list, description="List of detected bounding boxes")
    image_width: int = Field(ge=1, description="Original image width")
    image_height: int = Field(ge=1, description="Original image height")
    processing_time_ms: float = Field(ge=0.0, description="Detection duration in milliseconds")
    face_count: int = Field(ge=0, description="Number of detected faces")
    text_count: int = Field(ge=0, description="Number of detected text regions")
