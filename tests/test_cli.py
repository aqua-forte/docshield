"""Tests for DocShield Typer CLI interface."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from docshield.cli.main import app

runner = CliRunner()


def test_cli_info():
    """Test 'docshield info' diagnostic command."""
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "DocShield Version" in result.stdout
    assert "OpenCV Version" in result.stdout
    assert "Text Detector" in result.stdout


def test_cli_redact_single_file(tmp_path: Path, synthetic_document_image: np.ndarray):
    """Test 'docshield redact' processing a single image."""
    in_file = tmp_path / "sample_doc.jpg"
    out_file = tmp_path / "clean_doc.jpg"
    cv2.imwrite(str(in_file), synthetic_document_image)

    result = runner.invoke(
        app,
        ["redact", str(in_file), "--output", str(out_file), "--mode", "blur"],
    )

    assert result.exit_code == 0
    assert "Success: All sensitive data successfully sanitized" in result.stdout
    assert out_file.exists()

    # Read output image and verify valid shape
    read_back = cv2.imread(str(out_file))
    assert read_back is not None
    assert read_back.shape == synthetic_document_image.shape


def test_cli_redact_directory_batch(tmp_path: Path, synthetic_document_image: np.ndarray):
    """Test 'docshield redact' processing an entire directory in batch mode."""
    in_dir = tmp_path / "input_batch"
    out_dir = tmp_path / "clean_batch"
    in_dir.mkdir()

    # Create 3 image files
    for idx in range(3):
        f = in_dir / f"doc_{idx}.png"
        cv2.imwrite(str(f), synthetic_document_image)

    result = runner.invoke(
        app,
        ["redact", str(in_dir), "--output", str(out_dir), "--mode", "pixelate"],
    )

    assert result.exit_code == 0
    assert out_dir.exists()
    assert (out_dir / "doc_0.png").exists()
    assert (out_dir / "doc_1.png").exists()
    assert (out_dir / "doc_2.png").exists()
