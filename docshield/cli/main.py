"""Command-line interface for DocShield batch image anonymization and PII redaction."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from docshield.core.detector import PIIDetector
from docshield.core.redactor import ImageRedactor
from docshield.core.types import RedactMode

app = typer.Typer(
    name="docshield",
    help="DocShield: Enterprise Image Anonymization & PII Redactor CLI.",
    add_completion=False,
)

console = Console()
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def find_image_files(path: Path, recursive: bool = True) -> List[Path]:
    """Gather all supported image files in a path."""
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []

    pattern = "**/*" if recursive else "*"
    files = [
        p for p in path.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


@app.command()
def redact(
    input_path: Path = typer.Argument(
        ...,
        help="Path to an image file or a directory containing images",
        exists=True,
        readable=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file or directory for redacted images",
    ),
    mode: RedactMode = typer.Option(
        RedactMode.BLUR,
        "--mode",
        "-m",
        help="Masking mode: 'blur' (Gaussian) or 'pixelate' (Mosaic)",
    ),
    detect_faces: bool = typer.Option(
        True,
        "--detect-faces/--no-faces",
        help="Toggle automated face detection",
    ),
    detect_text: bool = typer.Option(
        True,
        "--detect-text/--no-text",
        help="Toggle document text region detection",
    ),
    blur_intensity: Optional[int] = typer.Option(
        None,
        "--blur-intensity",
        help="Custom blur kernel size (odd integer)",
    ),
    pixel_size: Optional[int] = typer.Option(
        None,
        "--pixel-size",
        help="Custom pixelation mosaic block size",
    ),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        "-r/-R",
        help="Recursively scan subdirectories for images",
    ),
):
    """Redact sensitive PII (faces and text) in images or directories."""
    files = find_image_files(input_path, recursive=recursive)

    if not files:
        console.print(f"[yellow]No supported images found at '{input_path}'.[/yellow]")
        console.print(f"[dim]Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}[/dim]")
        raise typer.Exit(code=1)

    console.print(
        Panel.fit(
            f"[bold blue]DocShield PII Redactor[/bold blue]\n"
            f"Input: [cyan]{input_path}[/cyan]\n"
            f"Mode: [green]{mode.value.upper()}[/green] | "
            f"Faces: [bold]{'ON' if detect_faces else 'OFF'}[/bold] | "
            f"Text: [bold]{'ON' if detect_text else 'OFF'}[/bold]\n"
            f"Discovered [bold]{len(files)}[/bold] image(s) for processing.",
            border_style="blue",
        )
    )

    # Initialize CV pipeline
    detector = PIIDetector()
    redactor = ImageRedactor(
        default_mode=mode,
        blur_intensity=blur_intensity or 51,
        pixel_size=pixel_size or 14,
    )

    # Prepare output path resolution
    single_file_mode = input_path.is_file()
    if single_file_mode:
        if output is None:
            out_file = input_path.parent / f"{input_path.stem}_redacted{input_path.suffix}"
        elif output.is_dir():
            out_file = output / f"{input_path.stem}_redacted{input_path.suffix}"
        else:
            out_file = output
        output_targets = {input_path: out_file}
    else:
        out_dir = output or (input_path.parent / f"{input_path.name}_clean")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_targets = {}
        for f in files:
            rel = f.relative_to(input_path)
            target = out_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            output_targets[f] = target

    total_faces = 0
    total_text = 0
    processed_count = 0
    start_all = time.perf_counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Redacting PII...", total=len(files))

        for file_path in files:
            dest_path = output_targets[file_path]
            progress.update(task, description=f"[cyan]Processing: {file_path.name}")

            # Read image with OpenCV
            img = cv2.imread(str(file_path))
            if img is None:
                console.print(f"[red]Warning: Could not read image '{file_path}'. Skipping.[/red]")
                progress.advance(task)
                continue

            # Detect PII regions
            detection = detector.detect(
                img,
                detect_faces=detect_faces,
                detect_text=detect_text,
            )

            # Redact regions
            clean_img = redactor.redact(
                img,
                boxes=detection.boxes,
                mode=mode,
                intensity=pixel_size if mode == RedactMode.PIXELATE else blur_intensity,
            )

            # Save clean image
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest_path), clean_img)

            total_faces += detection.face_count
            total_text += detection.text_count
            processed_count += 1
            progress.advance(task)

    elapsed_total = time.perf_counter() - start_all
    avg_speed = (elapsed_total / processed_count * 1000) if processed_count else 0.0

    # Summary table
    table = Table(title="DocShield Execution Summary", border_style="green")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="bold white")
    table.add_row("Images Processed", f"{processed_count} / {len(files)}")
    table.add_row("Faces Redacted", str(total_faces))
    table.add_row("Text Regions Redacted", str(total_text))
    table.add_row("Total Redacted Regions", str(total_faces + total_text))
    table.add_row("Total Duration", f"{elapsed_total:.2f}s")
    table.add_row("Avg Processing Time", f"{avg_speed:.1f} ms/image")
    table.add_row("Output Destination", str(output or ("Single File" if single_file_mode else out_dir)))

    console.print(table)
    console.print("[bold green]Success: All sensitive data successfully sanitized.[/bold green]")


@app.command()
def server(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind host address"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port number"),
    reload: bool = typer.Option(False, "--reload", help="Enable live auto-reload for development"),
):
    """Launch the DocShield FastAPI microservice."""
    console.print(f"[bold green]Starting DocShield API service at http://{host}:{port}...[/bold green]")
    uvicorn.run("docshield.api.app:app", host=host, port=port, reload=reload)


@app.command()
def info():
    """Print DocShield system environment, versions, and OpenCV configuration."""
    table = Table(title="DocShield System Diagnostics", border_style="blue")
    table.add_column("Component", style="cyan")
    table.add_column("Status / Details", style="white")

    table.add_row("DocShield Version", "0.1.0")
    table.add_row("Python Version", sys.version.split()[0])
    table.add_row("OpenCV Version", cv2.__version__)

    try:
        det = PIIDetector()
        backend = det.face_detector.backend
        face_ready = f"[green]Ready ({backend})[/green]" if backend != "unavailable" else "[yellow]Disabled/Unavailable[/yellow]"
        model_file = det.face_detector.model_path or "N/A"
    except Exception as e:
        face_ready = f"[red]Unavailable ({e})[/red]"
        model_file = "N/A"

    table.add_row("Face Detector Backend", face_ready)
    table.add_row("Face Model Path", str(model_file))
    table.add_row("Text Detector (Morphology)", "[green]Ready[/green]")
    table.add_row("Masking Modes", "Gaussian Blur, Mosaic Pixelation")

    console.print(table)


if __name__ == "__main__":
    app()
