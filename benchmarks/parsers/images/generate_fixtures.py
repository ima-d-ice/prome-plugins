"""Generate deterministic synthetic image fixtures for OCR benchmarks."""

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

FIXTURES = Path(__file__).parent / "fixtures"


def make_clean_text() -> None:
    img = Image.new("RGB", (600, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Use default bitmap font
    draw.text((20, 40), "PROME RAG PLUGINS", fill=(0, 0, 0))
    draw.text((20, 80), "Optical Character Recognition Benchmark", fill=(0, 0, 0))
    draw.text((20, 120), "High accuracy text extraction", fill=(0, 0, 0))
    img.save(FIXTURES / "clean_text.png")


def make_receipt() -> None:
    img = Image.new("RGB", (400, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((30, 30), "STORE RECEIPT", fill=(0, 0, 0))
    draw.text((30, 70), "Item: Widget A  $15.00", fill=(0, 0, 0))
    draw.text((30, 100), "Item: Widget B  $25.00", fill=(0, 0, 0))
    draw.text((30, 140), "Subtotal: $40.00", fill=(0, 0, 0))
    draw.text((30, 170), "Tax: $3.20", fill=(0, 0, 0))
    draw.text((30, 210), "Total: $43.20", fill=(0, 0, 0))
    img.save(FIXTURES / "receipt.png")


def make_blank() -> None:
    img = Image.new("RGB", (300, 300), color=(255, 255, 255))
    img.save(FIXTURES / "blank.png")


def make_corrupt() -> None:
    (FIXTURES / "corrupt.png").write_bytes(b"NOT A PNG IMAGE")


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    make_clean_text()
    make_receipt()
    make_blank()
    make_corrupt()
    print(f"Generated image fixtures in {FIXTURES}")


if __name__ == "__main__":
    main()
