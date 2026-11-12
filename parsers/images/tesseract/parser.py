"""Tesseract OCR parser: image text extraction with bounding boxes and confidence scores."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from parsers.contracts import (
    BlockType,
    BoundingBox,
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedBlock,
    ParsedDocument,
    ParsedPage,
    ParserDependencyError,
)
from parsers.contracts.validation import validate_source

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


def _load_tesseract():
    try:
        from PIL import Image, UnidentifiedImageError
        import pytesseract

        # Verify tesseract binary is actually found
        pytesseract.get_tesseract_version()
        return pytesseract, Image, UnidentifiedImageError
    except ImportError as exc:
        raise ParserDependencyError(
            "pytesseract or Pillow is not installed. Install with:\n"
            "  pip install 'prome-rag-plugins[images]'\n"
            "or:\n"
            "  pip install pytesseract pillow"
        ) from exc
    except Exception as exc:
        raise ParserDependencyError(
            f"Tesseract OCR executable was not found on your system PATH: {exc}.\n"
            "Install tesseract (e.g. 'brew install tesseract' or 'apt-get install tesseract-ocr')."
        ) from exc


class TesseractParser:
    """Extract text and bounding boxes from raster images using Tesseract OCR."""

    name = "tesseract_ocr"

    def __init__(self, lang: str = "eng", psm: int = 3) -> None:
        self.lang = lang
        self.psm = psm

    def parse(self, source: str | Path) -> ParsedDocument:
        pytesseract, Image, UnidentifiedImageError = _load_tesseract()
        path = validate_source(source)

        if path.stat().st_size == 0:
            raise EmptyDocumentError(f"Image file '{path}' is empty (0 bytes).")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise CorruptDocumentError(f"Unsupported image format '{ext}' for file '{path}'.")

        try:
            img = Image.open(str(path))
            img_w, img_h = img.size
        except UnidentifiedImageError as exc:
            raise CorruptDocumentError(f"Cannot decode image file '{path}': {exc}") from exc
        except Exception as exc:
            raise CorruptDocumentError(f"Error opening image file '{path}': {exc}") from exc

        config = f"--psm {self.psm}"
        try:
            data = pytesseract.image_to_data(
                img,
                lang=self.lang,
                config=config,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:
            raise CorruptDocumentError(f"Tesseract OCR processing failed for '{path}': {exc}") from exc

        # Group words by line
        lines_map: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
        n_boxes = len(data["text"])

        for i in range(n_boxes):
            text = str(data["text"][i]).strip()
            conf = float(data["conf"][i])
            if not text or conf < 0:
                continue

            block_num = data["block_num"][i]
            par_num = data["par_num"][i]
            line_num = data["line_num"][i]
            key = (block_num, par_num, line_num)

            word_entry = {
                "text": text,
                "conf": conf,
                "left": float(data["left"][i]),
                "top": float(data["top"][i]),
                "width": float(data["width"][i]),
                "height": float(data["height"][i]),
            }
            lines_map.setdefault(key, []).append(word_entry)

        blocks: list[ParsedBlock] = []

        for key, words in lines_map.items():
            line_text = " ".join(w["text"] for w in words)
            min_x0 = min(w["left"] for w in words)
            min_y0 = min(w["top"] for w in words)
            max_x1 = max(w["left"] + w["width"] for w in words)
            max_y1 = max(w["top"] + w["height"] for w in words)
            avg_conf = sum(w["conf"] for w in words) / len(words)

            bbox = BoundingBox(x0=min_x0, y0=min_y0, x1=max_x1, y1=max_y1)
            blocks.append(
                ParsedBlock(
                    block_type=BlockType.TEXT,
                    content=line_text,
                    page_number=1,
                    bbox=bbox,
                    metadata={
                        "confidence": round(avg_conf, 2),
                        "words": [w["text"] for w in words],
                        "word_count": len(words),
                        "image_width": img_w,
                        "image_height": img_h,
                    },
                )
            )

        if not blocks:
            raise EmptyDocumentError(f"Tesseract detected no readable text in image '{path}'.")

        page = ParsedPage(
            page_number=1,
            width=float(img_w),
            height=float(img_h),
            blocks=blocks,
        )

        metadata = {
            "image_format": img.format,
            "image_mode": img.mode,
            "width": img_w,
            "height": img_h,
            "ocr_engine": "tesseract",
            "ocr_lang": self.lang,
            "block_count": len(blocks),
        }

        return ParsedDocument(
            source=str(path),
            parser_name=self.name,
            parser_version=str(getattr(pytesseract, "__version__", "unknown")),
            metadata=metadata,
            pages=[page],
            blocks=[],
            warnings=[],
        )
