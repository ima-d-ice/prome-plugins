"""PaddleOCR parser: deep-learning OCR extraction with bounding polygons and reading order."""

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


def _load_paddleocr():
    try:
        from PIL import Image, UnidentifiedImageError
        from paddleocr import PaddleOCR
        import paddleocr

        return PaddleOCR, Image, UnidentifiedImageError, paddleocr
    except ImportError as exc:
        raise ParserDependencyError(
            "PaddleOCR or paddlepaddle is not installed. Install it with:\n"
            "  pip install 'prome-rag-plugins[paddleocr]'\n"
            "or:\n"
            "  pip install paddlepaddle paddleocr\n"
            "Note: PaddleOCR is a heavy deep-learning dependency and downloads model weights on first run."
        ) from exc


class PaddleOCRParser:
    """Extract text and bounding regions from raster images using PaddleOCR."""

    name = "paddleocr"

    def __init__(self, lang: str = "en", use_angle_cls: bool = True) -> None:
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self._ocr = None

    def _get_ocr(self, PaddleOCR):
        if self._ocr is None:
            self._ocr = PaddleOCR(use_angle_cls=self.use_angle_cls, lang=self.lang)
        return self._ocr

    def parse(self, source: str | Path) -> ParsedDocument:
        PaddleOCR, Image, UnidentifiedImageError, paddleocr_mod = _load_paddleocr()
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

        ocr = self._get_ocr(PaddleOCR)
        try:
            result = ocr.ocr(str(path), cls=self.use_angle_cls)
        except Exception as exc:
            raise CorruptDocumentError(f"PaddleOCR processing failed for '{path}': {exc}") from exc

        blocks: list[ParsedBlock] = []

        if result and result[0]:
            for line_idx, line in enumerate(result[0]):
                poly, (text, conf) = line
                text = text.strip()
                if not text:
                    continue

                xs = [pt[0] for pt in poly]
                ys = [pt[1] for pt in poly]
                bbox = BoundingBox(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))

                blocks.append(
                    ParsedBlock(
                        block_type=BlockType.TEXT,
                        content=text,
                        page_number=1,
                        bbox=bbox,
                        metadata={
                            "confidence": round(float(conf) * 100.0, 2),
                            "polygon": poly,
                            "line_index": line_idx,
                            "image_width": img_w,
                            "image_height": img_h,
                        },
                    )
                )

        if not blocks:
            raise EmptyDocumentError(f"PaddleOCR detected no text in image '{path}'.")

        page = ParsedPage(
            page_number=1,
            width=float(img_w),
            height=float(img_h),
            blocks=blocks,
        )

        metadata = {
            "image_format": img.format,
            "width": img_w,
            "height": img_h,
            "ocr_engine": "paddleocr",
            "ocr_lang": self.lang,
            "block_count": len(blocks),
        }

        return ParsedDocument(
            source=str(path),
            parser_name=self.name,
            parser_version=str(getattr(paddleocr_mod, "__version__", "unknown")),
            metadata=metadata,
            pages=[page],
            blocks=[],
            warnings=[],
        )
