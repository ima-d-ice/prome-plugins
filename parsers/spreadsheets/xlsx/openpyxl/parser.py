"""openpyxl-based XLSX parser: structure-preserving Excel extraction.

Preserves sheet order, cell coordinates, data types, formulas, merged cells,
headers, and row/column relationships without lossy flattening.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from parsers.contracts import (
    BlockType,
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedBlock,
    ParsedDocument,
    ParsedPage,
    ParserDependencyError,
)
from parsers.contracts.validation import validate_source

logger = logging.getLogger(__name__)


def _load_openpyxl():
    try:
        import openpyxl

        return openpyxl
    except ImportError as exc:
        raise ParserDependencyError(
            "openpyxl is not installed. Install it with:\n"
            "  pip install 'prome-rag-plugins[spreadsheets]'\n"
            "or:\n"
            "  pip install openpyxl"
        ) from exc


def _cell_to_str(val: Any) -> str:
    if val is None:
        return ""
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return str(val)


class OpenpyxlParser:
    """Extract structured spreadsheet workbooks using openpyxl."""

    name = "openpyxl"

    def parse(self, source: str | Path) -> ParsedDocument:
        openpyxl_mod = _load_openpyxl()
        path = validate_source(source)

        try:
            wb = openpyxl_mod.load_workbook(str(path), data_only=False)
        except Exception as exc:
            raise CorruptDocumentError(f"Could not open '{path}' as an XLSX file: {exc}") from exc

        # Workbook metadata
        wb_meta: dict[str, Any] = {
            "sheet_names": wb.sheetnames,
            "sheet_count": len(wb.sheetnames),
        }
        props = getattr(wb, "properties", None)
        if props:
            for attr in ("creator", "lastModifiedBy", "title", "subject", "category"):
                val = getattr(props, attr, None)
                if val:
                    wb_meta[attr] = str(val)
            for attr in ("created", "modified"):
                val = getattr(props, attr, None)
                if val:
                    wb_meta[attr] = val.isoformat() if hasattr(val, "isoformat") else str(val)

        pages: list[ParsedPage] = []
        total_data_cells = 0

        for sheet_idx, sheet_name in enumerate(wb.sheetnames, start=1):
            ws = wb[sheet_name]
            merged = [str(r) for r in ws.merged_cells.ranges]
            visibility = getattr(ws.sheet_state, "value", str(ws.sheet_state))

            rows_data: list[list[Any]] = []
            formulas: dict[str, str] = {}
            cell_types: dict[str, str] = {}
            coords_map: dict[str, Any] = {}

            for row in ws.iter_rows():
                row_vals = []
                for cell in row:
                    val = cell.value
                    coord = cell.coordinate
                    c_type = cell.data_type
                    cell_types[coord] = str(c_type)
                    if c_type == "f" or (isinstance(val, str) and val.startswith("=")):
                        formulas[coord] = str(val)
                    if val is not None:
                        total_data_cells += 1
                        coords_map[coord] = val
                    row_vals.append(_cell_to_str(val))
                if any(row_vals):
                    rows_data.append(row_vals)

            if not rows_data:
                continue

            headers = rows_data[0]
            # Construct markdown representation
            md_lines = [" | ".join(headers)]
            md_lines.append(" | ".join(["---"] * len(headers)))
            for r in rows_data[1:]:
                md_lines.append(" | ".join(r))
            md_content = "\n".join(md_lines)

            sheet_block_meta: dict[str, Any] = {
                "sheet_name": sheet_name,
                "sheet_index": sheet_idx,
                "rows": len(rows_data),
                "cols": max(len(r) for r in rows_data),
                "headers": headers,
                "has_header": bool(headers and any(headers)),
                "row_cells": rows_data,
                "dimensions": ws.dimensions,
                "merged_cells": merged,
                "formulas": formulas,
                "cell_types": cell_types,
                "visibility": visibility,
            }

            block = ParsedBlock(
                block_type=BlockType.TABLE,
                content=md_content,
                page_number=sheet_idx,
                metadata=sheet_block_meta,
            )

            pages.append(
                ParsedPage(
                    page_number=sheet_idx,
                    blocks=[block],
                )
            )

        if not pages or total_data_cells == 0:
            raise EmptyDocumentError(f"XLSX '{path}' contains no data cells or extractable content.")

        return ParsedDocument(
            source=str(path),
            parser_name=self.name,
            parser_version=str(getattr(openpyxl_mod, "__version__", "unknown")),
            metadata=wb_meta,
            pages=pages,
            blocks=[],
            warnings=[],
        )
