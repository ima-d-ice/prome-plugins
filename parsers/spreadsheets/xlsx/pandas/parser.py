"""pandas-based XLSX parser: tabular and DataFrame-oriented Excel extraction.

Prioritizes tabular schema extraction, column data types, missing value statistics,
and DataFrame serialization. Does not preserve raw formulas, merged cell spans,
or visual cell formatting.
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


def _load_pandas():
    try:
        import pandas as pd

        return pd
    except ImportError as exc:
        raise ParserDependencyError(
            "pandas is not installed. Install it with:\n"
            "  pip install 'prome-rag-plugins[spreadsheets]'\n"
            "or:\n"
            "  pip install pandas openpyxl"
        ) from exc


class PandasXlsxParser:
    """Extract tabular Excel content into DataFrames using pandas."""

    name = "pandas_xlsx"

    def parse(self, source: str | Path) -> ParsedDocument:
        pd = _load_pandas()
        path = validate_source(source)

        try:
            dfs = pd.read_excel(str(path), sheet_name=None)
        except Exception as exc:
            raise CorruptDocumentError(f"Could not open '{path}' as an Excel file: {exc}") from exc

        if not dfs or all(df.empty for df in dfs.values()):
            raise EmptyDocumentError(f"Excel file '{path}' contains no data or empty sheets.")

        pages: list[ParsedPage] = []
        wb_meta: dict[str, Any] = {
            "sheet_names": list(dfs.keys()),
            "sheet_count": len(dfs),
        }

        for sheet_idx, (sheet_name, df) in enumerate(dfs.items(), start=1):
            if df.empty:
                continue

            headers = [str(c) for c in df.columns]
            rows_data = [headers]
            for row in df.itertuples(index=False, name=None):
                rows_data.append([str(v) if v is not None and not (isinstance(v, float) and pd.isna(v)) else "" for v in row])

            # Markdown representation
            md_lines = [" | ".join(headers)]
            md_lines.append(" | ".join(["---"] * len(headers)))
            for r in rows_data[1:]:
                md_lines.append(" | ".join(r))
            md_content = "\n".join(md_lines)

            dtypes_map = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
            missing_map = {str(col): int(count) for col, count in df.isna().sum().items()}

            meta: dict[str, Any] = {
                "sheet_name": sheet_name,
                "sheet_index": sheet_idx,
                "rows": len(df),
                "cols": len(df.columns),
                "headers": headers,
                "has_header": True,
                "row_cells": rows_data,
                "dtypes": dtypes_map,
                "missing_values": missing_map,
                "preserves_formulas": False,
                "preserves_merged_cells": False,
            }

            block = ParsedBlock(
                block_type=BlockType.TABLE,
                content=md_content,
                page_number=sheet_idx,
                metadata=meta,
            )

            pages.append(
                ParsedPage(
                    page_number=sheet_idx,
                    blocks=[block],
                )
            )

        if not pages:
            raise EmptyDocumentError(f"Excel file '{path}' contains no non-empty sheets.")

        return ParsedDocument(
            source=str(path),
            parser_name=self.name,
            parser_version=str(getattr(pd, "__version__", "unknown")),
            metadata=wb_meta,
            pages=pages,
            blocks=[],
            warnings=[],
        )
