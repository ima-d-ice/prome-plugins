"""Python AST parser: syntax-aware Python repository and source-file extraction."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

from parsers.contracts import (
    BlockType,
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedBlock,
    ParsedDocument,
)
from parsers.contracts.validation import validate_source

logger = logging.getLogger(__name__)


def _get_decorator_name(dec: ast.AST) -> str:
    if isinstance(dec, ast.Name):
        return dec.id
    if isinstance(dec, ast.Attribute):
        return f"{_get_decorator_name(dec.value)}.{dec.attr}"
    if isinstance(dec, ast.Call):
        return _get_decorator_name(dec.func)
    return ast.unparse(dec) if hasattr(ast, "unparse") else "decorator"


def _format_arg(arg: ast.arg) -> str:
    res = arg.arg
    if arg.annotation:
        res += f": {ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else '...'}"
    return res


def _format_signature(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = [_format_arg(a) for a in func_node.args.args]
    if func_node.args.vararg:
        args.append(f"*{_format_arg(func_node.args.vararg)}")
    for a in func_node.args.kwonlyargs:
        args.append(_format_arg(a))
    if func_node.args.kwarg:
        args.append(f"**{_format_arg(func_node.args.kwarg)}")

    sig = f"{func_node.name}({', '.join(args)})"
    if func_node.returns:
        ret = ast.unparse(func_node.returns) if hasattr(ast, "unparse") else "..."
        sig += f" -> {ret}"
    return sig


class PythonAstParser:
    """Extract code definitions, docstrings, classes, and imports using Python's AST."""

    name = "python_ast"

    def parse(self, source: str | Path) -> ParsedDocument:
        path = validate_source(source)

        if path.stat().st_size == 0:
            raise EmptyDocumentError(f"Python source file '{path}' is empty (0 bytes).")

        try:
            source_text = path.read_text(encoding="utf-8")
        except Exception as exc:
            raise CorruptDocumentError(f"Could not read source file '{path}': {exc}") from exc

        if not source_text.strip():
            raise EmptyDocumentError(f"Python source file '{path}' contains only whitespace.")

        try:
            tree = ast.parse(source_text, filename=str(path))
        except SyntaxError as exc:
            raise CorruptDocumentError(f"Syntax error parsing '{path}': {exc.msg} at line {exc.lineno}") from exc
        except Exception as exc:
            raise CorruptDocumentError(f"Could not parse '{path}' as Python source: {exc}") from exc

        source_lines = source_text.splitlines()
        blocks: list[ParsedBlock] = []

        def get_slice(start: int, end: int) -> str:
            return "\n".join(source_lines[start - 1 : end])

        # 1. Module docstring
        mod_doc = ast.get_docstring(tree)
        if mod_doc:
            blocks.append(
                ParsedBlock(
                    block_type=BlockType.TEXT,
                    content=mod_doc,
                    metadata={"kind": "module_docstring"},
                )
            )

        classes_count = 0
        functions_count = 0
        methods_count = 0
        imports_count = 0

        # Traverse body
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports_count += 1
                start = getattr(node, "lineno", 1)
                end = getattr(node, "end_lineno", start)
                code_text = get_slice(start, end)
                meta: dict[str, Any] = {
                    "kind": "import",
                    "start_line": start,
                    "end_line": end,
                }
                if isinstance(node, ast.Import):
                    meta["names"] = [alias.name for alias in node.names]
                else:
                    meta["module"] = node.module or ""
                    meta["names"] = [alias.name for alias in node.names]

                blocks.append(
                    ParsedBlock(
                        block_type=BlockType.CODE,
                        content=code_text,
                        metadata=meta,
                    )
                )

            elif isinstance(node, ast.ClassDef):
                classes_count += 1
                start = node.lineno
                end = getattr(node, "end_lineno", start)
                cls_doc = ast.get_docstring(node)
                decorators = [_get_decorator_name(d) for d in node.decorator_list]
                bases = [ast.unparse(b) if hasattr(ast, "unparse") else str(b) for b in node.bases]

                blocks.append(
                    ParsedBlock(
                        block_type=BlockType.DEFINITION,
                        content=get_slice(start, end),
                        metadata={
                            "kind": "class",
                            "name": node.name,
                            "docstring": cls_doc,
                            "decorators": decorators,
                            "bases": bases,
                            "start_line": start,
                            "end_line": end,
                        },
                    )
                )

                # Class methods
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods_count += 1
                        m_start = sub.lineno
                        m_end = getattr(sub, "end_lineno", m_start)
                        m_doc = ast.get_docstring(sub)
                        m_decorators = [_get_decorator_name(d) for d in sub.decorator_list]
                        sig = _format_signature(sub)

                        blocks.append(
                            ParsedBlock(
                                block_type=BlockType.DEFINITION,
                                content=get_slice(m_start, m_end),
                                metadata={
                                    "kind": "method",
                                    "name": sub.name,
                                    "signature": sig,
                                    "parent_class": node.name,
                                    "docstring": m_doc,
                                    "decorators": m_decorators,
                                    "is_async": isinstance(sub, ast.AsyncFunctionDef),
                                    "start_line": m_start,
                                    "end_line": m_end,
                                },
                            )
                        )

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions_count += 1
                start = node.lineno
                end = getattr(node, "end_lineno", start)
                f_doc = ast.get_docstring(node)
                decorators = [_get_decorator_name(d) for d in node.decorator_list]
                sig = _format_signature(node)

                blocks.append(
                    ParsedBlock(
                        block_type=BlockType.DEFINITION,
                        content=get_slice(start, end),
                        metadata={
                            "kind": "function",
                            "name": node.name,
                            "signature": sig,
                            "docstring": f_doc,
                            "decorators": decorators,
                            "is_async": isinstance(node, ast.AsyncFunctionDef),
                            "start_line": start,
                            "end_line": end,
                        },
                    )
                )

        if not blocks:
            # If there are statements like expressions or assignments
            blocks.append(
                ParsedBlock(
                    block_type=BlockType.CODE,
                    content=source_text.strip(),
                    metadata={"kind": "module_body", "start_line": 1, "end_line": len(source_lines)},
                )
            )

        metadata = {
            "module_docstring": mod_doc,
            "classes_count": classes_count,
            "functions_count": functions_count,
            "methods_count": methods_count,
            "imports_count": imports_count,
            "line_count": len(source_lines),
        }

        return ParsedDocument(
            source=str(path),
            parser_name=self.name,
            parser_version="stdlib",
            metadata=metadata,
            pages=[],
            blocks=blocks,
            warnings=[],
        )
