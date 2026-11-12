"""Tree-sitter multi-language code parser: AST-driven syntax extraction.

Supports Python, JavaScript, TypeScript, Java, C, C++, and Go.
Extracts functions, classes, methods, interfaces, imports, line positions, and raw syntax text.
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
    ParserDependencyError,
    ParserError,
)
from parsers.contracts.validation import validate_source

logger = logging.getLogger(__name__)

EXTENSION_LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".go": "go",
}

FUNCTION_NODE_TYPES = {
    "function_definition",
    "function_declaration",
    "method_declaration",
    "method_definition",
}

CLASS_NODE_TYPES = {
    "class_definition",
    "class_declaration",
    "class_specifier",
    "type_declaration",
    "struct_specifier",
}

INTERFACE_NODE_TYPES = {
    "interface_declaration",
}

IMPORT_NODE_TYPES = {
    "import_statement",
    "import_declaration",
    "preproc_include",
    "package_clause",
    "package_declaration",
}


def _load_tree_sitter_language(lang_name: str):
    try:
        import tree_sitter
        from tree_sitter import Language, Parser
    except ImportError as exc:
        raise ParserDependencyError(
            "tree-sitter is not installed. Install it with:\n"
            "  pip install 'prome-rag-plugins[code]'\n"
            "or:\n"
            "  pip install tree-sitter"
        ) from exc

    try:
        if lang_name == "python":
            import tree_sitter_python as m
            raw_lang = m.language()
        elif lang_name == "javascript":
            import tree_sitter_javascript as m
            raw_lang = m.language()
        elif lang_name == "typescript":
            import tree_sitter_typescript as m
            raw_lang = m.language_typescript()
        elif lang_name == "java":
            import tree_sitter_java as m
            raw_lang = m.language()
        elif lang_name == "c":
            import tree_sitter_c as m
            raw_lang = m.language()
        elif lang_name == "cpp":
            import tree_sitter_cpp as m
            raw_lang = m.language()
        elif lang_name == "go":
            import tree_sitter_go as m
            raw_lang = m.language()
        else:
            raise ParserError(f"Unsupported language grammar '{lang_name}'")
        return Language(raw_lang), Parser, tree_sitter
    except ImportError as exc:
        raise ParserDependencyError(
            f"tree-sitter grammar for '{lang_name}' is not installed. Install it with:\n"
            f"  pip install tree-sitter-{lang_name}"
        ) from exc


def _extract_name(node: Any) -> str:
    name_child = node.child_by_field_name("name")
    if name_child:
        return name_child.text.decode("utf-8", errors="replace")
    for child in node.children:
        if child.type in ("identifier", "type_identifier", "field_identifier"):
            return child.text.decode("utf-8", errors="replace")
        if child.type in ("function_declarator", "declarator", "type_specifier", "type_spec"):
            res = _extract_name(child)
            if res != "anonymous":
                return res
    return "anonymous"


class TreeSitterParser:
    """Multi-language AST code parser using tree-sitter."""

    name = "tree_sitter"

    def __init__(self, language: str | None = None) -> None:
        self.language = language

    def parse(self, source: str | Path) -> ParsedDocument:
        path = validate_source(source)

        if path.stat().st_size == 0:
            raise EmptyDocumentError(f"Source file '{path}' is empty (0 bytes).")

        ext = path.suffix.lower()
        lang_name = self.language or EXTENSION_LANG_MAP.get(ext)
        if not lang_name:
            raise ParserError(f"Unsupported code language or file extension '{ext}' for file '{path}'.")

        lang_obj, ParserCls, ts_mod = _load_tree_sitter_language(lang_name)
        parser = ParserCls(lang_obj)

        try:
            source_bytes = path.read_bytes()
        except Exception as exc:
            raise CorruptDocumentError(f"Could not read code file '{path}': {exc}") from exc

        if not source_bytes.strip():
            raise EmptyDocumentError(f"Source file '{path}' contains only whitespace.")

        try:
            tree = parser.parse(source_bytes)
        except Exception as exc:
            raise CorruptDocumentError(f"Tree-sitter failed to parse '{path}': {exc}") from exc

        root = tree.root_node
        if root.has_error and len(root.children) == 0:
            raise CorruptDocumentError(f"Syntax error parsing code file '{path}'")

        blocks: list[ParsedBlock] = []
        counts = {
            "functions": 0,
            "classes": 0,
            "interfaces": 0,
            "imports": 0,
        }

        def visit(node: Any, parent_class: str | None = None) -> None:
            n_type = node.type
            start_row = node.start_point.row + 1
            end_row = node.end_point.row + 1
            start_col = node.start_point.column
            end_col = node.end_point.column
            node_text = node.text.decode("utf-8", errors="replace")

            if n_type in IMPORT_NODE_TYPES:
                counts["imports"] += 1
                blocks.append(
                    ParsedBlock(
                        block_type=BlockType.CODE,
                        content=node_text,
                        metadata={
                            "kind": "import",
                            "node_type": n_type,
                            "start_line": start_row,
                            "end_line": end_row,
                            "start_col": start_col,
                            "end_col": end_col,
                        },
                    )
                )

            elif n_type in CLASS_NODE_TYPES:
                counts["classes"] += 1
                cls_name = _extract_name(node)
                blocks.append(
                    ParsedBlock(
                        block_type=BlockType.DEFINITION,
                        content=node_text,
                        metadata={
                            "kind": "class",
                            "name": cls_name,
                            "node_type": n_type,
                            "start_line": start_row,
                            "end_line": end_row,
                            "start_col": start_col,
                            "end_col": end_col,
                        },
                    )
                )
                # Recurse inside class body with parent_class context
                for child in node.children:
                    visit(child, parent_class=cls_name)
                return

            elif n_type in INTERFACE_NODE_TYPES:
                counts["interfaces"] += 1
                if_name = _extract_name(node)
                blocks.append(
                    ParsedBlock(
                        block_type=BlockType.DEFINITION,
                        content=node_text,
                        metadata={
                            "kind": "interface",
                            "name": if_name,
                            "node_type": n_type,
                            "start_line": start_row,
                            "end_line": end_row,
                            "start_col": start_col,
                            "end_col": end_col,
                        },
                    )
                )

            elif n_type in FUNCTION_NODE_TYPES:
                fn_name = _extract_name(node)
                is_method = bool(parent_class or "method" in n_type)
                counts["functions"] += 1
                meta = {
                    "kind": "method" if is_method else "function",
                    "name": fn_name,
                    "node_type": n_type,
                    "start_line": start_row,
                    "end_line": end_row,
                    "start_col": start_col,
                    "end_col": end_col,
                }
                if parent_class:
                    meta["parent_class"] = parent_class

                blocks.append(
                    ParsedBlock(
                        block_type=BlockType.DEFINITION,
                        content=node_text,
                        metadata=meta,
                    )
                )

            # Recurse children
            for child in node.children:
                visit(child, parent_class=parent_class)

        visit(root)

        if not blocks:
            # Emit top-level module code
            blocks.append(
                ParsedBlock(
                    block_type=BlockType.CODE,
                    content=source_bytes.decode("utf-8", errors="replace"),
                    metadata={"kind": "source_body", "language": lang_name},
                )
            )

        metadata = {
            "language": lang_name,
            "tree_sitter_version": str(getattr(ts_mod, "__version__", "unknown")),
            "functions_count": counts["functions"],
            "classes_count": counts["classes"],
            "interfaces_count": counts["interfaces"],
            "imports_count": counts["imports"],
            "has_syntax_errors": root.has_error,
        }

        return ParsedDocument(
            source=str(path),
            parser_name=self.name,
            parser_version=str(getattr(ts_mod, "__version__", "unknown")),
            metadata=metadata,
            pages=[],
            blocks=blocks,
            warnings=["syntax_error_detected"] if root.has_error else [],
        )
