"""Context-aware prompt injection scanner for retrieved RAG documents.

Scans retrieved chunks, parent sections, and child chunks for prompt
injection/jailbreak attempts. Respects the 512-token model limit via
segmentation and aggregates results conservatively.

Supports scanning of all eligible segments from retrieved evidence,
recording the result of every segment and producing a conservative
overall classification.

Do not depend on the project's retrieval contracts: this scanner uses
dict-based adapters for compatibility with any retrieval backend.
"""

from __future__ import annotations

from typing import Any

from guardrails.prompt_injection.scanner import PromptInjectionScanner
from guardrails.contracts.models import GuardrailStage
from guardrails.contracts.exceptions import GuardrailError


class ContextScanner:
    """Scan retrieved RAG context for prompt injection.

    Scans user-supplied untrusted context including:
    - Retrieved documents (full text or chunks)
    - Retrieved chunks
    - Parent sections
    - Child chunks
    - Other caller-supplied untrusted context

    Each segment is scanned independently and the results are aggregated
    conservatively: ANY malicious segment -> overall MALICIOUS.

    The scanner respects the 512-token Prompt Guard model context window
    via token-aware segmentation.

    Do not depend on the project's retrieval contracts: this scanner uses
    dict-based adapters for compatibility with any retrieval backend.
    """

    def __init__(self, scanner: PromptInjectionScanner | None = None) -> None:
        """Initialize the context scanner.

        Args:
            scanner: optional PromptInjectionScanner instance. When None,
                a default scanner is created.
        """

        self._scanner = scanner or PromptInjectionScanner()

    def scan_retrieved_item(
        self,
        item: dict[str, Any],
        *,
        language_code: str | None = None,
    ) -> dict[str, Any]:
        """Scan a retrieved item's text for prompt injection.

        A retrieved item dict should have at least a "text" key, and
        optionally "chunk_id", "doc_id", and "context" keys.

        Args:
            item: the retrieved item dict to scan.
            language_code: optional ISO-639-1 language code.

        Returns:
            classification result dict with diagnostics including the chunk
            ID and source reference.
        """

        # Scan the primary chunk text
        text = item.get("text", "")
        primary_result = self._scanner.scan_target(
            text=text,
            language_code=language_code,
        )

        # Build diagnostics with source information
        diagnostics: dict[str, Any] = {
            "chunk_id": item.get("chunk_id"),
            "doc_id": item.get("doc_id"),
            "scan_target": "retrieved_chunk",
        }
        primary_result["diagnostics"] = diagnostics

        # Also scan attached context if present
        context = item.get("context")
        if context:
            context_results: list[dict[str, Any]] = []
            for ctx_item in context:
                ctx_text = ctx_item.get("text", "") if isinstance(ctx_item, dict) else str(ctx_item)
                ctx_result = self._scanner.scan_target(
                    text=ctx_text,
                    language_code=language_code,
                )
                # Add context item metadata to diagnostics
                ctx_diagnostics = ctx_result.get("diagnostics", {})
                if isinstance(ctx_item, dict):
                    ctx_diagnostics["chunk_id"] = ctx_item.get("chunk_id")
                    ctx_diagnostics["role"] = ctx_item.get("role", "")
                context_results.append(ctx_result)

            # If any context segment is malicious, update the overall result
            if context_results:
                any_malicious = any(
                    r.get("classification") == "MALICIOUS"
                    for r in context_results
                )
                if any_malicious and primary_result.get("classification") != "MALICIOUS":
                    primary_result["classification"] = "MALICIOUS"
                    # Aggregate malicious segments from context
                    for cr in context_results:
                        if cr.get("malicious_segments"):
                            primary_result.setdefault("malicious_segments", []).extend(
                                cr["malicious_segments"]
                            )

        return primary_result

    def scan_chunk_text(
        self,
        chunk_text: str,
        chunk_id: str | None = None,
        doc_id: str | None = None,
        *,
        language_code: str | None = None,
    ) -> dict[str, Any]:
        """Scan a chunk of text for prompt injection.

        Args:
            chunk_text: the chunk text to scan.
            chunk_id: optional chunk identifier for diagnostics.
            doc_id: optional document identifier for diagnostics.
            language_code: optional ISO-639-1 language code.

        Returns:
            classification result dict with chunk/doc diagnostics.
        """

        result = self._scanner.scan_target(
            text=chunk_text,
            language_code=language_code,
        )

        # Add chunk and doc diagnostics
        diagnostics: dict[str, Any] = {}
        if chunk_id:
            diagnostics["chunk_id"] = chunk_id
        if doc_id:
            diagnostics["doc_id"] = doc_id
        result["diagnostics"] = diagnostics

        return result

    def scan_parent_section(
        self,
        section_text: str,
        section_id: str | None = None,
        doc_id: str | None = None,
        *,
        language_code: str | None = None,
    ) -> dict[str, Any]:
        """Scan a parent section text for prompt injection.

        Parent sections are enclosing sections that contain child chunks.
        Scanning them is important because malicious instructions can be
        embedded in section headings or provenance text.

        Args:
            section_text: the parent section text to scan.
            section_id: optional section identifier for diagnostics.
            doc_id: optional document identifier for diagnostics.
            language_code: optional ISO-639-1 language code.

        Returns:
            classification result section/doc diagnostics.
        """

        result = self._scanner.scan_target(
            text=section_text,
            language_code=language_code,
        )

        # Add section and doc diagnostics
        diagnostics: dict[str, Any] = {}
        if section_id:
            diagnostics["section_id"] = section_id
        if doc_id:
            diagnostics["doc_id"] = doc_id
        result["diagnostics"] = diagnostics

        return result

    def scan_child_chunk(
        self,
        chunk_text: str,
        chunk_id: str | None = None,
        doc_id: str | None = None,
        *,
        language_code: str | None = None,
    ) -> dict[str, Any]:
        """Scan a child chunk text for prompt injection.

        Child chunks are the individual evidence chunks retrieved from the
        vector store. Scanning them ensures that malicious instructions
        hidden in child chunks are detected.

        Args:
            chunk_text: the child chunk text to scan.
            chunk_id: optional chunk identifier for diagnostics.
            doc_id: optional document identifier for diagnostics.
            language_code: optional ISO-639-1 language code.

        Returns:
            classification result with chunk/doc diagnostics.
        """

        result = self._scanner.scan_target(
            text=chunk_text,
            language_code=language_code,
        )

        # Add chunk and doc diagnostics
        diagnostics: dict[str, Any] = {}
        if chunk_id:
            diagnostics["chunk_id"] = chunk_id
        if doc_id:
            diagnostics["doc_id"] = doc_id
        result["diagnostics"] = diagnostics

        return result

    def scan_all_chunks(
        self,
        chunks: list[dict[str, Any]],
        *,
        language_code: str | None = None,
    ) -> dict[str, Any]:
        """Scan multiple chunks for prompt injection, aggregating results.

        Scans every chunk and aggregates results conservatively: if ANY
        chunk is classified as MALICIOUS, the overall result is MALICIOUS.

        Args:
            chunks: list of dicts with at least "text" key, optionally
                "chunk_id" and "doc_id".
            language_code: optional ISO-639-1 language code.

        Returns:
            overall classification result with per-chunk diagnostics and
            consolidated malicious segment references.
        """

        overall_result: dict[str, Any] = {
            "classification": "BENIGN",
            "malicious_segments": [],
            "diagnostics": {
                "num_chunks_scanned": 0,
                "chunks_malicious": 0,
            },
        }

        any_malicious_overall = False

        for chunk_info in chunks:
            chunk_text = chunk_info.get("text", "")
            chunk_id_val = chunk_info.get("chunk_id")
            doc_id_val = chunk_info.get("doc_id")

            chunk_result = self.scan_chunk_text(
                chunk_text=chunk_text,
                chunk_id=chunk_id_val,
                doc_id=doc_id_val,
                language_code=language_code,
            )

            # Aggregate classification
            if chunk_result.get("classification") == "MALICIOUS":
                any_malicious_overall = True
                overall_result["diagnostics"]["chunks_malicious"] = (
                    overall_result["diagnostics"].get("chunks_malicious", 0) + 1
                )

            # Aggregate malicious segments
            malicious_segments = chunk_result.get("malicious_segments", [])
            if malicious_segments:
                overall_result.setdefault("malicious_segments", []).extend(malicious_segments)

            # Update chunk count
            overall_result["diagnostics"][
                "num_chunks_scanned"
            ] = overall_result["diagnostics"].get("num_chunks_scanned", 0) + 1

        # Set overall classification
        if any_malicious_overall:
            overall_result["classification"] = "MALICIOUS"

        return overall_result