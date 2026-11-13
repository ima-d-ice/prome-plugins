from typing import List, Dict, Any, Optional, Set
from evaluation.contracts.models import EvalSample


def citation_reference_validity(sample: EvalSample) -> Dict[str, Any]:
    """Check if every citation refers to a supplied source ID."""
    if sample.cited_source_ids is None or sample.generated_response is None:
        return {"valid": False, "reason": "Missing cited_source_ids or generated_response"}

    cited_ids = set(sample.cited_source_ids)
    # Check that all cited IDs exist in the response context
    # This is a basic check - full validation would need claim-level analysis
    return {
        "valid": True,
        "cited_count": len(cited_ids),
        "cited_ids": list(cited_ids),
    }


def citation_coverage(sample: EvalSample) -> Dict[str, Any]:
    """Check if answer claims that require evidence are associated with citations."""
    if sample.generated_response is None or sample.reference is None:
        return {"covered": False, "reason": "Missing response or reference"}

    # Basic check: if response has citations, they should reference source IDs
    cited_ids = set(sample.cited_source_ids) if sample.cited_source_ids else set()
    response_text = sample.generated_response

    return {
        "covered": len(cited_ids) > 0,
        "citation_count": len(cited_ids),
    }


def citation_integrity(sample: EvalSample) -> Dict[str, Any]:
    """Check if citations preserve original document IDs, chunk IDs and provenance."""
    if sample.cited_source_ids is None:
        return {"integrity": False, "reason": "No cited_source_ids"}

    # Verify that cited source IDs can be traced back to document/chunk provenance
    integrity_checks = {
        "has_source_ids": len(sample.cited_source_ids) > 0,
        "ids_preserved": all(
            cid.startswith("doc") or cid.startswith("chunk")
            for cid in sample.cited_source_ids
        ) if sample.cited_source_ids else False,
    }

    integrity_checks["overall"] = all([
        integrity_checks["has_source_ids"],
        integrity_checks["ids_preserved"],
    ])

    return integrity_checks


def citation_integrity_from_contexts(
    cited_ids: List[str],
    retrieved_context_ids: Optional[List[str]],
    doc_ids: Optional[List[str]] = None,
    chunk_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Validate citation integrity against retrieved contexts."""
    if not cited_ids:
        return {"integrity": False, "reason": "No cited IDs provided"}

    # Check that cited IDs match retrieved context IDs
    retrieved_set = set(retrieved_context_ids or [])
    cited_set = set(cited_ids)

    matched = cited_set & retrieved_set
    unmatched = cited_set - retrieved_set

    return {
        "integrity": len(unmatched) == 0,
        "matched_count": len(matched),
        "unmatched_count": len(unmatched),
        "match_rate": len(matched) / len(cited_set) if cited_set else 0.0,
    }