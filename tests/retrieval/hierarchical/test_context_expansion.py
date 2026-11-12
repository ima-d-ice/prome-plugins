"""Tests of the context-expansion component."""

from __future__ import annotations

import pytest

from chunkers.contracts.models import BlockReference, Chunk
from retrieval.postprocessing.context_expansion import (
    ContextExpansionConfig,
    ContextExpander,
)


def child(chunk_id: str, parent_id: str, index: int, text: str) -> Chunk:
    """Build a child chunk for the tests."""
    return Chunk(
        id=chunk_id,
        doc_id="doc.md",
        text=text,
        index=index,
        count=3,
        length=len(text),
        length_unit="characters",
        block_refs=[BlockReference(block_index=0, page_number=1, block_type="text")],
        metadata={
            "parent_id": parent_id,
            "child_index": index,
            "heading_path": ["Section"],
            "source": "doc.md",
        },
    )


def parent(chunk_id: str, child_ids: list[str]) -> Chunk:
    """Build a parent chunk for the tests."""
    return Chunk(
        id=chunk_id,
        doc_id="doc.md",
        text="the whole parent section",
        index=0,
        count=1,
        length=len("the whole parent section"),
        length_unit="characters",
        block_refs=[BlockReference(block_index=0, page_number=1, block_type="text")],
        metadata={
            "level": "parent",
            "parent_id": chunk_id,
            "child_ids": child_ids,
            "child_count": len(child_ids),
            "heading_path": ["Section"],
            "source": "doc.md",
        },
    )


@pytest.fixture()
def hierarchy():
    """Return a parent with three children."""
    children = [
        child("c0", "p0", 0, "first child"),
        child("c1", "p0", 1, "second child"),
        child("c2", "p0", 2, "third child"),
    ]
    return parent("p0", [c.id for c in children]), children


def test_the_parent_context_is_attached(hierarchy):
    """The parent section is attached to the matched child."""
    parent_chunk, children = hierarchy
    expander = ContextExpander(
        parents={"p0": parent_chunk}, children={c.id: c for c in children}
    )

    expanded = expander.expand(children[1], config=ContextExpansionConfig())

    assert expanded.parent is not None
    assert expanded.parent.chunk_id == "p0"
    assert expanded.parent.role == "parent"
    assert expanded.parent.text == parent_chunk.text
    assert expanded.parent.doc_id == "doc.md"


def test_the_parent_context_can_be_disabled(hierarchy):
    """Without the option the parent is not attached."""
    parent_chunk, children = hierarchy
    expander = ContextExpander(
        parents={"p0": parent_chunk}, children={c.id: c for c in children}
    )

    expanded = expander.expand(
        children[1], config=ContextExpansionConfig(include_parent=False)
    )

    assert expanded.parent is None


def test_the_adjacent_expansion_attaches_the_neighbours(hierarchy):
    """The neighbouring children of the same parent are attached."""
    parent_chunk, children = hierarchy
    expander = ContextExpander(
        parents={"p0": parent_chunk}, children={c.id: c for c in children}
    )

    expanded = expander.expand(
        children[1], config=ContextExpansionConfig(include_adjacent=True, max_adjacent=2)
    )

    assert {item.chunk_id for item in expanded.adjacent} == {"c0", "c2"}
    assert all(item.role == "adjacent" for item in expanded.adjacent)


def test_the_adjacent_expansion_budget_is_a_total(hierarchy):
    """A budget of one yields the single closest neighbour."""
    parent_chunk, children = hierarchy
    expander = ContextExpander(
        parents={"p0": parent_chunk}, children={c.id: c for c in children}
    )

    expanded = expander.expand(
        children[1], config=ContextExpansionConfig(include_adjacent=True, max_adjacent=1)
    )

    assert len(expanded.adjacent) == 1
    # The tie between the two neighbours at distance one is broken by the
    # document order, so the result is deterministic.
    assert expanded.adjacent[0].chunk_id == "c0"


def test_the_adjacent_expansion_respects_the_limit(hierarchy):
    """At most ``max_adjacent`` neighbours are attached."""
    parent_chunk, children = hierarchy
    expander = ContextExpander(
        parents={"p0": parent_chunk}, children={c.id: c for c in children}
    )

    expanded = expander.expand(
        children[1], config=ContextExpansionConfig(include_adjacent=True, max_adjacent=1)
    )

    assert len(expanded.adjacent) == 1


def test_the_adjacent_expansion_never_repeats_the_content(hierarchy):
    """A neighbour with the same text is not attached twice."""
    parent_chunk, children = hierarchy
    duplicated = child("c2", "p0", 2, "second child")
    expander = ContextExpander(
        parents={"p0": parent_chunk},
        children={"c0": children[0], "c1": children[1], "c2": duplicated},
    )

    expanded = expander.expand(
        children[1], config=ContextExpansionConfig(include_adjacent=True, max_adjacent=2)
    )

    texts = [item.text for item in expanded.adjacent]
    assert len(texts) == len(set(texts))


def test_the_adjacent_expansion_stays_inside_the_parent(hierarchy):
    """Only the children of the matched parent are attached."""
    parent_chunk, children = hierarchy
    other = child("x0", "p1", 0, "a child of another parent")
    expander = ContextExpander(
        parents={"p0": parent_chunk},
        children={c.id: c for c in children} | {"x0": other},
    )

    expanded = expander.expand(
        children[0], config=ContextExpansionConfig(include_adjacent=True, max_adjacent=5)
    )

    assert all(item.chunk_id != "x0" for item in expanded.adjacent)


def test_the_adjacent_expansion_stays_inside_the_document(hierarchy):
    """A chunk of another document is never attached."""
    parent_chunk, children = hierarchy
    foreign = child("f0", "p0", 0, "a chunk of another document")
    foreign.doc_id = "other.md"
    expander = ContextExpander(
        parents={"p0": parent_chunk},
        children={c.id: c for c in children} | {"f0": foreign},
    )

    expanded = expander.expand(
        children[1], config=ContextExpansionConfig(include_adjacent=True, max_adjacent=5)
    )

    assert all(item.doc_id == "doc.md" for item in expanded.adjacent)
    assert all(item.chunk_id != "f0" for item in expanded.adjacent)


def test_the_adjacent_expansion_never_returns_the_evidence(hierarchy):
    """The matched child is not attached as its own neighbour."""
    parent_chunk, children = hierarchy
    expander = ContextExpander(
        parents={"p0": parent_chunk}, children={c.id: c for c in children}
    )

    expanded = expander.expand(
        children[1], config=ContextExpansionConfig(include_adjacent=True, max_adjacent=5)
    )

    assert all(item.chunk_id != "c1" for item in expanded.adjacent)


def test_a_missing_parent_is_reported(hierarchy):
    """A missing parent is reported instead of being faked."""
    _, children = hierarchy
    expander = ContextExpander(parents={}, children={c.id: c for c in children})

    expanded = expander.expand(children[0], config=ContextExpansionConfig())

    assert expanded.parent is None
    assert expanded.notes


def test_the_expansion_never_rewrites_the_source_text(hierarchy):
    """The attached context is the original text."""
    parent_chunk, children = hierarchy
    expander = ContextExpander(
        parents={"p0": parent_chunk}, children={c.id: c for c in children}
    )

    expanded = expander.expand(
        children[1], config=ContextExpansionConfig(include_adjacent=True, max_adjacent=2)
    )

    assert expanded.parent.text == parent_chunk.text
    for item in expanded.adjacent:
        original = next(c for c in children if c.id == item.chunk_id)
        assert item.text == original.text


def test_the_context_keeps_the_source_references(hierarchy):
    """The provenance of the context is preserved."""
    parent_chunk, children = hierarchy
    expander = ContextExpander(
        parents={"p0": parent_chunk}, children={c.id: c for c in children}
    )

    expanded = expander.expand(
        children[1], config=ContextExpansionConfig(include_adjacent=True, max_adjacent=1)
    )

    assert expanded.parent.provenance is not None
    assert expanded.parent.provenance.source == "doc.md"
    for item in expanded.adjacent:
        assert item.provenance is not None
        assert item.provenance.pages == (1,)


def test_an_invalid_limit_is_rejected():
    """A negative limit is a configuration error."""
    with pytest.raises(ValueError):
        ContextExpansionConfig(max_adjacent=-1)
