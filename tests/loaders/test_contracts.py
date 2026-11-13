"""Tests for shared loader contracts."""

from __future__ import annotations

from loaders.contracts.models import (
    ContentType,
    LoaderResult,
    SourceProvenance,
    SourceReference,
    SourceType,
)


def make_ref(source_id="s1"):
    return SourceReference(
        source_id=source_id,
        source_type=SourceType.LOCAL_FILE,
        original_uri_or_path="/tmp/example.txt",
        filename="example.txt",
    )


def test_source_reference_is_stable_and_unique():
    ref = make_ref()
    assert ref.source_id == "s1"
    assert ref.source_type is SourceType.LOCAL_FILE
    assert ref.filename == "example.txt"


def test_loader_result_add_failure_and_skip():
    result = LoaderResult()
    assert result.success_count == 0
    failures_before = result.failure_count
    result.add_failure("something failed")
    assert result.failure_count == failures_before + 1
    skips_before = result.skip_count
    result.add_skip("skipped", reason="unsupported")
    assert result.skip_count == skips_before + 1
    # add_* returns self for chaining
    assert result.add_failure("boom") is result


def test_loader_result_counts_resources():
    from loaders.contracts.models import LoadedResource

    result = LoaderResult()
    result.resources.append(
        LoadedResource(
            reference=make_ref(),
            content=b"hello",
            content_type=ContentType.TEXT,
        )
    )
    assert result.success_count == 1
    assert result.failure_count == 0


def test_source_provenance_defaults():
    prov = SourceProvenance()
    assert prov.fetched_from is None
    assert prov.content_hash is None


def test_content_type_text_properties():
    from loaders.contracts.models import LoadedResource

    resource = LoadedResource(
        reference=make_ref(),
        content=b"hello",
        content_type=ContentType.TEXT,
    )
    assert resource.is_text
    assert not resource.is_binary


def test_source_type_values():
    assert SourceType.LOCAL_FILE.value == "local_file"
    assert SourceType.URL.value == "url"
    assert SourceType.CRAWLED.value == "crawled"
    assert SourceType.UPLOAD.value == "upload"
