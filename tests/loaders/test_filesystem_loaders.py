"""Tests for local filesystem loaders: files, directories, filters."""

from __future__ import annotations

import pytest

from loaders.contracts.exceptions import PathTraversalError
from loaders.filesystem import (
    DirectoryLoader,
    ExclusionFilter,
    ExtensionFilter,
    FileLoader,
    HiddenFilePolicy,
    SymlinkPolicy,
)


def test_file_loader_loads_file(local_files):
    loader = FileLoader(permitted_roots=[local_files])
    result = loader.load(local_files / "alpha.txt")
    assert result.success_count == 1
    resource = result.resources[0]
    assert resource.filename == "alpha.txt"
    assert resource.content == b"alpha content"
    assert resource.content_hash


def test_file_loader_rejects_path_outside_permitted_root(local_files):
    loader = FileLoader(permitted_roots=[local_files])
    with pytest.raises(PathTraversalError):
        loader.load("/etc/hosts")


def test_file_loader_missing_file_returns_failure(local_files):
    loader = FileLoader(permitted_roots=[local_files])
    result = loader.load(local_files / "does-not-exist.txt")
    assert result.failure_count == 1
    assert result.success_count == 0


def test_file_loader_max_size(local_files):
    loader = FileLoader(permitted_roots=[local_files], max_file_size=5)
    result = loader.load(local_files / "alpha.txt")
    assert result.failure_count == 1


def test_directory_loader_recursive(local_files):
    loader = DirectoryLoader(permitted_roots=[local_files], recursive=True)
    result = loader.load(local_files)
    names = {r.filename for r in result.resources}
    assert {"alpha.txt", "beta.md", "gamma.txt"} <= names


def test_directory_loader_non_recursive(local_files):
    loader = DirectoryLoader(permitted_roots=[local_files], recursive=False)
    result = loader.load(local_files)
    names = {r.filename for r in result.resources}
    assert "gamma.txt" not in names
    assert {"alpha.txt", "beta.md"} <= names


def test_directory_loader_excludes_hidden_by_default(local_files):
    loader = DirectoryLoader(permitted_roots=[local_files])
    result = loader.load(local_files)
    names = {r.filename for r in result.resources}
    assert ".hidden" not in names
    assert ".env" not in names


def test_directory_loader_excludes_env_even_with_hidden_include(local_files):
    loader = DirectoryLoader(
        permitted_roots=[local_files],
        hidden_file_policy=HiddenFilePolicy.INCLUDE,
    )
    result = loader.load(local_files)
    names = {r.filename for r in result.resources}
    assert ".env" not in names
    assert ".hidden" in names


def test_directory_loader_deterministic_order(local_files):
    loader = DirectoryLoader(permitted_roots=[local_files])
    result1 = loader.load(local_files)
    result2 = loader.load(local_files)
    assert [r.filename for r in result1.resources] == [r.filename for r in result2.resources]


def test_directory_loader_max_files(local_files):
    loader = DirectoryLoader(permitted_roots=[local_files], max_files=1)
    result = loader.load(local_files)
    assert result.success_count <= 1
    assert result.failure_count >= 1 or result.success_count == 1


def test_directory_loader_extension_allowlist(local_files):
    ext = ExtensionFilter({".md"})
    loader = DirectoryLoader(permitted_roots=[local_files], extension_filter=ext)
    result = loader.load(local_files)
    assert {r.filename for r in result.resources} == {"beta.md"}


def test_directory_loader_exclusion_patterns(local_files):
    exclusion = ExclusionFilter(["alpha.txt"])
    loader = DirectoryLoader(permitted_roots=[local_files], exclusion_filter=exclusion)
    result = loader.load(local_files)
    names = {r.filename for r in result.resources}
    assert "alpha.txt" not in names


def test_directory_loader_symlink_rejected(local_files, tmp_path):
    link = local_files / "link.txt"
    link.symlink_to(local_files / "alpha.txt")
    loader = DirectoryLoader(permitted_roots=[local_files], symlink_policy=SymlinkPolicy.REJECT)
    result = loader.load(local_files)
    assert "link.txt" not in {r.filename for r in result.resources}


def test_directory_loader_symlink_follow_preserves_root(local_files, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("outside content")
    link = local_files / "escape.txt"
    link.symlink_to(outside)
    loader = DirectoryLoader(
        permitted_roots=[local_files],
        symlink_policy=SymlinkPolicy.FOLLOW,
    )
    result = loader.load(local_files)
    # Symlink pointing outside permitted root is blocked
    assert "escape.txt" not in {r.filename for r in result.resources}


def test_directory_loader_does_not_scan_outside_root(local_files, tmp_path):
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "outside.txt").write_text("outside")
    loader = DirectoryLoader(permitted_roots=[local_files])
    result = loader.load(local_files)
    assert "outside.txt" not in {r.filename for r in result.resources}


def test_loader_never_modifies_source_files(local_files):
    before = (local_files / "alpha.txt").read_bytes()
    loader = FileLoader(permitted_roots=[local_files])
    loader.load(local_files / "alpha.txt")
    dloader = DirectoryLoader(permitted_roots=[local_files])
    dloader.load(local_files)
    assert (local_files / "alpha.txt").read_bytes() == before
