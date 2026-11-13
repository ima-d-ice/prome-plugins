"""Tests for the local upload storage backend."""

from __future__ import annotations

import pytest

from loaders.uploads.storage import StorageBackend
from tests.loaders.conftest import make_uploaded_file


@pytest.fixture()
def storage(tmp_path):
    return StorageBackend(tmp_path / "store", create_root=True)


def test_store_and_retrieve(storage, tmp_path):
    file = make_uploaded_file("stored.txt", b"stored content")
    location = storage.store(file)
    assert location.startswith(str(storage.storage_root))
    assert storage.retrieve(file.file_id) == b"stored content"


def test_storage_path_never_uses_original_filename(storage):
    file = make_uploaded_file("../../etc/evil.txt", b"data")
    location = storage.store(file)
    assert "evil.txt" not in location
    assert ".." not in location
    # Storage path is rooted inside the configured storage root
    assert location.startswith(str(storage.storage_root))


def test_store_is_atomic(storage, tmp_path):
    file = make_uploaded_file("atomic.txt", b"atomic content")
    storage.store(file)
    # No .tmp files left behind after successful store
    leftovers = list(storage.storage_root.rglob("*.tmp"))
    assert leftovers == []


def test_integrity_check_passes(storage):
    file = make_uploaded_file("integrity.txt", b"integrity check content")
    storage.store(file)
    assert storage.check_integrity(file.file_id) is True


def test_integrity_check_fails_after_tamper(storage):
    file = make_uploaded_file("tamper.txt", b"original content")
    location = storage.store(file)
    with open(location, "wb") as f:
        f.write(b"tampered content")
    assert storage.check_integrity(file.file_id) is False


def test_delete_file(storage):
    file = make_uploaded_file("delete-me.txt", b"delete me")
    storage.store(file)
    assert storage.delete(file.file_id) is True
    assert storage.retrieve(file.file_id) is None
    assert storage.delete(file.file_id) is False


def test_reindex_survives_reopen(tmp_path):
    file = make_uploaded_file("persist.txt", b"persistent content")
    storage1 = StorageBackend(tmp_path / "store")
    storage1.store(file)

    storage2 = StorageBackend(tmp_path / "store")
    assert storage2.retrieve(file.file_id) == b"persistent content"
    assert storage2.check_integrity(file.file_id) is True


def test_list_files(storage):
    files = [make_uploaded_file(f"f{i}.txt", f"content {i}", file_id=f"listfid-{i}") for i in range(3)]
    for f in files:
        storage.store(f)
    listed = storage.list_files()
    assert len(listed) >= 3
