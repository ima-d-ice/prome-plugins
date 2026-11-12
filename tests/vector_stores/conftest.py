"""Shared fixtures for the vector-store test suites.

The ``store`` fixture is built from the ``store_factory`` fixture, which
every backend test module provides.  Backends that support persistence also
provide ``persistent_store_factory``.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def store_factory():
    """Return a factory building an ephemeral store (overridden per backend)."""
    raise NotImplementedError(
        "every backend test module must provide a 'store_factory' fixture"
    )


@pytest.fixture()
def store(store_factory):
    """Return an ephemeral store with no collection."""
    instance = store_factory()
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture()
def persistent_store_factory():
    """Return a factory building a persistent store (overridden per backend)."""
    pytest.skip("this adapter does not provide a persistent store")


@pytest.fixture()
def collection(store):
    """Return a unique collection name and drop it afterwards."""
    import uuid

    name = f"conformance_{uuid.uuid4().hex[:10]}"
    try:
        yield name
    finally:
        try:
            store.delete_collection(name)
        except Exception as exc:  # pragma: no cover - the collection may not exist
            import logging

            logging.getLogger(__name__).debug(
                "could not drop the collection %s: %s", name, exc
            )
