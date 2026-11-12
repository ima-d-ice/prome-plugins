"""Generate deterministic synthetic JSON and NDJSON fixtures."""

from __future__ import annotations

import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def make_nested_object() -> None:
    data = {
        "company": "Prome Technologies",
        "headquarters": {
            "city": "San Francisco",
            "state": "CA",
            "address": {
                "street": "100 Market St",
                "suite": 400
            }
        },
        "departments": [
            {
                "name": "Engineering",
                "lead": "Alice",
                "active": True,
                "headcount": 42
            },
            {
                "name": "Product",
                "lead": "Bob",
                "active": True,
                "headcount": 15
            }
        ]
    }
    (FIXTURES / "nested_object.json").write_text(json.dumps(data, indent=2))


def make_primitives() -> None:
    data = {
        "text": "Hello world",
        "integer": 1024,
        "float": 3.14159,
        "boolean_true": True,
        "boolean_false": False,
        "nullable": None,
        "empty_list": [],
        "unicode": "Café — こんにちは — Привет"
    }
    (FIXTURES / "primitives.json").write_text(json.dumps(data, indent=2))


def make_stream_ndjson() -> None:
    lines = [
        {"event": "login", "user_id": 101, "timestamp": "2026-11-12T00:01:00Z"},
        {"event": "query", "user_id": 101, "query": "RAG parsers", "latency_ms": 12.5},
        {"event": "logout", "user_id": 101, "timestamp": "2026-11-12T00:05:00Z"},
    ]
    content = "\n".join(json.dumps(obj) for obj in lines)
    (FIXTURES / "stream.ndjson").write_text(content)


def make_empty_json() -> None:
    (FIXTURES / "empty.json").write_text("")


def make_malformed_json() -> None:
    (FIXTURES / "malformed.json").write_text('{"unclosed": "string')


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    make_nested_object()
    make_primitives()
    make_stream_ndjson()
    make_empty_json()
    make_malformed_json()
    print(f"Generated JSON fixtures in {FIXTURES}")


if __name__ == "__main__":
    main()
