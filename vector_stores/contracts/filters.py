"""Portable metadata filter language shared by all vector-store backends.

The filter AST is intentionally tiny: field conditions (equality, inequality,
membership, ordering, existence) combined with boolean operators.  Every
backend maps this AST onto its own query language; a backend that cannot
express an operation reports it in its capability descriptor and raises
:class:`~vector_stores.contracts.exceptions.UnsupportedFilterOperationError`
instead of silently returning a different result set.

Only scalar payload values are supported (string, int, float, bool, None).
Nested objects and arrays are out of scope for this milestone.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any

from vector_stores.contracts.exceptions import VectorStoreConfigError


class FilterOp(str, Enum):
    """Operations supported by the portable filter language."""

    EQ = "eq"
    NE = "ne"
    IN = "in"
    NOT_IN = "not-in"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    EXISTS = "exists"
    AND = "and"
    OR = "or"
    NOT = "not"


#: Operations that apply to a single payload field.
FIELD_OPS: frozenset[FilterOp] = frozenset(
    {
        FilterOp.EQ,
        FilterOp.NE,
        FilterOp.IN,
        FilterOp.NOT_IN,
        FilterOp.GT,
        FilterOp.GTE,
        FilterOp.LT,
        FilterOp.LTE,
        FilterOp.EXISTS,
    }
)

#: Operations that combine other filters.
LOGICAL_OPS: frozenset[FilterOp] = frozenset({FilterOp.AND, FilterOp.OR, FilterOp.NOT})

#: Operations that compare a field against an ordered value.
ORDERED_OPS: frozenset[FilterOp] = frozenset(
    {FilterOp.GT, FilterOp.GTE, FilterOp.LT, FilterOp.LTE}
)


@dataclass(frozen=True)
class MetadataFilter:
    """One node of the portable filter tree.

    A node is either a *field condition* (``op`` in :data:`FIELD_OPS`, with
    ``field`` set) or a *logical condition* (``op`` in :data:`LOGICAL_OPS`,
    with ``children`` set).
    """

    op: FilterOp
    field: str | None = None
    value: Any = None
    children: tuple[MetadataFilter, ...] = dataclasses.field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the filter node."""
        data: dict[str, Any] = {"op": self.op.value}
        if self.field is not None:
            data["field"] = self.field
        if self.op is FilterOp.EXISTS:
            data["value"] = bool(self.value)
        elif self.op not in LOGICAL_OPS and self.value is not None:
            data["value"] = self.value
        if self.children:
            data["children"] = [child.as_dict() for child in self.children]
        return data


def _scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _as_values(field: str, values: Any) -> tuple[Any, ...]:
    """Return ``values`` as a tuple, rejecting non-sequences."""
    if isinstance(values, (str, bytes)) or not isinstance(
        values, (list, tuple, set, frozenset)
    ):
        raise VectorStoreConfigError(
            f"IN/NOT IN filter on {field!r} requires a sequence of values, "
            f"got {type(values).__name__}"
        )
    return tuple(values)


def validate_filter(node: MetadataFilter) -> None:
    """Validate a filter tree.

    Raises:
        VectorStoreConfigError: if the tree is malformed.
    """
    if not isinstance(node, MetadataFilter):
        raise VectorStoreConfigError(
            f"filter node must be a MetadataFilter, got {type(node).__name__}"
        )
    if not isinstance(node.op, FilterOp):
        raise VectorStoreConfigError(
            f"filter operation must be a FilterOp, got {node.op!r}"
        )

    if node.op in LOGICAL_OPS:
        if node.field is not None:
            raise VectorStoreConfigError(
                f"logical filter {node.op.value!r} must not carry a field"
            )
        if node.op is FilterOp.NOT:
            if len(node.children) != 1:
                raise VectorStoreConfigError(
                    f"NOT filter takes exactly one child, got {len(node.children)}"
                )
        elif not node.children:
            raise VectorStoreConfigError(
                f"{node.op.value.upper()} filter requires at least one child"
            )
        if node.value is not None:
            raise VectorStoreConfigError(
                f"logical filter {node.op.value!r} must not carry a value"
            )
        for child in node.children:
            validate_filter(child)
        return

    if not node.field:
        raise VectorStoreConfigError(
            f"filter {node.op.value!r} requires a non-empty field name"
        )
    if node.children:
        raise VectorStoreConfigError(
            f"field filter {node.op.value!r} must not carry children"
        )

    if node.op is FilterOp.EXISTS:
        if not isinstance(node.value, bool):
            raise VectorStoreConfigError(
                "EXISTS filter value must be a bool (True: field exists, "
                "False: field missing)"
            )
        return

    if node.op in (FilterOp.IN, FilterOp.NOT_IN):
        if isinstance(node.value, (str, bytes)) or not isinstance(node.value, (list, tuple, set, frozenset)):
            raise VectorStoreConfigError(
                f"{node.op.value} filter requires a sequence of values, "
                f"got {type(node.value).__name__}"
            )
        if not node.value:
            raise VectorStoreConfigError(
                f"{node.op.value} filter requires at least one value"
            )
        for item in node.value:
            if not _scalar(item):
                raise VectorStoreConfigError(
                    f"{node.op.value} filter values must be scalars, got {item!r}"
                )
        return

    if node.value is None or not _scalar(node.value):
        raise VectorStoreConfigError(
            f"{node.op.value} filter requires a scalar value, got {node.value!r}"
        )
    if isinstance(node.value, bool) and node.op in ORDERED_OPS:
        raise VectorStoreConfigError(
            f"{node.op.value} filter does not accept boolean values"
        )


def eq(field: str, value: Any) -> MetadataFilter:
    """Field equals ``value``."""
    return MetadataFilter(FilterOp.EQ, field=field, value=value)


def ne(field: str, value: Any) -> MetadataFilter:
    """Field does not equal ``value``."""
    return MetadataFilter(FilterOp.NE, field=field, value=value)


def is_in(field: str, values: Any) -> MetadataFilter:
    """Field value is one of ``values``."""
    return MetadataFilter(FilterOp.IN, field=field, value=_as_values(field, values))


def not_in(field: str, values: Any) -> MetadataFilter:
    """Field value is none of ``values``."""
    return MetadataFilter(FilterOp.NOT_IN, field=field, value=_as_values(field, values))


def gt(field: str, value: Any) -> MetadataFilter:
    """Field is greater than ``value``."""
    return MetadataFilter(FilterOp.GT, field=field, value=value)


def gte(field: str, value: Any) -> MetadataFilter:
    """Field is greater than or equal to ``value``."""
    return MetadataFilter(FilterOp.GTE, field=field, value=value)


def lt(field: str, value: Any) -> MetadataFilter:
    """Field is less than ``value``."""
    return MetadataFilter(FilterOp.LT, field=field, value=value)


def lte(field: str, value: Any) -> MetadataFilter:
    """Field is less than or equal to ``value``."""
    return MetadataFilter(FilterOp.LTE, field=field, value=value)


def exists(field: str, present: bool = True) -> MetadataFilter:
    """Field is present (``present=True``) or missing (``present=False``)."""
    return MetadataFilter(FilterOp.EXISTS, field=field, value=bool(present))


def and_(*clauses: MetadataFilter) -> MetadataFilter:
    """Combine ``clauses`` with AND."""
    return MetadataFilter(FilterOp.AND, children=tuple(clauses))


def or_(*clauses: MetadataFilter) -> MetadataFilter:
    """Combine ``clauses`` with OR."""
    return MetadataFilter(FilterOp.OR, children=tuple(clauses))


def not_(clause: MetadataFilter) -> MetadataFilter:
    """Negate ``clause``."""
    return MetadataFilter(FilterOp.NOT, children=(clause,))
