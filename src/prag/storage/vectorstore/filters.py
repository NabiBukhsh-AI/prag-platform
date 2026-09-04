"""A small, vendor-neutral filter dialect.

Filters cross a module boundary, so they cannot be a Qdrant ``Filter`` or an OpenSearch query
body. They are a plain nested mapping that each store adapter translates into its own dialect,
and that translation is the only place a vendor's query language appears.

The dialect is deliberately small — equality, membership, negation, ranges, and boolean
composition. Everything the retrieval path actually needs is expressible in it, and every
operator added here has to be implementable by every backend, so the bar for adding one is that
retrieval genuinely cannot work without it.

The operator that matters most is ``$in`` against ``acl_hash``: it is how a principal's
permission set becomes an index-side filter rather than a post-hoc scan, which is what keeps
data the caller may not see from ever being fetched.
"""

from __future__ import annotations

from typing import Any

__all__ = ["FilterError", "acl_filter", "matches"]


class FilterError(ValueError):
    """A malformed filter expression.

    Raised at match time rather than silently returning nothing. A filter that matches nothing
    because it was misspelled looks exactly like a filter that matches nothing because the
    corpus is empty, and the two need very different responses.
    """


def _compare(operator: str, actual: Any, expected: Any) -> bool:
    match operator:
        case "$eq":
            return bool(actual == expected)
        case "$ne":
            return bool(actual != expected)
        case "$in":
            return actual in expected if isinstance(expected, list | tuple | set) else False
        case "$nin":
            return actual not in expected if isinstance(expected, list | tuple | set) else True
        case "$gt" | "$gte" | "$lt" | "$lte":
            if actual is None:
                # A missing field is not less than anything. Treating absence as zero would
                # sweep undated documents into every "newer than" filter.
                return False
            try:
                return {
                    "$gt": actual > expected,
                    "$gte": actual >= expected,
                    "$lt": actual < expected,
                    "$lte": actual <= expected,
                }[operator]
            except TypeError as exc:
                raise FilterError(
                    f"cannot compare {type(actual).__name__} with {type(expected).__name__}"
                ) from exc
        case "$exists":
            return (actual is not None) is bool(expected)
        case _:
            raise FilterError(f"unknown filter operator {operator!r}")


def matches(payload: dict[str, Any], expression: dict[str, Any] | None) -> bool:
    """Whether a payload satisfies a filter expression.

    An empty or absent expression matches everything, which is the correct reading of "no
    filter" and keeps callers from special-casing the unfiltered path.
    """
    if not expression:
        return True

    for key, condition in expression.items():
        if key == "$and":
            if not all(matches(payload, sub) for sub in condition):
                return False
        elif key == "$or":
            if not any(matches(payload, sub) for sub in condition):
                return False
        elif key == "$not":
            if matches(payload, condition):
                return False
        elif key.startswith("$"):
            raise FilterError(f"unknown top-level filter key {key!r}")
        elif isinstance(condition, dict):
            actual = payload.get(key)
            for operator, expected in condition.items():
                if not _compare(operator, actual, expected):
                    return False
        elif payload.get(key) != condition:
            return False

    return True


def acl_filter(tenant_id: str, acl_hashes: tuple[str, ...]) -> dict[str, Any]:
    """The filter every retrieval leg must carry.

    Tenant equality plus ACL membership, applied at the index rather than after it. Filtering
    downstream would mean fetching rows the caller may not see, and data that has been fetched
    can leak through a log line, a cache entry, or a timing difference even when it never
    reaches the response.

    ``public`` is always included: content marked public is visible to every principal within
    the tenant, and omitting it would make an unprivileged caller unable to see anything at all.
    """
    return {
        "tenant_id": tenant_id,
        "acl_hash": {"$in": [*acl_hashes, "public"]},
    }
