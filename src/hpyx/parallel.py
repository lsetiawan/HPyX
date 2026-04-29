"""hpyx.parallel — Python-callback parallel algorithms over integer ranges
and iterables.

Every function takes a policy (from ``hpyx.execution``) as the first
argument.  When the policy carries the ``task`` tag, the function returns
an ``hpyx.Future[T]`` instead of the synchronous result.

For ``par`` and ``par_unseq`` policies with Python callbacks, each iteration
is submitted as an independent ``hpyx.async_`` task on an HPX worker thread.
The ``seq`` and ``unseq`` policies call the C++ layer directly for zero
overhead.  Pure C++ kernels (no Python callback) can use HPX parallel
policies natively — see ``hpyx.kernels``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable
from typing import Any, Union

from hpyx import _core, _runtime
from hpyx.execution import _Policy
from hpyx.futures import Future, async_


def _task_not_supported(name: str) -> str:
    return (
        f"Task variant of {name} is not yet supported. "
        "Use a synchronous policy (e.g. par, seq) instead."
    )


def _token_fields(policy: _Policy) -> tuple:
    t = policy._token()
    return (t.kind, t.task, t.chunk, t.chunk_size)


def for_loop(
    policy: _Policy,
    first: int,
    last: int,
    body: Callable[[int], None],
) -> Union[None, Future]:
    """Invoke ``body(i)`` for i in [first, last) under ``policy``."""
    _runtime.ensure_started()
    if policy.task:
        raise NotImplementedError(
            "Task variant of for_loop is not yet supported. "
            "Use a synchronous policy (e.g. par, seq) instead."
        )

    if policy.name in ("par", "par_unseq"):
        futs = [async_(body, i) for i in range(first, last)]
        for f in futs:
            f.result()
    else:
        kind, task_flag, chunk, chunk_size = _token_fields(policy)
        _core.parallel.for_loop(kind, task_flag, chunk, chunk_size, first, last, body)
    return None


def for_each(
    policy: _Policy,
    iterable: Any,
    fn: Callable[[Any], None],
) -> Union[None, Future]:
    """Apply ``fn(x)`` to every element in ``iterable`` under ``policy``."""
    _runtime.ensure_started()
    if policy.task:
        raise NotImplementedError(
            "Task variant of for_each is not yet supported. "
            "Use a synchronous policy (e.g. par, seq) instead."
        )

    items = list(iterable)

    if policy.name in ("par", "par_unseq"):
        futs = [async_(fn, item) for item in items]
        for f in futs:
            f.result()
    else:
        kind, task_flag, chunk, chunk_size = _token_fields(policy)
        _core.parallel.for_each(kind, task_flag, chunk, chunk_size, items, fn)
    return None


def transform[T, U](
    policy: _Policy,
    iterable: Iterable[T],
    fn: Callable[[T], U],
) -> list[U]:
    """Apply ``fn`` to each element, return a new list of results."""
    _runtime.ensure_started()
    if policy.task:
        raise NotImplementedError(_task_not_supported("transform"))

    items = list(iterable)
    if policy.name in ("par", "par_unseq"):
        futs = [async_(fn, item) for item in items]
        return [f.result() for f in futs]
    return [fn(item) for item in items]


def reduce[T](
    policy: _Policy,
    iterable: Iterable[T],
    *,
    init: T,
    op: Callable[[T, T], T],
) -> T:
    """Reduce ``iterable`` with ``op``, starting from ``init``."""
    _runtime.ensure_started()
    if policy.task:
        raise NotImplementedError(_task_not_supported("reduce"))

    items = list(iterable)
    return functools.reduce(op, items, init)


def transform_reduce[T, U](
    policy: _Policy,
    iterable: Iterable[T],
    *,
    init: U,
    reduce_op: Callable[[U, U], U],
    transform_op: Callable[[T], U],
) -> U:
    """Transform each element with ``transform_op`` then reduce with ``reduce_op``."""
    _runtime.ensure_started()
    if policy.task:
        raise NotImplementedError(_task_not_supported("transform_reduce"))

    items = list(iterable)
    if policy.name in ("par", "par_unseq"):
        futs = [async_(transform_op, item) for item in items]
        transformed = [f.result() for f in futs]
    else:
        transformed = [transform_op(item) for item in items]
    return functools.reduce(reduce_op, transformed, init)


# ---------------------------------------------------------------------------
# Search algorithms
# ---------------------------------------------------------------------------

def count[T](
    policy: _Policy,
    iterable: Iterable[T],
    value: T,
) -> int:
    """Count elements equal to ``value``."""
    _runtime.ensure_started()
    if policy.task:
        raise NotImplementedError(_task_not_supported("count"))

    items = list(iterable)
    if policy.name in ("par", "par_unseq"):
        futs = [async_(lambda x: x == value, item) for item in items]
        return sum(f.result() for f in futs)
    return sum(1 for item in items if item == value)


def count_if[T](
    policy: _Policy,
    iterable: Iterable[T],
    pred: Callable[[T], bool],
) -> int:
    """Count elements satisfying ``pred``."""
    _runtime.ensure_started()
    if policy.task:
        raise NotImplementedError(_task_not_supported("count_if"))

    items = list(iterable)
    if policy.name in ("par", "par_unseq"):
        futs = [async_(pred, item) for item in items]
        return sum(bool(f.result()) for f in futs)
    return sum(1 for item in items if pred(item))


def find[T](
    policy: _Policy,
    iterable: Iterable[T],
    value: T,
) -> int:
    """Return index of first element equal to ``value``, or -1."""
    _runtime.ensure_started()
    if policy.task:
        raise NotImplementedError(_task_not_supported("find"))

    items = list(iterable)
    if policy.name in ("par", "par_unseq"):
        futs = [async_(lambda x: x == value, item) for item in items]
        for i, f in enumerate(futs):
            if f.result():
                return i
        return -1
    for i, item in enumerate(items):
        if item == value:
            return i
    return -1


def find_if[T](
    policy: _Policy,
    iterable: Iterable[T],
    pred: Callable[[T], bool],
) -> int:
    """Return index of first element satisfying ``pred``, or -1."""
    _runtime.ensure_started()
    if policy.task:
        raise NotImplementedError(_task_not_supported("find_if"))

    items = list(iterable)
    if policy.name in ("par", "par_unseq"):
        futs = [async_(pred, item) for item in items]
        for i, f in enumerate(futs):
            if f.result():
                return i
        return -1
    for i, item in enumerate(items):
        if pred(item):
            return i
    return -1


def all_of[T](
    policy: _Policy,
    iterable: Iterable[T],
    pred: Callable[[T], bool],
) -> bool:
    """Return True if ``pred`` is true for all elements."""
    _runtime.ensure_started()
    if policy.task:
        raise NotImplementedError(_task_not_supported("all_of"))

    items = list(iterable)
    if policy.name in ("par", "par_unseq"):
        futs = [async_(pred, item) for item in items]
        return all(f.result() for f in futs)
    return all(pred(item) for item in items)


def any_of[T](
    policy: _Policy,
    iterable: Iterable[T],
    pred: Callable[[T], bool],
) -> bool:
    """Return True if ``pred`` is true for any element."""
    _runtime.ensure_started()
    if policy.task:
        raise NotImplementedError(_task_not_supported("any_of"))

    items = list(iterable)
    if policy.name in ("par", "par_unseq"):
        futs = [async_(pred, item) for item in items]
        return any(f.result() for f in futs)
    return any(pred(item) for item in items)


def none_of[T](
    policy: _Policy,
    iterable: Iterable[T],
    pred: Callable[[T], bool],
) -> bool:
    """Return True if ``pred`` is false for all elements."""
    _runtime.ensure_started()
    if policy.task:
        raise NotImplementedError(_task_not_supported("none_of"))

    items = list(iterable)
    if policy.name in ("par", "par_unseq"):
        futs = [async_(pred, item) for item in items]
        return not any(f.result() for f in futs)
    return not any(pred(item) for item in items)


# ---------------------------------------------------------------------------
# Sort algorithms
# ---------------------------------------------------------------------------

def sort[T](
    policy: _Policy,
    data: Iterable[T],
    *,
    key: Callable[[T], Any] | None = None,
    reverse: bool = False,
) -> list[T]:
    """Return a new sorted list."""
    _runtime.ensure_started()
    if policy.task:
        raise NotImplementedError(_task_not_supported("sort"))

    return sorted(data, key=key, reverse=reverse)


def stable_sort[T](
    policy: _Policy,
    data: Iterable[T],
    *,
    key: Callable[[T], Any] | None = None,
    reverse: bool = False,
) -> list[T]:
    """Return a new sorted list (stable)."""
    _runtime.ensure_started()
    if policy.task:
        raise NotImplementedError(_task_not_supported("stable_sort"))

    return sorted(data, key=key, reverse=reverse)


# ---------------------------------------------------------------------------
# Fill / copy / iota
# ---------------------------------------------------------------------------

def fill[T](
    policy: _Policy,
    n: int,
    value: T,
) -> list[T]:
    """Return a list of ``n`` copies of ``value``."""
    _runtime.ensure_started()
    if policy.task:
        raise NotImplementedError(_task_not_supported("fill"))

    return [value] * n


def fill_n[T](
    policy: _Policy,
    n: int,
    value: T,
) -> list[T]:
    """Alias for :func:`fill`."""
    return fill(policy, n, value)


def copy[T](
    policy: _Policy,
    iterable: Iterable[T],
) -> list[T]:
    """Return a new list copy of ``iterable``."""
    _runtime.ensure_started()
    if policy.task:
        raise NotImplementedError(_task_not_supported("copy"))

    return list(iterable)


def copy_if[T](
    policy: _Policy,
    iterable: Iterable[T],
    pred: Callable[[T], bool],
) -> list[T]:
    """Return a list of elements satisfying ``pred``."""
    _runtime.ensure_started()
    if policy.task:
        raise NotImplementedError(_task_not_supported("copy_if"))

    items = list(iterable)
    if policy.name in ("par", "par_unseq"):
        futs = [async_(pred, item) for item in items]
        return [item for item, f in zip(items, futs, strict=True) if f.result()]
    return [item for item in items if pred(item)]


def iota(
    policy: _Policy,
    n: int,
    start: int = 0,
) -> list[int]:
    """Return ``[start, start+1, ..., start+n-1]``."""
    _runtime.ensure_started()
    if policy.task:
        raise NotImplementedError(_task_not_supported("iota"))

    return list(range(start, start + n))


# ---------------------------------------------------------------------------
# Scan algorithms
# ---------------------------------------------------------------------------

def inclusive_scan[T](
    policy: _Policy,
    iterable: Iterable[T],
    *,
    op: Callable[[T, T], T],
) -> list[T]:
    """Return a list of running totals (inclusive)."""
    _runtime.ensure_started()
    if policy.task:
        raise NotImplementedError(_task_not_supported("inclusive_scan"))

    items = list(iterable)
    if not items:
        return []
    result = [items[0]]
    for item in items[1:]:
        result.append(op(result[-1], item))
    return result


def exclusive_scan[T](
    policy: _Policy,
    iterable: Iterable[T],
    *,
    init: T,
    op: Callable[[T, T], T],
) -> list[T]:
    """Return a list of running totals starting from ``init`` (exclusive)."""
    _runtime.ensure_started()
    if policy.task:
        raise NotImplementedError(_task_not_supported("exclusive_scan"))

    items = list(iterable)
    if not items:
        return []
    result = [init]
    for item in items[:-1]:
        result.append(op(result[-1], item))
    return result


__all__ = [
    "all_of",
    "any_of",
    "copy",
    "copy_if",
    "count",
    "count_if",
    "exclusive_scan",
    "fill",
    "fill_n",
    "find",
    "find_if",
    "for_each",
    "for_loop",
    "inclusive_scan",
    "iota",
    "none_of",
    "reduce",
    "sort",
    "stable_sort",
    "transform",
    "transform_reduce",
]
