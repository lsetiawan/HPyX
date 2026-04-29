"""hpyx.parallel — Python-callback parallel algorithms over integer ranges
and iterables.

Every function takes a policy (from `hpyx.execution`) as the first
argument. When the policy carries the `task` tag, the function returns
an `hpyx.Future[T]` instead of the synchronous result.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Union

from hpyx import _core, _runtime
from hpyx.execution import _Policy
from hpyx.futures import Future


def _token_fields(policy: _Policy) -> tuple:
    """Extract token fields from a Python _Policy as a tuple of ints."""
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
    kind, task, chunk, chunk_size = _token_fields(policy)
    _core.parallel.for_loop(kind, task, chunk, chunk_size, first, last, body)
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
    kind, task, chunk, chunk_size = _token_fields(policy)
    _core.parallel.for_each(kind, task, chunk, chunk_size, iterable, fn)
    return None


__all__ = ["for_loop", "for_each"]
