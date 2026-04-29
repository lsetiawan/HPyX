"""Tests for hpyx.parallel — Python-callback parallel algorithms."""

import threading

import pytest

import hpyx
from hpyx.execution import par, par_unseq, seq, static_chunk_size, task, unseq


# ---- for_loop (seq) ----

def test_for_loop_seq_calls_body_in_order():
    order = []
    hpyx.parallel.for_loop(seq, 0, 10, lambda i: order.append(i))
    assert order == list(range(10))


def test_for_loop_seq_large_range():
    count = 0
    lock = threading.Lock()

    def body(i):
        nonlocal count
        with lock:
            count += 1

    hpyx.parallel.for_loop(seq, 0, 1000, body)
    assert count == 1000


def test_for_loop_seq_with_chunk_size():
    count = 0
    lock = threading.Lock()

    def body(i):
        nonlocal count
        with lock:
            count += 1

    hpyx.parallel.for_loop(seq.with_(static_chunk_size(10)), 0, 100, body)
    assert count == 100


def test_for_loop_propagates_exception():
    def body(i):
        if i == 5:
            raise ValueError(f"fail at {i}")

    with pytest.raises((ValueError, RuntimeError), match="fail at 5"):
        hpyx.parallel.for_loop(seq, 0, 10, body)


def test_for_loop_empty_range():
    called = False

    def body(i):
        nonlocal called
        called = True

    hpyx.parallel.for_loop(seq, 0, 0, body)
    assert not called


def test_for_loop_task_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        hpyx.parallel.for_loop(seq(task), 0, 10, lambda i: None)


# ---- for_loop (par) ----

def test_for_loop_par_visits_all():
    result = set()
    lock = threading.Lock()

    def body(i):
        with lock:
            result.add(i)

    hpyx.parallel.for_loop(par, 0, 100, body)
    assert result == set(range(100))


def test_for_loop_par_unseq_visits_all():
    result = set()
    lock = threading.Lock()

    def body(i):
        with lock:
            result.add(i)

    hpyx.parallel.for_loop(par_unseq, 0, 50, body)
    assert result == set(range(50))


def test_for_loop_par_empty_range():
    called = False

    def body(i):
        nonlocal called
        called = True

    hpyx.parallel.for_loop(par, 0, 0, body)
    assert not called


def test_for_loop_par_propagates_exception():
    def body(i):
        if i == 5:
            raise ValueError(f"fail at {i}")

    with pytest.raises((ValueError, RuntimeError), match="fail at 5"):
        hpyx.parallel.for_loop(par, 0, 10, body)


# ---- for_each (seq) ----

def test_for_each_seq_mutates_in_order():
    data = [0, 1, 2, 3, 4]
    result = []
    hpyx.parallel.for_each(seq, data, lambda x: result.append(x))
    assert result == [0, 1, 2, 3, 4]


def test_for_each_seq_visits_every_element():
    data = list(range(50))
    result = set()
    lock = threading.Lock()

    def visit(x):
        with lock:
            result.add(x)

    hpyx.parallel.for_each(seq, data, visit)
    assert result == set(range(50))


def test_for_each_empty_iterable():
    called = False

    def fn(x):
        nonlocal called
        called = True

    hpyx.parallel.for_each(seq, [], fn)
    assert not called


def test_for_each_task_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        hpyx.parallel.for_each(seq(task), [1, 2, 3], lambda x: None)


# ---- for_each (par) ----

def test_for_each_par_visits_all():
    data = list(range(100))
    result = set()
    lock = threading.Lock()

    def visit(x):
        with lock:
            result.add(x)

    hpyx.parallel.for_each(par, data, visit)
    assert result == set(range(100))


def test_for_each_par_unseq_visits_all():
    data = list(range(50))
    result = set()
    lock = threading.Lock()

    def visit(x):
        with lock:
            result.add(x)

    hpyx.parallel.for_each(par_unseq, data, visit)
    assert result == set(range(50))


def test_for_each_par_empty():
    called = False

    def fn(x):
        nonlocal called
        called = True

    hpyx.parallel.for_each(par, [], fn)
    assert not called
