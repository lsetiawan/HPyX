import pytest

import hpyx
from hpyx.runtime import HPXRuntime


def test_hpx_for_loop_basic():
    """Test basic for_loop functionality."""
    results = []
    
    def collect_value(i):
        results.append(i)
    
    with HPXRuntime():
        hpyx.for_loop(0, 5, collect_value)
    
    # Results should contain values from 0 to 4
    assert len(results) == 5
    assert set(results) == {0, 1, 2, 3, 4}


def test_hpx_for_loop_larger_range():
    """Test for_loop with a larger range."""
    results = []
    
    def collect_value(i):
        results.append(i)
    
    with HPXRuntime():
        hpyx.for_loop(10, 20, collect_value)
    
    # Results should contain values from 10 to 19
    assert len(results) == 10
    assert set(results) == set(range(10, 20))


def test_hpx_for_loop_computation():
    """Test for_loop with actual computation."""
    sum_result = [0]  # Use list to make it mutable in closure
    
    def add_square(i):
        sum_result[0] += i * i
    
    with HPXRuntime():
        hpyx.for_loop(1, 6, add_square)  # 1^2 + 2^2 + 3^2 + 4^2 + 5^2 = 55
    
    assert sum_result[0] == 55


def test_hpx_for_loop_empty_range():
    """Test for_loop with empty range."""
    calls = [0]
    
    def count_calls(i):
        calls[0] += 1
    
    with HPXRuntime():
        hpyx.for_loop(5, 5, count_calls)  # Empty range
    
    assert calls[0] == 0


def test_hpx_for_loop_with_closure():
    """Test for_loop with closure capturing external variables."""
    multiplier = 3
    results = []
    
    def multiply_and_collect(i):
        results.append(i * multiplier)
    
    with HPXRuntime():
        hpyx.for_loop(0, 4, multiply_and_collect)
    
    expected = [0, 3, 6, 9]
    assert len(results) == 4
    assert set(results) == set(expected)