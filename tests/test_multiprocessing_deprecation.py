"""Test that hpyx.multiprocessing emits DeprecationWarning."""

from __future__ import annotations

import importlib
import sys
import warnings


def test_multiprocessing_import_warns():
    sys.modules.pop("hpyx.multiprocessing", None)
    sys.modules.pop("hpyx.multiprocessing._for_loop", None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        importlib.import_module("hpyx.multiprocessing")
        assert any(
            issubclass(item.category, DeprecationWarning)
            and "hpyx.multiprocessing is deprecated" in str(item.message)
            for item in w
        ), f"Expected deprecation warning; got: {[str(x.message) for x in w]}"


def test_multiprocessing_for_loop_warns():
    import hpyx
    import hpyx.multiprocessing as mp

    hpyx.init()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mp.for_loop(lambda x: None, [1, 2, 3], policy="seq")
        assert any(
            issubclass(item.category, DeprecationWarning)
            and "hpyx.multiprocessing.for_loop is deprecated" in str(item.message)
            for item in w
        ), f"Expected deprecation warning; got: {[str(x.message) for x in w]}"
