"""Tests for the IBKR numeric-cleanup helpers."""

from __future__ import annotations

from optionsbot.ibkr._util import clean_float, clean_int


def test_clean_float_passes_through_normal_floats() -> None:
    assert clean_float(1.5) == 1.5
    assert clean_float(-0.25) == -0.25
    assert clean_float(0.0) == 0.0


def test_clean_float_returns_none_for_none() -> None:
    assert clean_float(None) is None


def test_clean_float_returns_none_for_nan() -> None:
    assert clean_float(float("nan")) is None


def test_clean_float_handles_non_float_gracefully() -> None:
    """isnan raises TypeError on non-numeric inputs; pass them through."""
    # int passes isnan and returns itself
    assert clean_float(42) == 42


def test_clean_int_passes_through_normal_ints() -> None:
    assert clean_int(100) == 100
    assert clean_int(0) == 0
    assert clean_int(-5) == -5


def test_clean_int_returns_none_for_none() -> None:
    assert clean_int(None) is None


def test_clean_int_returns_none_for_nan() -> None:
    assert clean_int(float("nan")) is None


def test_clean_int_truncates_float_to_int() -> None:
    assert clean_int(3.7) == 3
    assert clean_int(-2.9) == -2


def test_clean_int_returns_none_for_unconvertible_string() -> None:
    """If a non-numeric string sneaks through, return None rather than crash."""
    assert clean_int("not-a-number") is None  # type: ignore[arg-type]
