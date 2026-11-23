"""Stage 6 Tests: Safe AST Calculator (Zero `eval`)."""
from __future__ import annotations

from tools.calculator import calculate


def test_basic_and_complex_operations():
    r1 = calculate("25 * 4 + sqrt(144)")
    assert r1["ok"] is True
    assert r1["result"] == 112

    r2 = calculate("2 ** 10")
    assert r2["ok"] is True
    assert r2["result"] == 1024

    r3 = calculate("root(27, 3) + factorial(5)")
    assert r3["ok"] is True
    assert r3["result"] == 123


def test_persian_digits_and_natural_persian_math():
    r1 = calculate("ریشه دوم ۱۴۴ به علاوه ۲۵ ضربدر ۴ چند می‌شود؟")
    assert r1["ok"] is True
    assert r1["result"] == 112

    r2 = calculate("۲۰ درصد ۴۵۰")
    assert r2["ok"] is True
    assert r2["result"] == 90


def test_division_by_zero_and_invalid_inputs():
    div0 = calculate("100 / 0")
    assert div0["ok"] is False
    assert "صفر" in div0["error"] or "zero" in div0["error"].lower()

    huge_pow = calculate("99 ** 999999")
    assert huge_pow["ok"] is False

    malicious = calculate("__import__('os').system('whoami')")
    assert malicious["ok"] is False
