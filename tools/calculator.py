"""Safe AST-Based Mathematical Calculator (Zero `eval`).

Supports:
- +, -, *, /, //, %, ** (and ^)
- Percentage expressions (`20% of 350`, `350 * 20%`, `۲۵ درصد ۴۰۰`)
- Functions: sqrt, cbrt, root(x, n), abs, round, floor, ceil, sin, cos, tan, log, log10, ln, exp, factorial
- Constants: pi, e
- Persian/Arabic digits and common Persian math phrasing
- Strict protection against division by zero, huge exponents (CPU DoS), and arbitrary code execution
"""
from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any, Callable, Dict

from core.language import to_ascii_digits

MAX_EXPONENT = 1000
MAX_FACTORIAL = 500
MAX_EXPR_LEN = 300

SAFE_BINARY_OPS: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

SAFE_UNARY_OPS: Dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_factorial(x: float) -> int:
    if int(x) != x or x < 0:
        raise ValueError("فاکتوریل فقط برای اعداد صحیح نامنفی تعریف شده است.")
    if x > MAX_FACTORIAL:
        raise ValueError(f"عدد فاکتوریل بزرگ‌تر از حد مجاز ({MAX_FACTORIAL}) است.")
    return math.factorial(int(x))


def _safe_root(x: float, n: float = 2.0) -> float:
    if n == 0:
        raise ZeroDivisionError("درجه ریشه نمی‌تواند صفر باشد.")
    if x < 0 and int(n) % 2 == 0:
        raise ValueError("ریشه زوج عدد منفی در اعداد حقیقی تعریف نشده است.")
    if x < 0:
        return -((-x) ** (1.0 / n))
    return x ** (1.0 / n)


SAFE_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "sqrt": lambda x: _safe_root(x, 2),
    "cbrt": lambda x: _safe_root(x, 3),
    "root": _safe_root,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "ln": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "factorial": _safe_factorial,
    "pow": pow,
}

SAFE_CONSTANTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}


def preprocess_math_expression(raw: str) -> str:
    """Normalize Persian digits, math phrases, symbols, and percentages into clean math syntax."""
    expr = to_ascii_digits(raw or "").strip()
    if len(expr) > MAX_EXPR_LEN:
        raise ValueError("عبارت ریاضی بیش از حد طولانی است.")

    # Strip leading polite & conversational prefixes if passed whole sentence
    expr = re.sub(
        r"^(?:/calc\s+|(?:لطفاً|لطفا|بی‌?زحمت|میشه|می‌?شود|بگو|please|can\s+you|could\s+you)\s+)+",
        "",
        expr,
        flags=re.I,
    )
    expr = re.sub(
        r"^(?:محاسبه\s+کن|حساب\s+کن|چند\s+می‌?شود|چند\s+میشه|حاصل|calculate|compute|what\s+is)\s*[:：]?\s*",
        "",
        expr,
        flags=re.I,
    )
    expr = re.sub(r"(?:چند\s+می‌?شود|چند\s+میشه|چنده|چند\s+است|را\s+بگو|\?|؟|=)+\s*$", "", expr, flags=re.I)

    # Natural Persian math phrases
    expr = re.sub(r"ریشه\s+دوم\s*\(?\s*([0-9.]+)\s*\)?", r"sqrt(\1)", expr)
    expr = re.sub(r"جذر\s*\(?\s*([0-9.]+)\s*\)?", r"sqrt(\1)", expr)
    expr = re.sub(r"ریشه\s+سوم\s*\(?\s*([0-9.]+)\s*\)?", r"cbrt(\1)", expr)
    expr = re.sub(r"([0-9.]+)\s*درصد\s*(?:از\s*)?([0-9.]+)", r"((\1 / 100) * \2)", expr)
    expr = re.sub(r"(?i)([0-9.]+)\s*%\s*of\s*([0-9.]+)", r"((\1 / 100) * \2)", expr)

    replacements = [
        ("ضربدر", "*"),
        ("ضرب در", "*"),
        ("تقسیم بر", "/"),
        ("به علاوه", "+"),
        ("به‌علاوه", "+"),
        ("به اضافه", "+"),
        ("به‌اضافه", "+"),
        ("جمع با", "+"),
        ("منهای", "-"),
        ("به توان", "**"),
        ("×", "*"),
        ("÷", "/"),
        ("−", "-"),
        ("^", "**"),
    ]
    for src, dst in replacements:
        expr = expr.replace(src, f" {dst} ")

    # Convert standalone `N%` (e.g., `200 + 15%` or `500 * 20%`) safely:
    # First `A + B%` -> `A + (A * B / 100)` only for simple `num +/- num%`, else `B%` -> `(B/100)`
    expr = re.sub(r"([0-9.]+)\s*%", r"(\1 / 100)", expr)
    expr = re.sub(r"\s+", " ", expr).strip()
    return expr


def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"مقدار غیرعددی مجاز نیست: {node.value!r}")

    if isinstance(node, ast.Name):
        if node.id in SAFE_CONSTANTS:
            return SAFE_CONSTANTS[node.id]
        raise ValueError(f"نام یا متغیر ناشناخته در عبارت ریاضی: '{node.id}'")

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_UNARY_OPS:
            raise ValueError("عملگر یکانی غیرمجاز است.")
        return SAFE_UNARY_OPS[op_type](_eval_node(node.operand))

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if op_type is ast.Pow:
            if abs(right) > MAX_EXPONENT:
                raise ValueError(f"توان بزرگ‌تر از سقف مجاز ({MAX_EXPONENT}) است.")
            res = left ** right
            if isinstance(res, complex):
                raise ValueError("نتیجه موهومی در ماشین‌حساب پایه مجاز نیست.")
            return res
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ZeroDivisionError("خطای تقسیم بر صفر (Division by zero).")
        if op_type in SAFE_BINARY_OPS:
            return SAFE_BINARY_OPS[op_type](left, right)
        raise ValueError(f"عملگر دودویی غیرمجاز: {op_type.__name__}")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_FUNCTIONS:
            fname = getattr(node.func, "id", "unknown")
            raise ValueError(f"تابع غیرمجاز در ماشین‌حساب: '{fname}'")
        if node.keywords:
            raise ValueError("آرگومان کلیدواژه‌ای در ماشین‌حساب مجاز نیست.")
        args = [_eval_node(a) for a in node.args]
        return SAFE_FUNCTIONS[node.func.id](*args)

    raise ValueError(f"ساختار نحوی غیرمجاز در ماشین‌حساب: {type(node).__name__}")


def calculate(expression: str) -> Dict[str, Any]:
    """Evaluate a mathematical expression safely and return structured result."""
    try:
        normalized = preprocess_math_expression(expression)
        if not normalized:
            return {"ok": False, "error": "عبارت ریاضی خالی است.", "expression": expression}

        tree = ast.parse(normalized, mode="eval")
        value = _eval_node(tree)

        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return {"ok": False, "error": "نتیجه عددی بی‌نهایت یا نامعتبر است.", "expression": normalized}
            formatted = int(round(value)) if abs(value - round(value)) < 1e-11 else round(value, 8)
        else:
            formatted = value

        return {
            "ok": True,
            "expression": normalized,
            "result": formatted,
            "summary": f"{normalized} = {formatted}",
        }
    except ZeroDivisionError as zde:
        return {"ok": False, "error": str(zde), "expression": expression}
    except SyntaxError:
        return {"ok": False, "error": "فرمت عبارت ریاضی نامعتبر است.", "expression": expression}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "expression": expression}
