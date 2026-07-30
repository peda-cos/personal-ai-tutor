"""A small, dependency-free algebra parser and answer-equivalence checker.

The grader never uses ``eval``. Expressions are represented as multivariate
polynomials with rational coefficients, which is enough for the K-12 algebra
item bank (including expanded/factored equivalence and linear/quadratic solving).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence


class AnswerParseError(ValueError):
    """Raised when learner input cannot be interpreted as the requested kind."""


_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


@dataclass(frozen=True)
class Poly:
    vars: tuple[str, ...]
    terms: dict[tuple[int, ...], Fraction]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "terms", {k: v for k, v in self.terms.items() if v != 0}
        )

    @classmethod
    def constant(cls, value: Fraction) -> Poly:
        return cls((), {(): value} if value else {})

    @classmethod
    def variable(cls, name: str) -> Poly:
        return cls((name,), {(1,): Fraction(1)})

    def _aligned(self, other: Poly) -> tuple[tuple[str, ...], Poly, Poly]:
        variables = tuple(sorted(set(self.vars) | set(other.vars)))
        return variables, self.reindexed(variables), other.reindexed(variables)

    def reindexed(self, variables: tuple[str, ...]) -> Poly:
        if variables == self.vars:
            return self
        index = {name: i for i, name in enumerate(variables)}
        terms: dict[tuple[int, ...], Fraction] = {}
        for exponents, coefficient in self.terms.items():
            new = [0] * len(variables)
            for name, exponent in zip(self.vars, exponents, strict=True):
                new[index[name]] = exponent
            terms[tuple(new)] = terms.get(tuple(new), Fraction(0)) + coefficient
        return Poly(variables, terms)

    def add(self, other: Poly) -> Poly:
        variables, left, right = self._aligned(other)
        terms = dict(left.terms)
        for exponents, coefficient in right.terms.items():
            terms[exponents] = terms.get(exponents, Fraction(0)) + coefficient
        return Poly(variables, terms)

    def sub(self, other: Poly) -> Poly:
        variables, left, right = self._aligned(other)
        terms = dict(left.terms)
        for exponents, coefficient in right.terms.items():
            terms[exponents] = terms.get(exponents, Fraction(0)) - coefficient
        return Poly(variables, terms)

    def mul(self, other: Poly) -> Poly:
        variables, left, right = self._aligned(other)
        terms: dict[tuple[int, ...], Fraction] = {}
        for le, lc in left.terms.items():
            for re_, rc in right.terms.items():
                exponents = tuple(a + b for a, b in zip(le, re_, strict=True))
                terms[exponents] = terms.get(exponents, Fraction(0)) + lc * rc
        return Poly(variables, terms)

    def scale(self, value: Fraction) -> Poly:
        return Poly(self.vars, {k: v * value for k, v in self.terms.items()})

    def pow(self, exponent: int) -> Poly:
        if exponent < 0:
            raise AnswerParseError("negative powers are only supported for constants")
        result = Poly.constant(Fraction(1))
        for _ in range(exponent):
            result = result.mul(self)
        return result

    def is_zero(self) -> bool:
        return not self.terms

    def is_constant(self) -> bool:
        return all(not any(exponents) for exponents in self.terms)

    def constant_value(self) -> Fraction:
        if not self.is_constant():
            raise AnswerParseError("expression is not a constant")
        return self.terms.get((), Fraction(0))

    def coefficient(self, variable: str) -> Fraction:
        if variable not in self.vars:
            return Fraction(0)
        index = self.vars.index(variable)
        total = Fraction(0)
        for exponents, coefficient in self.terms.items():
            if exponents[index] == 1 and all(
                e == 0 for i, e in enumerate(exponents) if i != index
            ):
                total += coefficient
        return total

    def degree(self) -> int:
        return max((sum(e) for e in self.terms), default=0)

    def to_text(self) -> str:
        if not self.terms:
            return "0"
        pieces: list[str] = []
        for exponents in sorted(self.terms, reverse=True):
            coefficient = self.terms[exponents]
            factors = []
            for name, exponent in zip(self.vars, exponents, strict=True):
                if exponent:
                    factors.append(name if exponent == 1 else f"{name}^{exponent}")
            body = "*".join(factors)
            if not body:
                pieces.append(str(coefficient))
            elif coefficient == 1:
                pieces.append(body)
            elif coefficient == -1:
                pieces.append(f"-{body}")
            else:
                pieces.append(f"{coefficient}*{body}")
        return " + ".join(pieces).replace("+ -", "- ")


@dataclass(frozen=True)
class Interval:
    lo: Fraction | None
    lo_closed: bool
    hi: Fraction | None
    hi_closed: bool

    def contains(self, value: Fraction) -> bool:
        if self.lo is not None:
            if value < self.lo or (value == self.lo and not self.lo_closed):
                return False
        if self.hi is not None:
            if value > self.hi or (value == self.hi and not self.hi_closed):
                return False
        return True


def normalize(text: str) -> str:
    text = text.strip()
    text = (
        text.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("×", "*")
        .replace("·", "*")
        .replace("÷", "/")
        .replace("²", "^2")
        .replace("³", "^3")
        .replace("**", "^")
    )
    return text


def _parse_number_word(text: str) -> Fraction | None:
    lowered = text.strip().lower()
    sign = 1
    if lowered.startswith("negative "):
        sign = -1
        lowered = lowered[len("negative ") :].strip()
    elif lowered.startswith("minus "):
        sign = -1
        lowered = lowered[len("minus ") :].strip()
    if lowered in _NUMBER_WORDS:
        return Fraction(sign * _NUMBER_WORDS[lowered])
    return None


def _tokenize(text: str) -> list[str]:
    cleaned = normalize(text)
    raw = re.findall(r"\d+\.\d+|\d+|\.\d+|[a-zA-Z]|[+\-*/^()=]", cleaned)
    if not raw:
        raise AnswerParseError("empty expression")
    tokens: list[str] = []
    for token in raw:
        starts_atom = token[0].isalnum() or token == "("
        prev_ends_atom = bool(tokens) and (
            tokens[-1][0].isdigit() or tokens[-1][0].isalpha() or tokens[-1] == ")"
        )
        if starts_atom and prev_ends_atom:
            tokens.append("*")
        tokens.append(token)
    return tokens


class _Parser:
    def __init__(self, text: str) -> None:
        self.tokens = _tokenize(text)
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self, expected: str | None = None) -> str:
        token = self.peek()
        if token is None:
            raise AnswerParseError("unexpected end of expression")
        if expected is not None and token != expected:
            raise AnswerParseError(f"expected {expected!r}, got {token!r}")
        self.pos += 1
        return token

    def parse(self) -> Poly:
        result = self.parse_expr()
        if self.peek() is not None:
            raise AnswerParseError(f"unexpected token {self.peek()!r}")
        return result

    def parse_expr(self) -> Poly:
        result = self.parse_term()
        while self.peek() in {"+", "-"}:
            op = self.take()
            rhs = self.parse_term()
            result = result.add(rhs) if op == "+" else result.sub(rhs)
        return result

    def parse_term(self) -> Poly:
        result = self.parse_unary()
        while self.peek() in {"*", "/"}:
            op = self.take()
            rhs = self.parse_unary()
            if op == "*":
                result = result.mul(rhs)
            else:
                if not rhs.is_constant():
                    raise AnswerParseError("division by a variable is not supported")
                divisor = rhs.constant_value()
                if divisor == 0:
                    raise AnswerParseError("division by zero")
                result = result.scale(Fraction(1, 1) / divisor)
        return result

    def parse_unary(self) -> Poly:
        token = self.peek()
        if token == "+":
            self.take()
            return self.parse_unary()
        if token == "-":
            self.take()
            return self.parse_unary().scale(Fraction(-1))
        return self.parse_power()

    def parse_power(self) -> Poly:
        base = self.parse_primary()
        if self.peek() != "^":
            return base
        self.take("^")
        exponent = self.parse_unary()
        if not exponent.is_constant():
            raise AnswerParseError("exponents must be constants")
        power = exponent.constant_value()
        if power.denominator != 1:
            raise AnswerParseError("non-integer exponents are not supported")
        value = int(power)
        if value < 0:
            if not base.is_constant() or base.constant_value() == 0:
                raise AnswerParseError("negative powers require a nonzero constant base")
            return Poly.constant(Fraction(1, 1) / (base.constant_value() ** -value))
        return base.pow(value)

    def parse_primary(self) -> Poly:
        token = self.peek()
        if token is None:
            raise AnswerParseError("unexpected end of expression")
        if token == "(":
            self.take("(")
            result = self.parse_expr()
            self.take(")")
            return result
        if token[0].isdigit() or token.startswith("."):
            self.take()
            try:
                return Poly.constant(Fraction(token))
            except (ValueError, ZeroDivisionError) as exc:
                raise AnswerParseError(f"invalid number {token!r}") from exc
        if token.isalpha() and len(token) == 1:
            self.take()
            return Poly.variable(token.lower())
        raise AnswerParseError(f"unexpected token {token!r}")


def parse_expression(text: str) -> Poly:
    value = normalize(text)
    if "=" in value:
        raise AnswerParseError("expected an expression, got an equation")
    word = _parse_number_word(value)
    if word is not None:
        return Poly.constant(word)
    return _Parser(value).parse()


def parse_equation(text: str) -> tuple[Poly, Poly]:
    value = normalize(text)
    if value.count("=") != 1:
        raise AnswerParseError("equation must contain exactly one '='")
    left, right = value.split("=", 1)
    return parse_expression(left), parse_expression(right)


def solve_linear(lhs: Poly, rhs: Poly, var: str = "x") -> Fraction:
    diff = lhs.sub(rhs)
    all_vars = diff.vars
    variable = var.lower()
    if variable not in all_vars:
        raise AnswerParseError(f"equation has no variable {var!r}")
    index = all_vars.index(variable)
    for exponents in diff.terms:
        if exponents[index] > 1 or any(
            e > 0 for i, e in enumerate(exponents) if i != index
        ):
            raise AnswerParseError("not a single-variable linear equation")
    coefficient = diff.coefficient(variable)
    if coefficient == 0:
        raise AnswerParseError("equation is not uniquely solvable")
    constant = diff.terms.get(tuple(0 for _ in all_vars), Fraction(0))
    return -constant / coefficient


def solve_quadratic(lhs: Poly, rhs: Poly, var: str = "x") -> tuple[float, ...]:
    diff = lhs.sub(rhs)
    variable = var.lower()
    if variable not in diff.vars:
        raise AnswerParseError(f"equation has no variable {var!r}")
    index = diff.vars.index(variable)
    if any(
        any(e > 0 for i, e in enumerate(exponents) if i != index)
        for exponents in diff.terms
    ):
        raise AnswerParseError("not a single-variable quadratic")
    if any(exponents[index] > 2 for exponents in diff.terms):
        raise AnswerParseError("not a quadratic")
    a = _coefficient_at(diff, index, 2)
    b = _coefficient_at(diff, index, 1)
    c = _coefficient_at(diff, index, 0)
    if a == 0:
        raise AnswerParseError("quadratic coefficient is zero")
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return ()
    root = math.sqrt(float(discriminant))
    denominator = 2 * float(a)
    return (
        (-float(b) + root) / denominator,
        (-float(b) - root) / denominator,
    )


def _coefficient_at(poly: Poly, index: int, exponent: int) -> Fraction:
    total = Fraction(0)
    for exponents, coefficient in poly.terms.items():
        if exponents[index] == exponent and all(
            e == 0 for i, e in enumerate(exponents) if i != index
        ):
            total += coefficient
    return total


def _number(text: str) -> Fraction:
    value = normalize(text).strip().rstrip(".")
    if not value:
        raise AnswerParseError("empty number")
    match = re.fullmatch(r"[a-zA-Z]\s*=\s*(.+)", value)
    if match:
        value = match.group(1).strip()
    word = _parse_number_word(value)
    if word is not None:
        return word
    if value.startswith("(") and value.endswith(")"):
        value = value[1:-1].strip()
    parsed = parse_expression(value)
    if not parsed.is_constant():
        raise AnswerParseError(f"{text!r} is not a number")
    return parsed.constant_value()


def _numbers(text: str) -> list[Fraction]:
    text = normalize(text)
    if "," in text or ";" in text:
        parts: Iterable[str] = re.split(r"[,;]", text)
    elif re.search(r"\bor\b", text, flags=re.IGNORECASE):
        parts = re.split(r"\bor\b", text, flags=re.IGNORECASE)
    else:
        parts = [text]
    values: list[Fraction] = []
    for part in parts:
        item = part.strip().strip("{}()[]").strip()
        if item:
            values.append(_number(item))
    if not values:
        raise AnswerParseError("no numbers found")
    return values


def _close(left: Fraction, right: Fraction) -> bool:
    return abs(float(left) - float(right)) <= 1e-6 * max(1.0, abs(float(right)))


_BOUND = r"-?\d+(?:\.\d+)?(?:/\d+)?"


def _bound(text: str) -> Fraction:
    return _number(text)


def _interval_from_text(text: str) -> Interval:
    value = normalize(text).strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"^([a-zA-Z])\s*=\s*", "", value)
    chain = re.fullmatch(
        rf"({_BOUND})\s*(<=|<)\s*([a-zA-Z])\s*(<=|<)\s*({_BOUND})",
        value,
    )
    if chain:
        lo = _bound(chain.group(1))
        hi = _bound(chain.group(5))
        return Interval(lo, chain.group(2) == "<=", hi, chain.group(4) == "<=")
    single = re.fullmatch(rf"([a-zA-Z])\s*(<=|>=|<|>)\s*({_BOUND})", value)
    if single:
        var, op, number = single.groups()
        bound = _bound(number)
        if op == ">":
            return Interval(bound, False, None, False)
        if op == ">=":
            return Interval(bound, True, None, False)
        if op == "<":
            return Interval(None, False, bound, False)
        return Interval(None, False, bound, True)
    reverse = re.fullmatch(rf"({_BOUND})\s*(<=|<|>=|>)\s*([a-zA-Z])", value)
    if reverse:
        number, op, var = reverse.groups()
        bound = _bound(number)
        if op == ">":
            return Interval(None, False, bound, False)
        if op == ">=":
            return Interval(None, False, bound, True)
        if op == "<":
            return Interval(bound, False, None, False)
        return Interval(bound, True, None, False)
    raise AnswerParseError(f"cannot parse inequality {text!r}")


def parse_interval(text: str) -> tuple[Interval, ...]:
    value = normalize(text).strip()
    parts = re.split(r"\bor\b", value, flags=re.IGNORECASE)
    intervals: list[Interval] = []
    for part in parts:
        chunk = part.strip()
        if re.search(r"\band\b", chunk, flags=re.IGNORECASE):
            pieces = [
                _interval_from_text(piece)
                for piece in re.split(r"\band\b", chunk, flags=re.IGNORECASE)
                if piece.strip()
            ]
            intervals.append(_intersect(pieces))
        else:
            intervals.append(_interval_from_text(chunk))
    return tuple(_merge(sorted(intervals, key=_interval_sort_key)))


def _interval_sort_key(interval: Interval) -> tuple[float, int, float]:
    return (
        float("-inf") if interval.lo is None else float(interval.lo),
        0 if interval.lo_closed else 1,
        float("inf") if interval.hi is None else float(interval.hi),
    )


def _intersect(intervals: Sequence[Interval]) -> Interval:
    lo: Fraction | None = None
    lo_closed = False
    hi: Fraction | None = None
    hi_closed = False
    for interval in intervals:
        if interval.lo is not None and (lo is None or interval.lo > lo):
            lo, lo_closed = interval.lo, interval.lo_closed
        if interval.hi is not None and (hi is None or interval.hi < hi):
            hi, hi_closed = interval.hi, interval.hi_closed
        if lo is not None and hi is not None and (lo > hi or (lo == hi and not (lo_closed and hi_closed))):
            raise AnswerParseError("intersection is empty")
    return Interval(lo, lo_closed, hi, hi_closed)


def _merge(intervals: Sequence[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for interval in intervals:
        if not merged:
            merged.append(interval)
            continue
        previous = merged[-1]
        if previous.hi is None or interval.lo is None:
            merged[-1] = Interval(previous.lo, previous.lo_closed, None, False)
            continue
        if interval.lo < previous.hi or (
            interval.lo == previous.hi and (previous.hi_closed or interval.lo_closed)
        ):
            merged[-1] = Interval(
                previous.lo,
                previous.lo_closed,
                interval.hi,
                interval.hi_closed,
            )
        else:
            merged.append(interval)
    return merged


def _intervals_equal(left: Sequence[Interval], right: Sequence[Interval]) -> bool:
    if len(left) != len(right):
        return False
    for a, b in zip(left, right, strict=True):
        if a.lo is None and b.lo is not None or a.lo is not None and b.lo is None:
            return False
        if a.lo is not None and b.lo is not None:
            if not _close(a.lo, b.lo) or a.lo_closed != b.lo_closed:
                return False
        if a.hi is None and b.hi is not None or a.hi is not None and b.hi is None:
            return False
        if a.hi is not None and b.hi is not None:
            if not _close(a.hi, b.hi) or a.hi_closed != b.hi_closed:
                return False
    return True


def check_answer(
    kind: str,
    expected: str,
    user: str,
    options: Sequence[str] | None = None,
) -> bool:
    """Return whether ``user`` is equivalent to ``expected`` for the item kind."""
    try:
        if not user.strip():
            return False
        if kind == "number":
            return _close(_number(user), _number(expected))
        if kind == "expression":
            expected_equation = "=" in normalize(expected)
            if expected_equation:
                expected_lhs, expected_rhs = parse_equation(expected)
                expected_poly = expected_lhs.sub(expected_rhs)
            else:
                expected_poly = parse_expression(expected)
            user_equation = "=" in normalize(user)
            if user_equation:
                user_lhs, user_rhs = parse_equation(user)
                user_poly = user_lhs.sub(user_rhs)
            else:
                user_poly = parse_expression(user)
            difference = expected_poly.sub(user_poly)
            # The two sides of an equation may be written in either order.
            return difference.is_zero() or (
                expected_equation and user_equation and expected_poly.add(user_poly).is_zero()
            )
        if kind == "pair":
            expected_values = _numbers(expected)
            user_values = _numbers(user)
            return len(expected_values) == 2 and len(user_values) == 2 and all(
                _close(a, b) for a, b in zip(expected_values, user_values, strict=True)
            )
        if kind == "set":
            expected_values = sorted(_numbers(expected), key=float)
            user_values = sorted(_numbers(user), key=float)
            return len(expected_values) == len(user_values) and all(
                _close(a, b) for a, b in zip(expected_values, user_values, strict=True)
            )
        if kind == "interval":
            return _intervals_equal(parse_interval(expected), parse_interval(user))
        if kind == "choice":
            if not options:
                raise AnswerParseError("choice items require options")
            expected_index = _option_index(expected, options)
            user_index = _option_index(user, options)
            return expected_index == user_index
        if kind == "boolean":
            allowed = {"yes", "true", "1"}
            denied = {"no", "false", "0"}
            normalized = normalize(user).strip().lower()
            expected_bool = normalize(expected).strip().lower() in allowed
            if normalized in allowed:
                return expected_bool
            if normalized in denied:
                return not expected_bool
            raise AnswerParseError("expected yes/no")
    except (AnswerParseError, ValueError, ZeroDivisionError, OverflowError):
        return False
    raise AnswerParseError(f"unknown answer kind {kind!r}")


def _option_index(value: str, options: Sequence[str]) -> int:
    cleaned = normalize(value).strip().lower()
    if len(cleaned) == 1 and cleaned.isalpha():
        index = ord(cleaned) - ord("a")
        if 0 <= index < len(options):
            return index
    try:
        number = int(cleaned)
        if 1 <= number <= len(options):
            return number - 1
    except ValueError:
        pass
    for index, option in enumerate(options):
        if normalize(option).strip().lower() == cleaned:
            return index
    raise AnswerParseError("choice was not one of the options")


def describe_equation(text: str) -> str:
    """Classify a linear equation for the photo/text intake path."""
    lhs, rhs = parse_equation(text)
    diff = lhs.sub(rhs)
    if "x" not in diff.vars:
        raise AnswerParseError("equation has no x variable")
    if diff.degree() > 1:
        raise AnswerParseError("only linear equations are supported for intake")
    coefficient = diff.coefficient("x")
    if coefficient == 0:
        raise AnswerParseError("equation is not uniquely solvable")
    variable_on_left = lhs.degree() >= 1
    variable_on_right = rhs.degree() >= 1
    if variable_on_left and variable_on_right:
        return "variables_both_sides"
    side = lhs if variable_on_left else rhs
    constant = side.terms.get((0,), Fraction(0))
    if coefficient == 1:
        return "one_step_add_sub" if constant != 0 else "one_step_mul_div"
    return "two_step_equations" if constant != 0 else "one_step_mul_div"
