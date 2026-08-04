"""Deterministic, multi-step Socratic algebra items.

Every family is a pure function of ``(concept, variant)``. Items contain the
Socratic step questions, three hint questions, and a worked example that uses
different numbers, so the tutor can scaffold without revealing the solution.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from .algebra import describe_equation, parse_equation, solve_linear
from .curriculum import Concept, Curriculum


class ItemError(ValueError):
    """Raised when an item cannot be generated."""


@dataclass(frozen=True)
class Step:
    question: str
    kind: str
    answer: str
    hints: tuple[str, str, str]
    worked: str
    options: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "question": self.question,
            "kind": self.kind,
            "answer": self.answer,
            "hints": list(self.hints),
            "worked": self.worked,
        }
        if self.options is not None:
            result["options"] = list(self.options)
        return result


@dataclass(frozen=True)
class Item:
    concept_id: str
    variant: int
    prompt: str
    steps: tuple[Step, ...]
    visual: dict[str, object] | None = None

    @property
    def final_answer(self) -> str:
        return self.steps[-1].answer

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "concept_id": self.concept_id,
            "variant": self.variant,
            "prompt": self.prompt,
            "steps": [step.to_dict() for step in self.steps],
        }
        if self.visual is not None:
            result["visual"] = self.visual
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> Item:
        raw_steps = raw.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ItemError("item has no steps")
        steps: list[Step] = []
        for raw_step in raw_steps:
            if not isinstance(raw_step, dict):
                raise ItemError("malformed step")
            try:
                hints_raw = raw_step.get("hints", [])
                if not isinstance(hints_raw, list) or len(hints_raw) < 3:
                    raise ItemError("step needs three hints")
                options_raw = raw_step.get("options")
                steps.append(
                    Step(
                        question=str(raw_step["question"]),
                        kind=str(raw_step["kind"]),
                        answer=str(raw_step["answer"]),
                        hints=tuple(str(h) for h in hints_raw[:3]),  # type: ignore[arg-type]
                        worked=str(raw_step["worked"]),
                        options=tuple(str(o) for o in options_raw)
                        if isinstance(options_raw, list)
                        else None,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ItemError("malformed step") from exc
        visual = raw.get("visual")
        return cls(
            concept_id=str(raw["concept_id"]),
            variant=int(str(raw.get("variant", 0))),
            prompt=str(raw["prompt"]),
            steps=tuple(steps),
            visual=visual if isinstance(visual, dict) else None,
        )


def _text(value: Fraction | int) -> str:
    fraction = value if isinstance(value, Fraction) else Fraction(value)
    if fraction.denominator == 1:
        return str(fraction.numerator)
    return f"{fraction.numerator}/{fraction.denominator}"


def _signed(value: int | Fraction) -> str:
    fraction = value if isinstance(value, Fraction) else Fraction(value)
    return f"+ {_text(fraction)}" if fraction >= 0 else f"- {_text(-fraction)}"


def _generic_hints(subject: str) -> tuple[str, str, str]:
    return (
        f"What operation would undo the {subject} first?",
        "Which move keeps both sides balanced?",
        "Try that move on both sides; what do you get?",
    )


def _worked_example(text: str) -> str:
    return f"For a similar problem: {text} What would you try here?"


def _step(
    question: str,
    answer: str,
    hints: Sequence[str] | None = None,
    kind: str = "number",
    worked: str | None = None,
    options: Sequence[str] | None = None,
) -> Step:
    chosen = tuple(hints or _generic_hints("operation"))
    if len(chosen) < 3:
        chosen = chosen + _generic_hints("operation")[: 3 - len(chosen)]
    return Step(
        question=question,
        kind=kind,
        answer=answer,
        hints=(chosen[0], chosen[1], chosen[2]),
        worked=worked or _worked_example("2x + 4 = 10, subtract 4 from both sides."),
        options=tuple(options) if options else None,
    )


def _visual(points: Sequence[tuple[int, int]]) -> dict[str, object]:
    return {
        "kind": "grid",
        "points": [list(point) for point in points],
        "x_range": [-5, 5],
        "y_range": [-5, 5],
    }


class ItemFactory:
    def __init__(self, curriculum: Curriculum) -> None:
        self.curriculum = curriculum
        self._families: dict[str, Callable[[Concept, int, random.Random], Item]] = {
            "number_line": _number_line,
            "integer_ops": _integer_ops,
            "fraction_ops": _fraction_ops,
            "decimal_percent": _decimal_percent,
            "order_ops": _order_ops,
            "exponents": _exponents,
            "square_roots": _square_roots,
            "proportions": _proportions,
            "evaluate_expression": _evaluate_expression,
            "combine_like": _combine_like,
            "distribute": _distribute,
            "simplify_expression": _simplify_expression,
            "translate_expression": _translate_expression,
            "one_step": _one_step,
            "two_step": _two_step,
            "both_sides": _both_sides,
            "fraction_equation": _fraction_equation,
            "literal_equation": _literal_equation,
            "linear_word": _linear_word,
            "inequality": _inequality,
            "coordinate": _coordinate,
            "slope": _slope_item,
            "line_equation": _line_equation,
            "systems": _systems,
            "polynomial_ops": _polynomial_ops,
            "factoring": _factoring,
            "quadratic": _quadratic,
            "rational": _rational,
            "functions": _functions,
        }

    def build(self, concept_id: str, variant: int) -> Item:
        try:
            concept = self.curriculum.concepts[concept_id]
        except KeyError as exc:
            raise ItemError(f"unknown concept {concept_id!r}") from exc
        family = self._families.get(concept.family)
        if family is None:
            raise ItemError(f"no item family {concept.family!r}")
        rng = _rng(concept_id, variant)
        return family(concept, variant, rng)

    def variants(self, concept_id: str, count: int = 5) -> list[Item]:
        return [self.build(concept_id, variant) for variant in range(count)]

    def build_from_equation(self, equation: str) -> Item:
        text = equation.strip().rstrip(".")
        lhs, rhs = parse_equation(text)
        concept_id = describe_equation(text)
        diff = lhs.sub(rhs)
        coefficient = diff.coefficient("x")
        if coefficient == 0:
            raise ItemError("equation is not solvable")
        variable_side = lhs if lhs.degree() >= 1 else rhs
        constant = variable_side.terms.get((0,), Fraction(0))
        solution = solve_linear(lhs, rhs, "x")
        model = f"Solve {text}."
        steps: tuple[Step, ...]

        if concept_id == "variables_both_sides":
            left_coefficient = lhs.coefficient("x")
            right_coefficient = rhs.coefficient("x")
            left_constant = lhs.terms.get((0,), Fraction(0))
            right_constant = rhs.terms.get((0,), Fraction(0))
            remaining = left_coefficient - right_coefficient
            new_constant = right_constant - left_constant
            steps = (
                _step(
                    "How many x terms remain after moving the x term on the right to the left?",
                    _text(remaining),
                    (
                        "Which coefficient would you subtract from both sides?",
                        "Subtract the same x term from both sides; what is left?",
                        "How many x's are on the left now?",
                    ),
                ),
                _step(
                    "After moving the x terms left and the constants right, "
                    "what constant is on the right?",
                    _text(new_constant),
                    _generic_hints("constant"),
                ),
                _step("What is x?", _text(solution)),
            )
        elif concept_id == "one_step_mul_div":
            side = lhs if lhs.degree() >= 1 else rhs
            factor = side.coefficient("x")
            steps = (
                _step(
                    "What number would you divide both sides by?",
                    _text(factor),
                    _generic_hints("coefficient"),
                ),
                _step("What is x?", _text(solution)),
            )
        elif concept_id == "one_step_add_sub":
            operation = "subtract" if constant > 0 else "add"
            steps = (
                _step(
                    f"What number would you {operation} on both sides to isolate x?",
                    _text(abs(constant)),
                    _generic_hints("constant"),
                ),
                _step("What is x?", _text(solution)),
            )
        else:
            operation = "subtract" if constant > 0 else "add"
            steps = (
                _step(
                    f"What number would you {operation} on both sides to start?",
                    _text(abs(constant)),
                    _generic_hints("constant"),
                ),
                _step(
                    "After that step, what is the variable term equal to?",
                    _text(solution * coefficient),
                    (
                        "What is left on the variable side?",
                        "Apply the same operation to both sides.",
                        "What is the variable term equal to now?",
                    ),
                ),
                _step("What is x?", _text(solution)),
            )
        return Item(
            concept_id=concept_id,
            variant=0,
            prompt=model,
            steps=steps,
        )


def _rng(concept_id: str, variant: int) -> random.Random:
    digest = hashlib.sha256(f"{concept_id}:{variant}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


# ---------------------------------------------------------------------------
# Number foundations


def _number_line(concept: Concept, variant: int, rng: random.Random) -> Item:
    if concept.family_mode == "compare":
        a = rng.randint(-9, 9)
        b = rng.randint(-9, 9)
        while b == a:
            b = rng.randint(-9, 9)
        greater = str(max(a, b))
        options = (str(a), str(b))
        return Item(
            concept.id,
            variant,
            f"Which integer is greater, {a} or {b}?",
            (
                _step(
                    "Which of the two numbers is farther to the right on the number line?",
                    greater,
                    (
                        "Which number is closer to positive infinity?",
                        "Remember that -9 is less than -1.",
                        "Compare their positions from left to right; which is greater?",
                    ),
                    kind="choice",
                    options=options,
                ),
            ),
        )
    if concept.family_mode == "absolute":
        n = rng.choice([-9, -7, -5, -3, -1, 2, 4, 6, 8])
        return Item(
            concept.id,
            variant,
            f"What is the absolute value of {n}?",
            (_step(f"How far is {n} from zero on the number line?", str(abs(n))),),
        )
    n = rng.randint(1, 9)
    return Item(
        concept.id,
        variant,
        f"Point P is {n} units to the right of 0 on a number line. What integer is at P?",
        (_step("Move right from zero by that many units; where do you land?", str(n)),),
    )


def _integer_ops(concept: Concept, variant: int, rng: random.Random) -> Item:
    a = rng.randint(-9, 9)
    b = rng.randint(-9, 9)
    if concept.family_mode == "addsub":
        return Item(
            concept.id,
            variant,
            f"Evaluate {a} + ({b}).",
            (
                _step(
                    "Do the signs match or differ, and what does that mean for the result?",
                    str(a + b),
                    (
                        "Adding a negative moves left on the number line.",
                        "If the signs differ, subtract the smaller absolute value.",
                        "What is the final signed sum?",
                    ),
                ),
            ),
        )
    while b == 0:
        b = rng.randint(-9, 9)
    return Item(
        concept.id,
        variant,
        f"Evaluate {a} * ({b}).",
        (
            _step(
                "Will the product be positive or negative, and what is its size?",
                str(a * b),
                (
                    "Same signs give a positive product.",
                    "Different signs give a negative product.",
                    "Multiply the absolute values, then attach the sign.",
                ),
            ),
        ),
    )


def _fraction_ops(concept: Concept, variant: int, rng: random.Random) -> Item:
    denominator = rng.choice([3, 4, 5, 6, 8, 10])
    numerator = rng.randint(1, denominator - 1)
    if concept.family_mode == "equiv":
        factor = rng.randint(2, 5)
        value = Fraction(numerator, denominator)
        return Item(
            concept.id,
            variant,
            f"Simplify {numerator * factor}/{denominator * factor}.",
            (
                _step(
                    "What is the fraction in lowest terms?",
                    _text(value),
                    (
                        "Look for a factor shared by both numbers.",
                        "Divide both parts by the same number.",
                        "What is the reduced fraction?",
                    ),
                    kind="expression",
                ),
            ),
        )
    if concept.family_mode == "addsub":
        other_denominator = rng.choice([3, 4, 5, 6, 8])
        other_numerator = rng.randint(1, other_denominator - 1)
        value = Fraction(numerator, denominator) + Fraction(
            other_numerator, other_denominator
        )
        return Item(
            concept.id,
            variant,
            f"Evaluate {numerator}/{denominator} + {other_numerator}/{other_denominator}.",
            (
                _step(
                    "What is the sum in lowest terms?",
                    _text(value),
                    (
                        "Find a common denominator first.",
                        "Rewrite each fraction with that denominator.",
                        "Add the numerators and simplify.",
                    ),
                    kind="expression",
                ),
            ),
        )
    other = Fraction(rng.randint(1, 8), rng.randint(2, 9))
    value = Fraction(numerator, denominator) * other
    return Item(
        concept.id,
        variant,
        f"Evaluate ({numerator}/{denominator}) * ({other.numerator}/{other.denominator}).",
        (
            _step(
                "Multiply across, then simplify; what fraction do you get?",
                _text(value),
                (
                    "Multiply the numerators and denominators separately.",
                    "Look for common factors before multiplying.",
                    "What is the simplified product?",
                ),
                kind="expression",
            ),
        ),
    )


def _decimal_percent(concept: Concept, variant: int, rng: random.Random) -> Item:
    if concept.family_mode == "change":
        old = rng.choice([20, 25, 40, 50, 80])
        new = old + rng.choice([5, 10, 15, 20])
        answer = Fraction(new - old, old) * 100
        return Item(
            concept.id,
            variant,
            f"A price changes from {old} to {new}. What is the percent increase?",
            (
                _step(
                    "What is the change, and what original amount do you divide it by?",
                    _text(answer),
                    (
                        "Subtract the old value from the new value.",
                        "Divide the change by the original value.",
                        "Multiply by 100 for a percent.",
                    ),
                ),
            ),
        )
    decimal = rng.choice([0.15, 0.25, 0.4, 0.6, 0.75, 0.8])
    return Item(
        concept.id,
        variant,
        f"Write {decimal} as a percent (enter just the number).",
        (
            _step(
                "How many places and in which direction do you move the decimal point?",
                _text(Fraction(str(decimal)) * 100),
                (
                    "Percent means out of 100.",
                    "Move the decimal point two places to the right.",
                    "What number goes before the percent sign?",
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Expressions and equations


def _order_ops(concept: Concept, variant: int, rng: random.Random) -> Item:
    a = rng.randint(2, 9)
    b = rng.randint(1, 9)
    c = rng.randint(1, 9)
    if concept.family_mode == "properties":
        if a == 2 and b == 2:
            b = 3
        options = (f"{a}x + {a * b}", f"{a}x + {b}", f"{a + b}x", f"{a}x + {a + b}")
        answer = f"{a}x + {a * b}"
        return Item(
            concept.id,
            variant,
            f"Which expression is equivalent to {a}(x + {b})?",
            (
                _step(
                    "Which expression distributes the factor to both terms?",
                    answer,
                    (
                        "Multiply the outside factor by each term inside.",
                        "The constant term should be a product.",
                        "Which option shows both products?",
                    ),
                    kind="choice",
                    options=options,
                ),
            ),
        )
    return Item(
        concept.id,
        variant,
        f"Evaluate {a} + {b} * {c}.",
        (
            _step(
                "Which operation happens first?",
                str(a + b * c),
                (
                    "Multiplication comes before addition.",
                    "Multiply before adding.",
                    "What is the final value?",
                ),
            ),
        ),
    )


def _exponents(concept: Concept, variant: int, rng: random.Random) -> Item:
    if concept.family_mode == "evaluate":
        base = rng.randint(2, 6)
        exponent = rng.randint(2, 4)
        return Item(
            concept.id,
            variant,
            f"Evaluate {base}^{exponent}.",
            (
                _step(
                    "How many times does the base appear in the product?",
                    str(base**exponent),
                    (
                        "An exponent counts repeated factors.",
                        f"Write {base} as a factor {exponent} times.",
                        "What is the product?",
                    ),
                ),
            ),
        )
    if concept.family_mode == "negative":
        base = rng.randint(2, 5)
        exponent = rng.choice([0, -1, -2, -3])
        answer = Fraction(1, base**-exponent) if exponent < 0 else Fraction(1)
        return Item(
            concept.id,
            variant,
            f"Evaluate {base}^{exponent}.",
            (
                _step(
                    "What does a negative or zero exponent do to the base?",
                    _text(answer),
                    (
                        "A zero exponent gives 1 for a nonzero base.",
                        "A negative exponent moves the power below 1.",
                        "What fraction do you get?",
                    ),
                    kind="expression",
                ),
            ),
        )
    if concept.family_mode == "scientific":
        coefficient = rng.choice([2.5, 3.2, 4.5, 6.1])
        exponent = rng.randint(2, 5)
        value = Fraction(str(coefficient)) * 10**exponent
        return Item(
            concept.id,
            variant,
            f"Write {coefficient} x 10^{exponent} as an ordinary number.",
            (
                _step(
                    "Which direction and how many places does the decimal move?",
                    _text(value),
                    (
                        "A positive exponent makes the number larger.",
                        "Move the decimal point to the right.",
                        "What is the expanded number?",
                    ),
                ),
            ),
        )
    if concept.family_mode == "product":
        a, b = rng.randint(2, 6), rng.randint(2, 6)
        return Item(
            concept.id,
            variant,
            f"Simplify x^{a} * x^{b}.",
            (
                _step(
                    "When multiplying the same base, what happens to the exponents?",
                    f"x^{a + b}",
                    (
                        "Count the total number of x factors.",
                        "Add the exponents; keep the base.",
                        "What is the exponent in the simplified power?",
                    ),
                    kind="expression",
                ),
            ),
        )
    if concept.family_mode == "quotient":
        a = rng.randint(4, 9)
        b = rng.randint(1, a - 1)
        return Item(
            concept.id,
            variant,
            f"Simplify x^{a} / x^{b}.",
            (
                _step(
                    "When dividing the same base, what happens to the exponents?",
                    f"x^{a - b}",
                    (
                        "Cancel matching x factors.",
                        "Subtract the exponents; keep the base.",
                        "What is the remaining power?",
                    ),
                    kind="expression",
                ),
            ),
        )
    if concept.family_mode == "power":
        a, b = rng.randint(2, 4), rng.randint(2, 3)
        return Item(
            concept.id,
            variant,
            f"Simplify (x^{a})^{b}.",
            (
                _step(
                    "What happens to the exponents when a power is raised to a power?",
                    f"x^{a * b}",
                    (
                        "Count groups of x factors.",
                        "Multiply the exponents.",
                        "What is the new exponent?",
                    ),
                    kind="expression",
                ),
            ),
        )
    raise ItemError(f"unknown exponents mode {concept.family_mode}")


def _square_roots(concept: Concept, variant: int, rng: random.Random) -> Item:
    root = rng.randint(2, 12)
    return Item(
        concept.id,
        variant,
        f"Evaluate sqrt({root * root}).",
        (
            _step(
                "Which nonnegative number squares to this value?",
                str(root),
                (
                    "Look for a number whose square is the radicand.",
                    "The square root is nonnegative.",
                    "What number times itself gives the radicand?",
                ),
            ),
        ),
    )


def _proportions(concept: Concept, variant: int, rng: random.Random) -> Item:
    a = rng.randint(2, 5)
    b = rng.randint(2, 6)
    multiplier = rng.randint(2, 4)
    missing = b * multiplier
    return Item(
        concept.id,
        variant,
        f"Solve n/{a} = {missing}/{a * multiplier}.",
        (
            _step(
                "What number is missing so the two ratios are equal?",
                str(b),
                (
                    "Cross multiply the two ratios.",
                    "Compare the denominators to find the scale factor.",
                    "What value of n makes the ratios equal?",
                ),
            ),
        ),
    )


def _evaluate_expression(concept: Concept, variant: int, rng: random.Random) -> Item:
    a = rng.randint(2, 6)
    b = rng.randint(1, 9)
    x = rng.randint(1, 6)
    if concept.family_mode == "identify":
        choose_coefficient = rng.choice([True, False])
        if choose_coefficient:
            return Item(
                concept.id,
                variant,
                f"In the expression {a}x + {b}, what is the coefficient of x?",
                (_step("Which number multiplies the variable?", str(a)),),
            )
        return Item(
            concept.id,
            variant,
            f"In the expression {a}x + {b}, what is the constant term?",
            (_step("Which term has no variable attached?", str(b)),),
        )
    return Item(
        concept.id,
        variant,
        f"Evaluate {a}x + {b} when x = {x}.",
        (
            _step(
                "Substitute the value of x, then multiply before adding; what do you get?",
                str(a * x + b),
                (
                    "Replace x with the given value.",
                    "Multiply the coefficient by that value.",
                    "Add the constant last.",
                ),
            ),
        ),
    )


def _combine_like(concept: Concept, variant: int, rng: random.Random) -> Item:
    if concept.family_mode == "identify":
        coefficient = rng.randint(2, 6)
        degree = rng.randint(1, 3)
        constant = rng.randint(1, 9)
        return Item(
            concept.id,
            variant,
            f"What is the degree of {coefficient}x^{degree} + {constant}?",
            (_step("What is the largest exponent on the variable?", str(degree)),),
        )
    a = rng.randint(2, 6)
    b = rng.randint(2, 6)
    c = rng.randint(1, 8)
    return Item(
        concept.id,
        variant,
        f"Simplify {a}x + {b}x - {c}.",
        (
            _step(
                "After combining like terms, what is the simplified expression?",
                f"{a + b}x - {c}",
                (
                    "Add the coefficients of the x terms.",
                    "Keep the variable part unchanged.",
                    "What is the simplified expression?",
                ),
                kind="expression",
            ),
        ),
    )


def _distribute(concept: Concept, variant: int, rng: random.Random) -> Item:
    a = rng.randint(2, 6)
    b = rng.randint(1, 9)
    return Item(
        concept.id,
        variant,
        f"Expand {a}(x + {b}).",
        (
            _step(
                "Which two products appear after distributing?",
                f"{a}x + {a * b}",
                (
                    "Multiply the outside factor by each term inside.",
                    "Do not forget the constant term.",
                    "What is the expanded expression?",
                ),
                kind="expression",
            ),
        ),
    )


def _simplify_expression(concept: Concept, variant: int, rng: random.Random) -> Item:
    a = rng.randint(2, 4)
    b = rng.randint(1, 5)
    c = rng.randint(2, 6)
    d = rng.randint(1, 8)
    return Item(
        concept.id,
        variant,
        f"Simplify {a}(x + {b}) + {c}x + {d}.",
        (
            _step(
                "After distributing, which x terms can be combined?",
                f"{a + c}x + {a * b + d}",
                (
                    "Distribute first.",
                    "Combine only like terms.",
                    "What is the simplified expression?",
                ),
                kind="expression",
            ),
        ),
    )


def _translate_expression(concept: Concept, variant: int, rng: random.Random) -> Item:
    number = rng.randint(2, 9)
    phrase_options = (
        (f"{number} more than twice a number x", f"2x + {number}"),
        (f"{number} less than a number x", f"x - {number}"),
        (f"the product of {number} and the sum of x and 2", f"{number}(x + 2)"),
        (f"three times a number x, plus {number}", f"3x + {number}"),
    )
    phrase, answer = phrase_options[variant % len(phrase_options)]
    return Item(
        concept.id,
        variant,
        f"Write an expression for: {phrase}.",
        (
            _step(
                "Which operation does each phrase signal?",
                answer,
                (
                    "Identify the unknown and the operation words.",
                    "Build the expression piece by piece.",
                    "What expression matches the phrase?",
                ),
                kind="expression",
            ),
        ),
    )


def _one_step(concept: Concept, variant: int, rng: random.Random) -> Item:
    if concept.family_mode == "addsub":
        a = rng.randint(2, 9)
        x = rng.randint(2, 12)
        b = x + a
        return Item(
            concept.id,
            variant,
            f"Solve x + {a} = {b}.",
            (
                _step(
                    "What number would you subtract from both sides?",
                    str(a),
                    _generic_hints("constant"),
                ),
                _step("What is x?", str(x)),
            ),
        )
    a = rng.randint(2, 6)
    x = rng.randint(2, 9)
    b = a * x
    if variant % 2:
        return Item(
            concept.id,
            variant,
            f"Solve x/{a} = {x}.",
            (
                _step(
                    "What number would you multiply both sides by?",
                    str(a),
                    _generic_hints("denominator"),
                ),
                _step("What is x?", str(b)),
            ),
        )
    return Item(
        concept.id,
        variant,
        f"Solve {a}x = {b}.",
        (
            _step(
                "What number would you divide both sides by?",
                str(a),
                _generic_hints("coefficient"),
            ),
            _step("What is x?", str(x)),
        ),
    )


def _two_step(concept: Concept, variant: int, rng: random.Random) -> Item:
    a = rng.randint(2, 5)
    x = rng.randint(2, 9)
    b = rng.randint(1, 9)
    c = a * x + b
    return Item(
        concept.id,
        variant,
        f"Solve {a}x + {b} = {c}.",
        (
            _step(
                "What number would you subtract from both sides to start?",
                str(b),
                _generic_hints("constant"),
            ),
            _step(
                f"After subtracting {b}, what is {a}x equal to?",
                str(a * x),
            ),
            _step("What is x?", str(x)),
        ),
    )


def _both_sides(concept: Concept, variant: int, rng: random.Random) -> Item:
    a = rng.randint(4, 8)
    c = rng.randint(1, a - 2)
    x = rng.randint(2, 8)
    b = rng.randint(-6, 6)
    d = (a - c) * x + b
    return Item(
        concept.id,
        variant,
        f"Solve {a}x + ({b}) = {c}x + ({d}).",
        (
            _step(
                "How many x terms remain after moving the x term on the right to the left?",
                str(a - c),
                _generic_hints("x coefficient"),
            ),
            _step(
                "After moving the x terms left and constants right, what constant is on the right?",
                str(d - b),
                _generic_hints("constant"),
            ),
            _step("What is x?", str(x)),
        ),
    )


def _fraction_equation(concept: Concept, variant: int, rng: random.Random) -> Item:
    a = rng.choice([2, 3, 4])
    x = a * rng.randint(1, 5)
    b = rng.randint(1, 5)
    c = x // a + b
    return Item(
        concept.id,
        variant,
        f"Solve x/{a} + {b} = {c}.",
        (
            _step(
                "What number would you multiply every term by to clear the fraction?",
                str(a),
                _generic_hints("denominator"),
            ),
            _step(
                f"After multiplying, what is x + {a * b} equal to?",
                str(a * c),
            ),
            _step("What is x?", str(x)),
        ),
    )


def _literal_equation(concept: Concept, variant: int, rng: random.Random) -> Item:
    a = rng.randint(2, 5)
    b = rng.randint(2, 5)
    c = rng.randint(3, 9)
    answer = f"({c} - {a}x)/{b}"
    return Item(
        concept.id,
        variant,
        f"Solve {a}x + {b}y = {c} for y.",
        (
            _step(
                "What should you undo first, and how do you isolate y?",
                answer,
                (
                    "Subtract the x term from both sides.",
                    "Divide every term by the coefficient of y.",
                    "What expression equals y?",
                ),
                kind="expression",
            ),
        ),
    )


def _linear_word(concept: Concept, variant: int, rng: random.Random) -> Item:
    base = rng.choice([2, 3, 4, 5])
    rate = rng.choice([2, 3, 5])
    distance = rng.randint(4, 12)
    return Item(
        concept.id,
        variant,
        f"A taxi charges ${base} plus ${rate} per kilometre. What is the cost for {distance} km?",
        (
            _step(
                "What is the total cost for this trip?",
                str(base + rate * distance),
                (
                    "The base fee is paid once.",
                    "Multiply the rate by the number of kilometres.",
                    "Add the base fee last.",
                ),
            ),
        ),
    )


def _inequality(concept: Concept, variant: int, rng: random.Random) -> Item:
    if concept.family_mode == "absolute":
        h = rng.randint(-3, 3)
        d = rng.randint(1, 5)
        answer = "{" + f"{h - d}, {h + d}" + "}"
        return Item(
            concept.id,
            variant,
            f"Solve |x - ({h})| = {d}.",
            (
                _step(
                    "What two distances from the centre give this absolute value?",
                    answer,
                    (
                        "Split the absolute value into two cases.",
                        "The inside expression can equal d or -d.",
                        "What two values of x result?",
                    ),
                    kind="set",
                ),
            ),
        )
    if concept.family_mode == "compound":
        lo = rng.randint(-4, 0)
        width = rng.randint(4, 8)
        a = rng.randint(1, 3)
        b = rng.randint(-3, 3)
        # lo <= a*x+b <= lo+width
        start = Fraction(lo - b, a)
        end = Fraction(lo + width - b, a)
        if start > end:
            start, end = end, start
        answer = f"{_text(start)} <= x <= {_text(end)}"
        return Item(
            concept.id,
            variant,
            f"Solve {lo} <= {a}x + ({b}) <= {lo + width}.",
            (
                _step(
                    "Apply the inverse operations to all three parts; what interval do you get?",
                    answer,
                    (
                        "Subtract the constant from every part first.",
                        "Then divide every part by the coefficient.",
                        "What compound inequality remains?",
                    ),
                    kind="interval",
                ),
            ),
        )
    a = rng.randint(2, 5)
    b = rng.randint(-5, 5)
    c = rng.randint(2, 12)
    answer = f"x {'<' if a > 0 else '>'} {_text(Fraction(c - b, a))}"
    return Item(
        concept.id,
        variant,
        f"Solve {a}x + ({b}) < {c}.",
        (
            _step(
                "After undoing the constant and coefficient, which inequality describes x?",
                answer,
                (
                    "Treat it like an equation first.",
                    "If you divide by a positive number, keep the sign.",
                    "What inequality describes all solutions?",
                ),
                kind="interval",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Graphing and linear relations


def _coordinate(concept: Concept, variant: int, rng: random.Random) -> Item:
    if concept.family_mode == "quadrant":
        x = rng.choice([-4, -3, -2, 2, 3, 4])
        y = rng.choice([-4, -3, -2, 2, 3, 4])
        quadrant = 1 if x > 0 and y > 0 else 2 if x < 0 and y > 0 else 3 if x < 0 else 4
        return Item(
            concept.id,
            variant,
            f"In which quadrant is the point ({x}, {y})?",
            (
                _step(
                    "What are the signs of x and y in that quadrant?",
                    str(quadrant),
                    (
                        "Quadrant I has both coordinates positive.",
                        "The signs identify the quadrant.",
                        "Which quadrant matches these signs?",
                    ),
                ),
            ),
            _visual([(x, y)]),
        )
    if concept.family_mode == "plot":
        x = rng.randint(-4, 4)
        y = rng.randint(-4, 4)
        direction = "right" if x >= 0 else "left"
        vertical = "up" if y >= 0 else "down"
        return Item(
            concept.id,
            variant,
            f"Point A is {abs(x)} units {direction} and {abs(y)} units {vertical} "
            "from the origin. What are its coordinates?",
            (
                _step(
                    "Which ordered pair records the horizontal and vertical moves?",
                    f"({x}, {y})",
                    (
                        "The x-coordinate comes first.",
                        "Right and up are positive directions.",
                        "What ordered pair describes point A?",
                    ),
                    kind="pair",
                ),
            ),
            _visual([(x, y)]),
        )
    points = [(rng.randint(-3, 3), rng.randint(-3, 3)) for _ in range(3)]
    points = list(dict.fromkeys(points))
    while len(points) < 3:
        points.append((rng.randint(-3, 3), rng.randint(-3, 3)))
        points = list(dict.fromkeys(points))
    domain = "{" + ", ".join(str(p[0]) for p in points) + "}"
    range_ = "{" + ", ".join(str(p[1]) for p in points) + "}"
    return Item(
        concept.id,
        variant,
        "A relation contains " + ", ".join(f"({x}, {y})" for x, y in points) + ".",
        (
            _step(
                "Which input values appear in the relation?",
                domain,
                (
                    "The domain is the set of first coordinates.",
                    "List each input once.",
                    "What is the domain set?",
                ),
                kind="set",
            ),
            _step(
                "Which output values appear in the relation?",
                range_,
                (
                    "The range is the set of second coordinates.",
                    "List each output once.",
                    "What is the range set?",
                ),
                kind="set",
            ),
        ),
        _visual(points),
    )


def _slope_item(concept: Concept, variant: int, rng: random.Random) -> Item:
    x1, y1 = rng.randint(-4, 3), rng.randint(-4, 4)
    x2 = x1 + rng.randint(1, 4)
    y2 = y1 + rng.randint(-4, 4)
    while y2 == y1:
        y2 = y1 + rng.randint(-4, 4)
    answer = Fraction(y2 - y1, x2 - x1)
    return Item(
        concept.id,
        variant,
        f"What is the slope of the line through ({x1}, {y1}) and ({x2}, {y2})?",
        (
            _step(
                "What is the change in y divided by the change in x?",
                _text(answer),
                (
                    "Subtract the y-values for the rise.",
                    "Subtract the x-values for the run.",
                    "Write rise over run and simplify.",
                ),
                kind="expression",
            ),
        ),
        _visual([(x1, y1), (x2, y2)]),
    )


def _line_equation(concept: Concept, variant: int, rng: random.Random) -> Item:
    m = rng.choice([-3, -2, -1, 1, 2, 3])
    b = rng.randint(-6, 6)
    if concept.family_mode == "slope_intercept":
        return Item(
            concept.id,
            variant,
            f"Find the slope and y-intercept of y = {m}x + ({b}).",
            (
                _step("Which number multiplies x?", str(m), _generic_hints("x coefficient")),
                _step("What is the y-intercept?", str(b), _generic_hints("constant")),
            ),
        )
    if concept.family_mode == "graph":
        x = rng.randint(-3, 3)
        answer = m * x + b
        return Item(
            concept.id,
            variant,
            f"For y = {m}x + ({b}), what is y when x = {x}?",
            (
                _step(
                    "Substitute x into the equation; what is y?",
                    str(answer),
                    (
                        "Replace x with the given value.",
                        "Multiply before adding the constant.",
                        "What is the y-value?",
                    ),
                ),
            ),
            _visual([(x, answer)]) if -5 <= x <= 5 and -5 <= answer <= 5 else None,
        )
    if concept.family_mode == "point_slope":
        x1, y1 = rng.randint(-3, 3), rng.randint(-5, 5)
        x2, y2 = x1 + 1, y1 + m
        slope_intercept_form = f"y = {m}x + ({y2 - m * x2})"
        return Item(
            concept.id,
            variant,
            f"Write the equation of the line with slope {m} through ({x1}, {y1}).",
            (
                _step(
                    "Use y - y1 = m(x - x1); what is the slope-intercept form?",
                    slope_intercept_form,
                    (
                        "Start with point-slope form.",
                        "Distribute the slope.",
                        "Solve for y to get slope-intercept form.",
                    ),
                    kind="expression",
                ),
            ),
        )
    if concept.family_mode == "parallel_perp":
        perpendicular = Fraction(-1, m)
        return Item(
            concept.id,
            variant,
            f"A line has slope {_text(Fraction(m))}. What is the slope of a perpendicular line?",
            (
                _step(
                    "What is the negative reciprocal of the given slope?",
                    _text(perpendicular),
                    (
                        "Flip the fraction and change the sign.",
                        "Perpendicular slopes multiply to -1.",
                        "What is the perpendicular slope?",
                    ),
                    kind="expression",
                ),
            ),
        )
    a = rng.randint(1, 4)
    c = rng.randint(2, 12)
    x_intercept = Fraction(c, a)
    return Item(
        concept.id,
        variant,
        f"For {a}x + {rng.randint(1, 4)}y = {c}, what is the x-intercept?",
        (
            _step(
                "What value of y gives the x-intercept?",
                _text(x_intercept),
                (
                    "Set y equal to zero.",
                    "Solve the remaining equation for x.",
                    "What is the x-intercept?",
                ),
                kind="expression",
            ),
        ),
    )


def _systems(concept: Concept, variant: int, rng: random.Random) -> Item:
    x = rng.randint(-4, 5)
    y = rng.randint(-4, 5)
    a, b = rng.randint(1, 4), rng.randint(-3, 3)
    while b == 0:
        b = rng.randint(-3, 3)
    c, d = rng.randint(-3, 3), rng.randint(1, 4)
    if a * d == b * c:
        d += 1
    e = a * x + b * y
    f = c * x + d * y
    method = "substitution" if concept.family_mode == "substitution" else "elimination"
    return Item(
        concept.id,
        variant,
        f"Solve the system {a}x + ({b})y = {e} and {c}x + ({d})y = {f}.",
        (
            _step(
                f"Using {method}, what is x?",
                str(x),
                (
                    "Choose a variable to eliminate or isolate.",
                    "Substitute or add the equations to get one variable.",
                    "What is the x-coordinate of the solution?",
                ),
            ),
            _step("What is y?", str(y), _generic_hints("remaining variable")),
        ),
    )


# ---------------------------------------------------------------------------
# Polynomials, factoring, quadratics


def _polynomial_ops(concept: Concept, variant: int, rng: random.Random) -> Item:
    if concept.family_mode == "addsub":
        a, c = rng.randint(1, 5), rng.randint(1, 5)
        b, d = rng.randint(-5, 5), rng.randint(-5, 5)
        answer = f"{a + c}x^2 + {b + d}x"
        return Item(
            concept.id,
            variant,
            f"Simplify ({a}x^2 + {b}x) + ({c}x^2 + {d}x).",
            (
                _step(
                    "Which terms can be combined?",
                    answer,
                    (
                        "Add matching powers of x.",
                        "Keep x^2 and x terms separate.",
                        "What is the simplified polynomial?",
                    ),
                    kind="expression",
                ),
            ),
        )
    if concept.family_mode == "monomial":
        a, b = rng.randint(2, 5), rng.randint(2, 5)
        m, n = rng.randint(1, 4), rng.randint(1, 4)
        answer = f"{a * b}x^{m + n}"
        return Item(
            concept.id,
            variant,
            f"Multiply ({a}x^{m})({b}x^{n}).",
            (
                _step(
                    "What happens to the coefficients and the exponents?",
                    answer,
                    (
                        "Multiply coefficients.",
                        "Add exponents of the same base.",
                        "What is the product?",
                    ),
                    kind="expression",
                ),
            ),
        )
    if concept.family_mode == "special":
        a = rng.randint(1, 5)
        choice = variant % 2
        if choice:
            answer = f"x^2 - {a * a}"
            return Item(
                concept.id,
                variant,
                f"Multiply (x + {a})(x - {a}).",
                (
                    _step(
                        "Which special product pattern applies?",
                        answer,
                        (
                            "The middle terms cancel.",
                            "This is a difference of squares.",
                            "What is the simplified expression?",
                        ),
                        kind="expression",
                    ),
                ),
            )
        answer = f"x^2 + {2 * a}x + {a * a}"
        return Item(
            concept.id,
            variant,
            f"Expand (x + {a})^2.",
            (
                _step(
                    "Which terms appear when the binomial is squared?",
                    answer,
                    (
                        "Square the first term.",
                        "Double the product of the two terms.",
                        "Square the last term.",
                    ),
                    kind="expression",
                ),
            ),
        )
    a, b = rng.randint(1, 4), rng.randint(1, 4)
    answer = f"x^2 + {a + b}x + {a * b}"
    return Item(
        concept.id,
        variant,
        f"Expand (x + {a})(x + {b}).",
        (
            _step(
                "Which four products appear before combining?",
                answer,
                (
                    "Multiply each term in the first binomial by each term in the second.",
                    "The two middle terms are like terms.",
                    "What is the expanded polynomial?",
                ),
                kind="expression",
            ),
        ),
    )


def _factoring(concept: Concept, variant: int, rng: random.Random) -> Item:
    if concept.family_mode == "gcf":
        factor = rng.randint(2, 5)
        inner = rng.randint(1, 5)
        first = rng.randint(1, 4)
        answer = f"{factor}x({first}x + {inner})"
        return Item(
            concept.id,
            variant,
            f"Factor {factor * first}x^2 + {factor * inner}x.",
            (
                _step(
                    "What is the greatest common factor of the two terms?",
                    f"{factor}x",
                    (
                        "Find the largest common number.",
                        "Include the shared x factor.",
                        "What factor is common to both terms?",
                    ),
                    kind="expression",
                ),
                _step(
                    "Which factored form uses that common factor?",
                    answer,
                    (
                        "Divide each term by the common factor.",
                        "Write the common factor outside parentheses.",
                        "What is the factored form?",
                    ),
                    kind="expression",
                ),
            ),
        )
    if concept.family_mode == "difference_squares":
        a = rng.randint(2, 6)
        answer = f"(x - {a})(x + {a})"
        return Item(
            concept.id,
            variant,
            f"Factor x^2 - {a * a}.",
            (
                _step(
                    "Which pair of conjugates multiplies to this difference?",
                    answer,
                    (
                        "Find the square root of each term.",
                        "Use (a - b)(a + b).",
                        "What is the factored form?",
                    ),
                    kind="expression",
                ),
            ),
        )
    if concept.family_mode == "mixed":
        a = rng.randint(1, 4)
        b = rng.randint(1, 4)
        answer = f"{a}(x + {b})(x - {b})" if variant % 2 else f"(x + {a})(x + {b})"
        if variant % 2:
            prompt = f"Factor {a}x^2 - {a * b * b}."
        else:
            prompt = f"Factor x^2 + {a + b}x + {a * b}."
        return Item(
            concept.id,
            variant,
            prompt,
            (
                _step(
                    "What pattern or factor pair applies?",
                    answer,
                    (
                        "Try the greatest common factor first.",
                        "For a trinomial, find two numbers with the right sum and product.",
                        "What is the fully factored form?",
                    ),
                    kind="expression",
                ),
            ),
        )
    a = rng.randint(1, 5)
    b = rng.randint(1, 5)
    return Item(
        concept.id,
        variant,
        f"Factor x^2 + {a + b}x + {a * b}.",
        (
            _step(
                "Which two integers multiply to the constant and add to the middle coefficient?",
                f"({a}, {b})",
                (
                    "List factor pairs of the constant.",
                    "Check which pair adds to the middle coefficient.",
                    "Which pair works?",
                ),
                kind="set",
            ),
            _step(
                "What is the factored form?",
                f"(x + {a})(x + {b})",
                (
                    "Use the two numbers as constants in the binomials.",
                    "Both signs are positive here.",
                    "What is the factored expression?",
                ),
                kind="expression",
            ),
        ),
    )


def _quadratic(concept: Concept, variant: int, rng: random.Random) -> Item:
    if concept.family_mode == "zero_product":
        a = rng.randint(1, 5)
        b = rng.randint(1, 5)
        return Item(
            concept.id,
            variant,
            f"Solve (x - {a})(x + {b}) = 0.",
            (
                _step(
                    "Which values make either factor zero?",
                    "{" + f"{a}, {-b}" + "}",
                    (
                        "Set each factor equal to zero.",
                        "Solve each mini-equation.",
                        "What is the solution set?",
                    ),
                    kind="set",
                ),
            ),
        )
    if concept.family_mode == "sqrt_property":
        a = rng.randint(2, 8)
        return Item(
            concept.id,
            variant,
            f"Solve x^2 = {a * a}.",
            (
                _step(
                    "Which two values have this square?",
                    "{" + f"{a}, {-a}" + "}",
                    (
                        "Take the square root of both sides.",
                        "Do not forget the negative root.",
                        "What is the solution set?",
                    ),
                    kind="set",
                ),
            ),
        )
    if concept.family_mode == "formula":
        r1 = rng.randint(1, 4)
        r2 = rng.randint(1, 4)
        while r2 == r1:
            r2 = rng.randint(1, 4)
        b = -(r1 + r2)
        c = r1 * r2
        disc = b * b - 4 * c
        return Item(
            concept.id,
            variant,
            f"Solve x^2 + ({b})x + ({c}) = 0 using the quadratic formula.",
            (
                _step(
                    "What is the discriminant b^2 - 4ac?",
                    str(disc),
                    (
                        "Identify a, b, and c first.",
                        "Square b and subtract 4ac.",
                        "What is the discriminant?",
                    ),
                ),
                _step(
                    "What are the two solutions?",
                    "{" + f"{r1}, {r2}" + "}",
                    (
                        "Substitute into the quadratic formula.",
                        "Simplify the square root.",
                        "What are the two roots?",
                    ),
                    kind="set",
                ),
            ),
        )
    a = rng.randint(1, 5)
    b = rng.randint(1, 5)
    return Item(
        concept.id,
        variant,
        f"Solve x^2 + {a + b}x + {a * b} = 0.",
        (
            _step(
                "Which two roots come from the factored form?",
                "{" + f"{-a}, {-b}" + "}",
                (
                    "Find two numbers that multiply to the constant.",
                    "The sum must match the middle coefficient.",
                    "What are the roots?",
                ),
                kind="set",
            ),
        ),
    )


def _rational(concept: Concept, variant: int, rng: random.Random) -> Item:
    a = rng.randint(1, 4)
    x = a + rng.randint(1, 4)
    answer = x + a
    return Item(
        concept.id,
        variant,
        f"Evaluate (x^2 - {a * a})/(x - {a}) when x = {x}.",
        (
            _step(
                "Factor the numerator first; what value do you get after cancelling?",
                str(answer),
                (
                    "The numerator is a difference of squares.",
                    "Cancel the common factor before substituting.",
                    "What number remains?",
                ),
            ),
        ),
    )


def _functions(concept: Concept, variant: int, rng: random.Random) -> Item:
    if concept.family_mode == "identify":
        points = [(1, 2), (2, 3), (3, 4)]
        if variant % 3 == 2:
            points = [(1, 2), (1, 3), (2, 4)]
        answer = "yes" if len({p[0] for p in points}) == len(points) else "no"
        return Item(
            concept.id,
            variant,
            "Is the relation " + ", ".join(f"({x}, {y})" for x, y in points) + " a function?",
            (
                _step(
                    "Does every input have exactly one output?",
                    answer,
                    (
                        "Check whether any input repeats.",
                        "A repeated input with different outputs breaks the function rule.",
                        "Is it a function?",
                    ),
                    kind="boolean",
                ),
            ),
            _visual(points),
        )
    if concept.family_mode == "quadratic_vertex":
        h = rng.randint(-3, 4)
        k = rng.randint(-5, 5)
        return Item(
            concept.id,
            variant,
            f"What is the vertex of y = x^2 {_signed(-2 * h)}x {_signed(h * h + k)}?",
            (
                _step(
                    "What is the x-coordinate of the vertex?",
                    str(h),
                    (
                        "For y = x^2 + bx + c, the vertex x-coordinate is -b/2.",
                        "Substitute the x-coordinate into the equation.",
                        "What is the vertex?",
                    ),
                ),
                _step("What is the vertex as an ordered pair?", f"({h}, {k})", kind="pair"),
            ),
        )
    if concept.family_mode == "exponential":
        a = rng.randint(2, 4)
        b = rng.randint(2, 3)
        x = rng.randint(1, 3)
        return Item(
            concept.id,
            variant,
            f"For f(x) = {a} * {b}^x, what is f({x})?",
            (
                _step(
                    "Evaluate the power first, then multiply; what is f(x)?",
                    str(a * b**x),
                    (
                        "Substitute the input for x.",
                        "Apply the exponent before multiplying.",
                        "What is the output?",
                    ),
                ),
            ),
        )
    if concept.family_mode == "sequence":
        start = rng.randint(1, 6)
        diff = rng.randint(2, 5)
        terms = [start + diff * i for i in range(4)]
        return Item(
            concept.id,
            variant,
            "What is the next term in the arithmetic sequence " + ", ".join(map(str, terms)) + "?",
            (
                _step(
                    "What is the next term?",
                    str(terms[-1] + diff),
                    (
                        "Subtract consecutive terms.",
                        "The same difference repeats.",
                        "What comes next?",
                    ),
                ),
            ),
        )
    if concept.family_mode == "linear_model":
        base = rng.randint(2, 5)
        rate = rng.randint(2, 5)
        return Item(
            concept.id,
            variant,
            f"A gym charges ${base} to join and ${rate} per visit. Write the cost for v visits.",
            (
                _step(
                    "Which expression combines the one-time fee and the per-visit fee?",
                    f"{base} + {rate}v",
                    (
                        "The joining fee is paid once.",
                        "The visit fee multiplies the number of visits.",
                        "What expression gives the total cost?",
                    ),
                    kind="expression",
                ),
            ),
        )
    a = rng.randint(2, 5)
    b = rng.randint(1, 6)
    x = rng.randint(2, 6)
    return Item(
        concept.id,
        variant,
        f"For f(x) = {a}x + {b}, what is f({x})?",
        (
            _step(
                "Substitute x and evaluate; what is the output?",
                str(a * x + b),
                (
                    "Replace x with the input.",
                    "Multiply before adding.",
                    "What is f(x)?",
                ),
            ),
        ),
    )
