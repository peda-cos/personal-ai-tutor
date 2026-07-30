from __future__ import annotations

import pytest

from personal_ai_tutor.algebra import (
    AnswerParseError,
    check_answer,
    describe_equation,
    parse_equation,
    parse_interval,
    solve_linear,
    solve_quadratic,
)


def test_expression_equivalence() -> None:
    assert check_answer("expression", "2x+6", "2(x+3)")
    assert check_answer("expression", "x^2+2x+1", "(x+1)^2")
    assert check_answer("expression", "y=2x+3", "2x+3=y")
    assert not check_answer("expression", "2x+6", "2x+5")


def test_numbers_and_unicode() -> None:
    assert check_answer("number", "3/4", "0.75")
    assert check_answer("number", "5", "five")
    assert check_answer("number", "-7", "−7")
    assert check_answer("number", "2", "x = 2")
    assert not check_answer("number", "2", "3")


def test_linear_and_quadratic_solving() -> None:
    lhs, rhs = parse_equation("3x+6=12")
    assert solve_linear(lhs, rhs) == 2
    lhs, rhs = parse_equation("5x-3=2x+9")
    assert solve_linear(lhs, rhs) == 4
    lhs, rhs = parse_equation("x^2-5x+6=0")
    assert set(solve_quadratic(lhs, rhs)) == {2.0, 3.0}


def test_equation_classification() -> None:
    assert describe_equation("3x+6=12") == "two_step_equations"
    assert describe_equation("5x-3=2x+9") == "variables_both_sides"
    assert describe_equation("x+7=12") == "one_step_add_sub"
    assert describe_equation("4x=28") == "one_step_mul_div"


def test_pairs_sets_and_intervals() -> None:
    assert check_answer("pair", "(2, 3)", "x=2, y=3")
    assert not check_answer("pair", "(2, 3)", "(3, 2)")
    assert check_answer("set", "{2, -3}", "x = 2 or x = -3")
    assert not check_answer("set", "{2, -3}", "2")
    assert check_answer("interval", "x < 2", "2 > x")
    assert check_answer("interval", "-1 <= x < 2", "-1 <= x and x < 2")
    assert not check_answer("interval", "-1 <= x < 2", "-1 < x < 2")


def test_choice_and_boolean() -> None:
    options = ("3x+12", "3x+4", "x+12")
    assert check_answer("choice", "3x+12", "A", options)
    assert check_answer("choice", "3x+12", "3x+12", options)
    assert not check_answer("choice", "3x+12", "B", options)
    assert check_answer("boolean", "yes", "true")
    assert not check_answer("boolean", "yes", "no")


def test_invalid_and_unsupported_inputs() -> None:
    assert not check_answer("expression", "x", "1/x")
    assert not check_answer("expression", "x", "(((")
    assert not check_answer("number", "2", "not a number")
    with pytest.raises(AnswerParseError):
        parse_equation("x=2=3")
    with pytest.raises(AnswerParseError):
        parse_interval("not an inequality")


def test_interval_parsing_is_normalized() -> None:
    intervals = parse_interval("x < 2 or x >= 5")
    assert len(intervals) == 2
    assert intervals[0].hi is not None and intervals[0].hi_closed is False
    assert intervals[1].lo is not None and intervals[1].lo_closed is True
