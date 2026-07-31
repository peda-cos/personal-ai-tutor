from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from personal_ai_tutor.fsrs import (
    ReviewState,
    initial_state,
    interval_for,
    retrievability,
    review,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_retrievability_and_interval() -> None:
    assert retrievability(10, 10) == pytest.approx(0.9)
    assert retrievability(10, 0) == pytest.approx(1.0)
    assert interval_for(0.9, 10) == pytest.approx(10, abs=1)
    # A lower desired retention allows a longer interval before that recall level.
    assert interval_for(0.8, 10) > interval_for(0.9, 10)


def test_grades_increase_stability() -> None:
    state = initial_state(3, NOW)
    assert state.reps == 1
    assert state.due_at > NOW
    delayed = NOW + timedelta(days=4)
    good = review(state, 3, delayed)
    easy = review(state, 4, delayed)
    assert good.stability > state.stability
    assert easy.stability > good.stability
    assert good.reps == 2


def test_lapse_and_same_day() -> None:
    state = initial_state(4, NOW)
    later = NOW + timedelta(days=10)
    again = review(state, 1, later)
    assert again.lapses == 1
    assert again.stability <= state.stability
    same_day = review(state, 3, NOW + timedelta(hours=2))
    assert same_day.stability >= state.stability


def test_difficulty_updates() -> None:
    state = initial_state(2, NOW)
    easier = review(state, 4, NOW + timedelta(days=3))
    harder = review(state, 1, NOW + timedelta(days=3))
    assert easier.difficulty < state.difficulty
    assert harder.difficulty > state.difficulty


def test_state_roundtrip_fields() -> None:
    state = initial_state(3, NOW)
    copy = ReviewState(**state.__dict__)
    assert copy.stability == state.stability
    assert copy.due_at == state.due_at


def test_invalid_grade() -> None:
    with pytest.raises(ValueError):
        initial_state(5, NOW)
