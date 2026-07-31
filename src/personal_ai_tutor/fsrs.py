"""FSRS-6 spaced repetition.

The equations and default weights follow the official FSRS-6 description:
https://github.com/open-spaced-repetition/awesome-fsrs/wiki/The-Algorithm

This implementation intentionally has no parameter optimizer; the defaults are
used for every concept. FSRS-6 (rather than FSRS-7) is used because it is the
stable 2024-era algorithm referenced by the capstone and has compact, auditable
DSR equations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

W = (
    0.212,
    1.2931,
    2.3065,
    8.2956,
    6.4133,
    0.8334,
    3.0194,
    0.001,
    1.8722,
    0.1666,
    0.796,
    1.4835,
    0.0614,
    0.2629,
    1.6483,
    0.6014,
    1.8729,
    0.5425,
    0.0912,
    0.0658,
    0.1542,
)
DECAY = -W[20]
FACTOR = 0.9 ** (1.0 / DECAY) - 1.0


@dataclass(frozen=True)
class ReviewState:
    stability: float
    difficulty: float
    due_at: datetime
    last_review: datetime
    reps: int
    lapses: int


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def retrievability(stability_days: float, elapsed_days: float) -> float:
    stability = max(1e-6, stability_days)
    elapsed = max(0.0, elapsed_days)
    return float((1.0 + FACTOR * elapsed / stability) ** DECAY)


def interval_for(requested_retention: float, stability_days: float) -> int:
    retention = _clamp(requested_retention, 0.5, 0.99)
    raw = float(stability_days / FACTOR * (retention ** (1.0 / DECAY) - 1.0))
    return max(1, min(3650, math.ceil(raw)))


def initial_difficulty(grade: int) -> float:
    _check_grade(grade)
    return _clamp(W[4] - math.exp(W[5] * (grade - 1)) + 1.0, 1.0, 10.0)


def _check_grade(grade: int) -> None:
    if grade not in (1, 2, 3, 4):
        raise ValueError("grade must be 1 (again), 2 (hard), 3 (good), or 4 (easy)")


def initial_state(grade: int, now: datetime, requested_retention: float = 0.9) -> ReviewState:
    _check_grade(grade)
    stability = W[grade - 1]
    return ReviewState(
        stability=stability,
        difficulty=initial_difficulty(grade),
        due_at=now + timedelta(days=interval_for(requested_retention, stability)),
        last_review=now,
        reps=1,
        lapses=1 if grade == 1 else 0,
    )


def review(
    state: ReviewState | None,
    grade: int,
    now: datetime,
    requested_retention: float = 0.9,
) -> ReviewState:
    _check_grade(grade)
    if state is None:
        return initial_state(grade, now, requested_retention)

    elapsed = max(0.0, (now - state.last_review).total_seconds() / 86400.0)
    r = retrievability(state.stability, elapsed)
    stability = state.stability
    difficulty = state.difficulty

    if grade == 1:
        failed = (
            W[11]
            * difficulty ** (-W[12])
            * ((stability + 1.0) ** W[13] - 1.0)
            * math.exp((1.0 - r) * W[14])
        )
        stability = min(failed, stability / math.exp(W[17] * W[18]))
        lapses = state.lapses + 1
    elif elapsed < 1.0:
        increase = (
            math.exp(W[17] * (grade - 3 + W[18])) * stability ** (-W[19])
        )
        if grade >= 2:
            increase = max(1.0, increase)
        stability = stability * increase
        lapses = state.lapses
    else:
        hard_penalty = W[15] if grade == 2 else 1.0
        easy_bonus = W[16] if grade == 4 else 1.0
        stability = stability * (
            1.0
            + math.exp(W[8])
            * (11.0 - difficulty)
            * stability ** (-W[9])
            * (math.exp((1.0 - r) * W[10]) - 1.0)
            * hard_penalty
            * easy_bonus
        )
        lapses = state.lapses

    delta_d = -W[6] * (grade - 3)
    difficulty = difficulty + delta_d * (10.0 - difficulty) / 9.0
    difficulty = W[7] * initial_difficulty(4) + (1.0 - W[7]) * difficulty
    difficulty = _clamp(difficulty, 1.0, 10.0)
    stability = max(0.01, stability)
    return ReviewState(
        stability=stability,
        difficulty=difficulty,
        due_at=now + timedelta(days=interval_for(requested_retention, stability)),
        last_review=now,
        reps=state.reps + 1,
        lapses=lapses,
    )
