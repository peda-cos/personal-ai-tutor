"""Classic four-parameter Bayesian knowledge tracing."""

from __future__ import annotations

from dataclasses import dataclass

from .curriculum import Concept


@dataclass(frozen=True)
class BktParams:
    p_init: float
    p_learn: float
    p_slip: float
    p_guess: float

    def __post_init__(self) -> None:
        for name, value in (
            ("p_init", self.p_init),
            ("p_learn", self.p_learn),
            ("p_slip", self.p_slip),
            ("p_guess", self.p_guess),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.p_slip + self.p_guess >= 1.0:
            raise ValueError("p_slip + p_guess must be less than 1")


def default_params(concept: Concept) -> BktParams:
    return BktParams(
        p_init=min(0.45, max(0.15, 0.20 + 0.04 * concept.difficulty)),
        p_learn=0.12,
        p_slip=0.10,
        p_guess=0.20,
    )


def update(p_known: float, correct: bool, params: BktParams) -> float:
    """Posterior knowledge probability after an item, then the learning transition."""
    p = min(0.999, max(0.001, p_known))
    if correct:
        numerator = p * (1.0 - params.p_slip)
        denominator = numerator + (1.0 - p) * params.p_guess
    else:
        numerator = p * params.p_slip
        denominator = numerator + (1.0 - p) * (1.0 - params.p_guess)
    if denominator <= 0.0:
        raise ValueError("degenerate BKT denominator")
    posterior = numerator / denominator
    result = posterior + (1.0 - posterior) * params.p_learn
    return min(0.999, max(0.001, result))


def predict_correct(p_known: float, params: BktParams) -> float:
    p = min(1.0, max(0.0, p_known))
    return p * (1.0 - params.p_slip) + (1.0 - p) * params.p_guess
