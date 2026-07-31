from __future__ import annotations

import pytest

from personal_ai_tutor.bkt import BktParams, default_params, predict_correct, update
from personal_ai_tutor.curriculum import Curriculum


def test_laser_worked_example() -> None:
    params = BktParams(p_init=0.4, p_learn=0.1, p_slip=0.3, p_guess=0.2)
    after_incorrect = update(0.4, False, params)
    assert after_incorrect == pytest.approx(0.28)
    after_correct = update(after_incorrect, True, params)
    assert after_correct == pytest.approx(0.6188235294117648)


def test_correctness_direction_and_bounds() -> None:
    params = BktParams(p_init=0.4, p_learn=0.1, p_slip=0.1, p_guess=0.2)
    correct = update(0.4, True, params)
    incorrect = update(0.4, False, params)
    assert 0.0 < incorrect < correct < 1.0
    assert 0.001 <= update(0.999, False, params) <= 0.999
    assert predict_correct(0.0, params) == pytest.approx(0.2)
    assert predict_correct(1.0, params) == pytest.approx(0.9)


def test_default_params_are_concept_sensitive(curriculum: Curriculum) -> None:
    easy = default_params(curriculum.concepts["number_line"])
    hard = default_params(curriculum.concepts["quadratic_formula"])
    assert easy.p_init < hard.p_init
    assert easy.p_learn == hard.p_learn


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        BktParams(p_init=1.2, p_learn=0.1, p_slip=0.1, p_guess=0.2)
    with pytest.raises(ValueError):
        BktParams(p_init=0.2, p_learn=0.1, p_slip=0.6, p_guess=0.5)
