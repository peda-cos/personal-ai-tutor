from __future__ import annotations

from pathlib import Path

import pytest

from personal_ai_tutor.study import (
    STUDY_CONCEPTS,
    _study_items,
    bootstrap_mean_ci,
    compare_gains,
    ingest_csv,
    run_study,
)


def test_bootstrap_and_known_gain_comparison() -> None:
    low, high = bootstrap_mean_ci([1, 2, 3, 4], seed=7, resamples=1000)
    assert low <= high
    result = compare_gains([5, 6, 7], [1, 2, 3])
    assert result.delta > 0
    assert result.delta_ci95[0] <= result.delta <= result.delta_ci95[1]
    assert result.cohens_d > 1.0


def test_study_items_shape() -> None:
    items = _study_items()
    assert len(items) == 30
    assert set(items).issubset(set(STUDY_CONCEPTS))
    assert "two_step_equations" in items


def test_simulated_study_runs_and_is_labeled() -> None:
    result = run_study(n_per_arm=2, seed=20261123)
    assert result.simulated
    assert result.learners_per_arm == 2
    assert result.pre_post_items == 30
    assert 0 <= result.adaptive.pre_mean <= 30
    assert 0 <= result.baseline.post_mean <= 30
    assert result.adaptive.gain == pytest.approx(
        result.adaptive.post_mean - result.adaptive.pre_mean
    )
    assert result.delta_ci95[0] <= result.delta <= result.delta_ci95[1]
    assert "simulated" in result.protocol


def test_csv_ingestion(tmp_path: Path) -> None:
    path = tmp_path / "study.csv"
    path.write_text(
        "learner_id,pre,post,arm\n"
        "a1,10,20,adaptive\n"
        "a2,12,21,adaptive\n"
        "b1,9,12,baseline\n"
        "b2,11,13,baseline\n",
        encoding="utf-8",
    )
    result = ingest_csv(path)
    assert not result.simulated
    assert result.adaptive.n == 2
    assert result.delta == pytest.approx(7.0)
    with pytest.raises(ValueError):
        bad = tmp_path / "bad.csv"
        bad.write_text("learner_id,pre,post\nx,1,2\n", encoding="utf-8")
        ingest_csv(bad)
