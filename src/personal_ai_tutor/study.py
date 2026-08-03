"""Simulated two-week efficacy study and real CSV ingestion statistics.

The simulation is intentionally labeled simulated. It runs the same policy,
BKT, FSRS, safety, and SQLite engine used by the app; it is not evidence of
human learning gains.
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .bkt import BktParams, predict_correct, update
from .config import Settings
from .db import FrozenClock
from .engine import TutorEngine

STUDY_CONCEPTS: tuple[str, ...] = (
    "number_line",
    "integers_compare",
    "absolute_value",
    "integer_add_sub",
    "integer_mul_div",
    "fractions_equiv",
    "fraction_add_sub",
    "fraction_mul_div",
    "decimals_percents",
    "order_of_operations",
    "variables_expressions",
    "evaluate_expressions",
    "combine_like_terms",
    "distributive_property",
    "simplify_expressions",
    "translate_words",
    "one_step_add_sub",
    "one_step_mul_div",
    "two_step_equations",
    "variables_both_sides",
    "clearing_fractions",
    "literal_equations",
    "linear_word_problems",
    "solve_inequalities",
    "compound_inequalities",
    "absolute_value_equations",
    "coordinate_plane",
    "plotting_points",
    "slope",
    "slope_intercept",
    "graph_linear_equations",
)

_PRE_POST_COUNT = 30
_SESSION_DAYS = (0, 2, 4, 7, 9, 13)


@dataclass(frozen=True)
class ArmStats:
    n: int
    pre_mean: float
    post_mean: float
    gain: float
    gain_sd: float
    gain_ci95: tuple[float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "pre_mean": self.pre_mean,
            "post_mean": self.post_mean,
            "gain": self.gain,
            "gain_sd": self.gain_sd,
            "gain_ci95": list(self.gain_ci95),
        }


@dataclass(frozen=True)
class StudyResult:
    simulated: bool
    seed: int
    learners_per_arm: int
    sessions: int
    items_per_session: int
    pre_post_items: int
    adaptive: ArmStats
    baseline: ArmStats
    delta: float
    delta_ci95: tuple[float, float]
    cohens_d: float
    protocol: str

    def to_dict(self) -> dict[str, object]:
        return {
            "simulated": self.simulated,
            "seed": self.seed,
            "learners_per_arm": self.learners_per_arm,
            "sessions": self.sessions,
            "items_per_session": self.items_per_session,
            "pre_post_items": self.pre_post_items,
            "adaptive": self.adaptive.to_dict(),
            "baseline": self.baseline.to_dict(),
            "delta": self.delta,
            "delta_ci95": list(self.delta_ci95),
            "cohens_d": self.cohens_d,
            "protocol": self.protocol,
        }


def bootstrap_mean_ci(
    values: Sequence[float], seed: int, resamples: int = 10_000
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    count = len(values)
    for _ in range(resamples):
        sample = [values[rng.randrange(count)] for _ in range(count)]
        means.append(sum(sample) / count)
    means.sort()
    return (means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1])


def bootstrap_delta_ci(
    adaptive: Sequence[float], baseline: Sequence[float], seed: int, resamples: int = 10_000
) -> tuple[float, float]:
    if not adaptive or not baseline:
        return (0.0, 0.0)
    rng = random.Random(seed)
    deltas = []
    for _ in range(resamples):
        left = [adaptive[rng.randrange(len(adaptive))] for _ in adaptive]
        right = [baseline[rng.randrange(len(baseline))] for _ in baseline]
        deltas.append(sum(left) / len(left) - sum(right) / len(right))
    deltas.sort()
    return (deltas[int(0.025 * resamples)], deltas[int(0.975 * resamples) - 1])


def cohens_d(adaptive: Sequence[float], baseline: Sequence[float]) -> float:
    if len(adaptive) < 2 or len(baseline) < 2:
        return 0.0
    pooled = (
        (len(adaptive) - 1) * statistics.variance(adaptive)
        + (len(baseline) - 1) * statistics.variance(baseline)
    ) / (len(adaptive) + len(baseline) - 2)
    if pooled <= 0:
        return 0.0
    return (statistics.mean(adaptive) - statistics.mean(baseline)) / math.sqrt(pooled)


def summarize_arm(
    gains: Sequence[float],
    pre: Sequence[float],
    post: Sequence[float],
    seed: int,
) -> ArmStats:
    return ArmStats(
        n=len(gains),
        pre_mean=statistics.mean(pre) if pre else 0.0,
        post_mean=statistics.mean(post) if post else 0.0,
        gain=statistics.mean(gains) if gains else 0.0,
        gain_sd=statistics.stdev(gains) if len(gains) > 1 else 0.0,
        gain_ci95=bootstrap_mean_ci(gains, seed),
    )


def compare_gains(
    adaptive_gains: Sequence[float],
    baseline_gains: Sequence[float],
    adaptive_pre: Sequence[float] | None = None,
    adaptive_post: Sequence[float] | None = None,
    baseline_pre: Sequence[float] | None = None,
    baseline_post: Sequence[float] | None = None,
    seed: int = 20261123,
) -> StudyResult:
    adaptive = summarize_arm(
        adaptive_gains,
        adaptive_pre or [0.0] * len(adaptive_gains),
        adaptive_post or [0.0] * len(adaptive_gains),
        seed,
    )
    baseline = summarize_arm(
        baseline_gains,
        baseline_pre or [0.0] * len(baseline_gains),
        baseline_post or [0.0] * len(baseline_gains),
        seed + 1,
    )
    delta = adaptive.gain - baseline.gain
    return StudyResult(
        simulated=False,
        seed=seed,
        learners_per_arm=len(adaptive_gains),
        sessions=0,
        items_per_session=0,
        pre_post_items=0,
        adaptive=adaptive,
        baseline=baseline,
        delta=delta,
        delta_ci95=bootstrap_delta_ci(adaptive_gains, baseline_gains, seed + 2),
        cohens_d=cohens_d(adaptive_gains, baseline_gains),
        protocol="ingested CSV",
    )


def run_study(
    n_per_arm: int = 10,
    seed: int = 20261123,
    sessions: int = 6,
    items_per_session: int = 5,
) -> StudyResult:
    if n_per_arm < 2:
        raise ValueError("n_per_arm must be at least 2")
    if sessions != len(_SESSION_DAYS):
        raise ValueError(f"sessions must be {len(_SESSION_DAYS)}")
    items = _study_items()
    adaptive_runs = _run_arm("adaptive", n_per_arm, seed, items, sessions, items_per_session)
    baseline_runs = _run_arm("linear", n_per_arm, seed, items, sessions, items_per_session)
    adaptive_gains = [run["gain"] for run in adaptive_runs]
    baseline_gains = [run["gain"] for run in baseline_runs]
    adaptive = summarize_arm(
        adaptive_gains,
        [run["pre"] for run in adaptive_runs],
        [run["post"] for run in adaptive_runs],
        seed,
    )
    baseline = summarize_arm(
        baseline_gains,
        [run["pre"] for run in baseline_runs],
        [run["post"] for run in baseline_runs],
        seed + 1,
    )
    return StudyResult(
        simulated=True,
        seed=seed,
        learners_per_arm=n_per_arm,
        sessions=sessions,
        items_per_session=items_per_session,
        pre_post_items=len(items),
        adaptive=adaptive,
        baseline=baseline,
        delta=adaptive.gain - baseline.gain,
        delta_ci95=bootstrap_delta_ci(adaptive_gains, baseline_gains, seed + 2),
        cohens_d=cohens_d(adaptive_gains, baseline_gains),
        protocol=(
            f"{n_per_arm} simulated learners per arm; 30-item pre/post; "
            f"{sessions} sessions x {items_per_session} solved items "
            "over 14 simulated days; paired pre/post draws; adaptive BKT selection "
            "vs fixed topological order"
        ),
    )


def _study_items() -> list[str]:
    rng = random.Random("study-items-v1")
    concepts = rng.sample(list(STUDY_CONCEPTS), _PRE_POST_COUNT)
    return sorted(concepts, key=STUDY_CONCEPTS.index)


def _learner_params(seed: int, index: int) -> dict[str, BktParams]:
    rng = random.Random(f"{seed}:learner:{index}")
    return {
        cid: BktParams(
            p_init=rng.uniform(0.15, 0.35),
            p_learn=rng.uniform(0.08, 0.16),
            p_slip=0.10,
            p_guess=0.20,
        )
        for cid in STUDY_CONCEPTS
    }


def _run_arm(
    mode: str,
    n: int,
    seed: int,
    items: Sequence[str],
    sessions: int,
    items_per_session: int,
) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    with tempfile.TemporaryDirectory(prefix=f"tutor-study-{mode}-") as tmp:
        db = Path(tmp) / "study.db"
        clock = FrozenClock(datetime(2026, 1, 1, tzinfo=UTC))
        active: dict[str, BktParams] = {}
        engine = TutorEngine(
            Settings(db_path=db, guard="rules", tts="tone", llm="off"),
            selection_mode=mode,
            concept_pool=STUDY_CONCEPTS,
            clock=clock,
            bkt_params_for=lambda cid: active[cid],
        )
        for index in range(n):
            params = _learner_params(seed, index)
            active.clear()
            active.update(params)
            # Pre/post draws are paired across arms; only the tutoring sequence differs.
            pre_rng = random.Random(f"{seed}:pretest:{index}")
            post_rng = random.Random(f"{seed}:posttest:{index}")
            answer_rng = random.Random(f"{seed}:{mode}:{index}")
            credentials = engine.create_learner(f"Simulated {mode} {index + 1}", "13_17", True)
            scope = engine.learner_scope(credentials.learner_token)
            pre = 0
            for concept_id in items:
                probability = predict_correct(params[concept_id].p_init, params[concept_id])
                correct = pre_rng.random() < probability
                pre += int(correct)
                current = params[concept_id].p_init
                updated = update(current, correct, params[concept_id])
                engine.store.set_mastery(scope, concept_id, updated, 1, int(correct))
            for session_index in range(sessions):
                session_day = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(
                    days=_SESSION_DAYS[session_index]
                )
                clock.set(session_day)
                turn = engine.start_session(scope)
                solved = 0
                turns = 0
                boost = False
                while solved < items_per_session and turns < 30:
                    session = engine.store.get_session(scope, turn.session_id)
                    if session is None:
                        break
                    step = session.item.steps[
                        min(session.step_index, len(session.item.steps) - 1)
                    ]
                    mastery = engine.store.mastery_map(scope).get(
                        session.concept_id, params[session.concept_id].p_init
                    )
                    probability = predict_correct(mastery, params[session.concept_id])
                    if boost:
                        probability = min(0.95, probability + 0.15)
                    if answer_rng.random() < probability:
                        answer = step.answer
                    elif answer_rng.random() < 0.05:
                        answer = "I'm stuck"
                    else:
                        answer = "1" if step.answer.strip() != "1" else "2"
                    turn = engine.reply(scope, turn.session_id, answer)
                    turns += 1
                    if turn.solved:
                        solved += 1
                        boost = False
                    elif turn.signal in {"incorrect", "help"}:
                        boost = True
            post = 0
            final_mastery = engine.store.mastery_map(scope)
            for concept_id in items:
                probability = predict_correct(
                    final_mastery.get(concept_id, params[concept_id].p_init),
                    params[concept_id],
                )
                post += int(post_rng.random() < probability)
            results.append(
                {
                    "pre": float(pre),
                    "post": float(post),
                    "gain": float(post - pre),
                }
            )
    return results


def ingest_csv(path: Path) -> StudyResult:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"learner_id", "pre", "post", "arm"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("CSV needs learner_id, pre, post, and arm columns")
        for row in reader:
            arm = row["arm"].strip().lower()
            if arm not in {"adaptive", "baseline"}:
                raise ValueError("arm must be adaptive or baseline")
            try:
                pre = float(row["pre"])
                post = float(row["post"])
            except ValueError as exc:
                raise ValueError("pre and post must be numeric") from exc
            if not 0 <= pre <= 30 or not 0 <= post <= 30:
                raise ValueError("scores must be between 0 and 30")
            rows.append(
                {
                    "id": row["learner_id"].strip(),
                    "arm": arm,
                    "pre": row["pre"],
                    "post": row["post"],
                }
            )
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("learner_id values must be unique")
    adaptive = [float(row["post"]) - float(row["pre"]) for row in rows if row["arm"] == "adaptive"]
    baseline = [float(row["post"]) - float(row["pre"]) for row in rows if row["arm"] == "baseline"]
    if len(adaptive) < 2 or len(baseline) < 2:
        raise ValueError("need at least two learners per arm")
    result = compare_gains(
        adaptive,
        baseline,
        [float(row["pre"]) for row in rows if row["arm"] == "adaptive"],
        [float(row["post"]) for row in rows if row["arm"] == "adaptive"],
        [float(row["pre"]) for row in rows if row["arm"] == "baseline"],
        [float(row["post"]) for row in rows if row["arm"] == "baseline"],
    )
    return result


def write_study(result: StudyResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
