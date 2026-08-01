from __future__ import annotations

from datetime import UTC, datetime

from personal_ai_tutor.db import FrozenClock
from personal_ai_tutor.memory import MemoryService
from personal_ai_tutor.store import Store, create_learner


def _interaction(store, scope, text: str) -> None:
    store.add_interaction(
        scope,
        session_id="s",
        modality="text",
        learner_text=text,
        tutor_text="question",
        concept_id="number_line",
        signal="help",
        correct=False,
        step_index=0,
        hints_used=1,
        mastery_after=None,
    )


def test_retention_boundary_and_semantic_expiry(tmp_path) -> None:
    clock = FrozenClock(datetime(2026, 1, 1, tzinfo=UTC))
    store = Store(tmp_path / "retention.db", clock)
    memory = MemoryService(store)
    credentials = create_learner(store, "Retention", "13_17", True)
    scope = store_scope(store, credentials.learner_token)
    _interaction(store, scope, "old")
    store.upsert_semantic(scope, "preferred_modality", "voice", 0.6)
    clock.advance(days=300)
    _interaction(store, scope, "new")
    counts = store.purge()
    assert counts["interactions"] == 0
    assert store.interaction_count(scope) == 2
    clock.advance(days=66)
    counts = store.purge()
    assert counts["interactions"] == 1
    assert counts["semantic"] == 1
    assert store.interaction_count(scope) == 1
    assert memory.retention_summary()["audit_days"] == 730


def store_scope(store: Store, token: str):
    from personal_ai_tutor.store import authenticate_learner

    return authenticate_learner(store, token)


def test_inactive_learner_is_deleted(tmp_path) -> None:
    clock = FrozenClock(datetime(2026, 1, 1, tzinfo=UTC))
    store = Store(tmp_path / "inactive.db", clock)
    credentials = create_learner(store, "Idle", "13_17", True)
    scope = store_scope(store, credentials.learner_token)
    clock.advance(days=366)
    counts = store.purge()
    assert counts["learners"] == 1
    assert store.interaction_count(scope) == 0
