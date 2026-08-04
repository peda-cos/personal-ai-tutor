from __future__ import annotations

from pathlib import Path

import pytest

from personal_ai_tutor.curriculum import Curriculum
from personal_ai_tutor.db import FrozenClock
from personal_ai_tutor.fsrs import initial_state
from personal_ai_tutor.items import ItemFactory
from personal_ai_tutor.store import (
    AuthError,
    ConsentRequired,
    LearnerScope,
    ScopeError,
    Session,
    Store,
    create_learner,
)


@pytest.fixture
def store(clock: FrozenClock, tmp_path: Path) -> Store:
    return Store(tmp_path / "store.db", clock)


def _session(store: Store, scope: LearnerScope, curriculum: Curriculum, session_id: str) -> Session:
    item = ItemFactory(curriculum).build("two_step_equations", 0)
    now = store.clock.now()
    return Session(
        id=session_id,
        learner_id=scope.learner_id,
        concept_id="two_step_equations",
        item=item,
        step_index=0,
        hint_level=0,
        hints_used=0,
        item_index=0,
        attempts=0,
        consecutive_wrong=0,
        status="active",
        created_at=now,
        updated_at=now,
    )


def test_authentication_and_wrong_secret(store: Store) -> None:
    from personal_ai_tutor.store import authenticate_learner, authenticate_parent

    create_learner(store, "Ada", "13_17", True)
    with pytest.raises(AuthError):
        authenticate_learner(store, "lt_wrong")
    with pytest.raises(AuthError):
        authenticate_parent(store, "pk_wrong")


def test_under_13_requires_consent(store: Store) -> None:
    with pytest.raises(ConsentRequired):
        create_learner(store, "Kid", "under_13", False)
    credentials = create_learner(store, "Kid", "under_13", True)
    assert credentials.learner_id.startswith("ln_")


def test_cross_learner_session_is_denied(
    store: Store, curriculum: Curriculum
) -> None:
    from personal_ai_tutor.store import authenticate_learner

    a = create_learner(store, "A", "13_17", True)
    b = create_learner(store, "B", "13_17", True)
    scope_a = authenticate_learner(store, a.learner_token)
    scope_b = authenticate_learner(store, b.learner_token)
    store.save_session(scope_b, _session(store, scope_b, curriculum, "sess_b"))
    with pytest.raises(ScopeError):
        store.get_session(scope_a, "sess_b")
    assert any(row.action == "scope_violation" for row in store.audit_rows())


def test_mastery_and_review_persistence(store: Store) -> None:
    from personal_ai_tutor.store import authenticate_learner

    credentials = create_learner(store, "M", "13_17", True)
    scope = authenticate_learner(store, credentials.learner_token)
    store.set_mastery(scope, "two_step_equations", 0.7, 2, 1)
    row = store.mastery_row(scope, "two_step_equations")
    assert row is not None and row.p == pytest.approx(0.7)
    assert store.mastery_map(scope)["two_step_equations"] == pytest.approx(0.7)
    state = initial_state(3, store.clock.now())
    store.set_review(scope, "two_step_equations", state)
    assert store.review_map(scope)["two_step_equations"].stability == pytest.approx(
        state.stability
    )


def test_export_excludes_secrets_and_delete_cascades(
    store: Store, curriculum: Curriculum
) -> None:
    from personal_ai_tutor.store import authenticate_learner, authenticate_parent

    a = create_learner(store, "Export A", "13_17", True)
    b = create_learner(store, "Export B", "13_17", True)
    scope_a = authenticate_learner(store, a.learner_token)
    parent_a = authenticate_parent(store, a.parent_key)
    scope_b = authenticate_learner(store, b.learner_token)
    store.set_mastery(scope_a, "number_line", 0.6, 1, 1)
    store.set_mastery(scope_b, "number_line", 0.9, 1, 1)
    store.add_interaction(
        scope_a,
        session_id="s",
        modality="text",
        learner_text="hello",
        tutor_text="question",
        concept_id="number_line",
        signal="help",
        correct=False,
        step_index=0,
        hints_used=1,
        mastery_after=None,
    )
    payload = store.export(parent_a)
    serialized = str(payload)
    assert "token_hash" not in serialized
    assert "parent_hash" not in serialized
    assert payload["learner"]["id"] == a.learner_id
    assert b.learner_id not in serialized
    store.delete(parent_a)
    assert store.get_session(scope_a, "sess_b") is None
    assert not store.audit_rows() or all(
        row.learner_id != a.learner_id for row in store.audit_rows()
    )
    assert store.mastery_map(scope_b)["number_line"] == pytest.approx(0.9)


def test_export_includes_all_interactions_not_only_recent(
    store: Store,
) -> None:
    from personal_ai_tutor.store import authenticate_learner, authenticate_parent

    credentials = create_learner(store, "Full Export", "13_17", True)
    scope = authenticate_learner(store, credentials.learner_token)
    parent = authenticate_parent(store, credentials.parent_key)
    for index in range(101):
        store.add_interaction(
            scope,
            session_id="s",
            modality="text",
            learner_text=f"attempt {index}",
            tutor_text="question",
            concept_id="number_line",
            signal="incorrect",
            correct=False,
            step_index=0,
            hints_used=0,
            mastery_after=None,
        )
    payload = store.export(parent)
    assert len(payload["interactions"]) == 101


def test_purge_old_rows_and_inactive_learners(
    store: Store, clock: FrozenClock
) -> None:
    from personal_ai_tutor.store import authenticate_learner

    old = create_learner(store, "Old", "13_17", True)
    recent = create_learner(store, "Recent", "13_17", True)
    old_scope = authenticate_learner(store, old.learner_token)
    recent_scope = authenticate_learner(store, recent.learner_token)
    store.add_interaction(
        old_scope,
        session_id="s",
        modality="text",
        learner_text="old",
        tutor_text="old",
        concept_id="number_line",
        signal="help",
        correct=False,
        step_index=0,
        hints_used=1,
        mastery_after=None,
    )
    clock.advance(days=400)
    store.add_interaction(
        recent_scope,
        session_id="s2",
        modality="text",
        learner_text="new",
        tutor_text="new",
        concept_id="number_line",
        signal="help",
        correct=False,
        step_index=0,
        hints_used=1,
        mastery_after=None,
    )
    counts = store.purge()
    assert counts["interactions"] >= 1
    assert counts["learners"] >= 1
    assert store.interaction_count(old_scope) == 0
    assert store.interaction_count(recent_scope) == 1
