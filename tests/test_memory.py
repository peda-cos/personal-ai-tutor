from __future__ import annotations

import pytest

from personal_ai_tutor.memory import MemoryService
from personal_ai_tutor.store import ScopeError


def _record(memory: MemoryService, scope, *, concept: str, correct: bool, modality: str = "text"):
    memory.record_interaction(
        scope,
        session_id="sess_test",
        modality=modality,
        learner_text="I tried 6",
        tutor_text="What operation undoes the constant?",
        concept_id=concept,
        signal="correct" if correct else "incorrect",
        correct=correct,
        step_index=0,
        hints_used=0,
        mastery_after=0.5,
    )


def test_mistake_promotion_and_note(engine, learner) -> None:
    memory = engine.memory
    for _ in range(3):
        _record(memory, learner.scope, concept="two_step_equations", correct=False)
    facts = engine.store.semantic_map(learner.scope)
    assert "mistake:two_step_equations" in facts
    assert facts["mistake:two_step_equations"].confidence >= 0.8
    assert "scaffold" in (memory.note_for(learner.scope, "two_step_equations") or "")


def test_modality_preference(engine, learner) -> None:
    memory = engine.memory
    for _ in range(3):
        _record(
            memory,
            learner.scope,
            concept="number_line",
            correct=True,
            modality="voice",
        )
    profile = memory.profile(learner.scope)
    assert profile.preferred_modality == "voice"


def test_profile_returns_only_own_memory(engine, learner) -> None:
    other = engine.create_learner("Other", "13_17", True)
    other_scope = engine.learner_scope(other.learner_token)
    _record(engine.memory, learner.scope, concept="number_line", correct=True)
    _record(engine.memory, other_scope, concept="slope", correct=True)
    profile = engine.memory.profile(learner.scope)
    assert all(item.concept_id != "slope" for item in profile.recent)
    assert engine.store.interaction_count(learner.scope) == 1
    assert engine.store.interaction_count(other_scope) == 1


def test_parent_export_delete_and_retention_summary(engine, learner) -> None:
    _record(engine.memory, learner.scope, concept="number_line", correct=True)
    payload = engine.memory.export(learner.parent)
    assert payload["learner"]["id"] == learner.credentials.learner_id
    summary = engine.memory.retention_summary()
    assert summary["interactions_days"] == 365
    assert "not a legal compliance certification" in str(summary["certification"])
    engine.memory.delete(learner.parent)
    assert engine.store.interaction_count(learner.scope) == 0
    with pytest.raises(ScopeError):
        engine.export(learner.parent)
