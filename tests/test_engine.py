from __future__ import annotations

import pytest

from personal_ai_tutor.engine import PhotoRecognitionError, SessionNotFound
from personal_ai_tutor.photo import render_png
from personal_ai_tutor.store import ScopeError


def test_text_demo_flow(engine, learner) -> None:
    turn = engine.start_session(learner.scope)
    turn = engine.reply(
        learner.scope,
        turn.session_id,
        "I don't understand why 3x + 6 = 12 means x = 2",
    )
    assert turn.concept_id == "two_step_equations"
    assert turn.message.endswith("?")
    assert "x = 2" not in turn.message
    turn = engine.reply(learner.scope, turn.session_id, "6")
    assert turn.signal == "correct" and not turn.solved
    turn = engine.reply(learner.scope, turn.session_id, "6")
    assert turn.signal == "correct" and not turn.solved
    turn = engine.reply(learner.scope, turn.session_id, "2")
    assert turn.solved
    assert turn.mastery_after is not None and turn.mastery_after > 0.5
    assert engine.store.interaction_count(learner.scope) >= 3


def test_photo_turn_recognizes_equation(engine, learner) -> None:
    turn = engine.start_session(learner.scope)
    photo = render_png("3x+6=12")
    result = engine.photo_turn(learner.scope, turn.session_id, photo)
    assert result.concept_id == "two_step_equations"
    assert "3x+6=12" in result.item_prompt
    assert result.message.endswith("?")


def test_photo_turn_rejects_garbage(engine, learner) -> None:
    turn = engine.start_session(learner.scope)
    with pytest.raises(PhotoRecognitionError):
        engine.photo_turn(learner.scope, turn.session_id, b"not an image")


def test_voice_turn_and_cross_speaker(engine, learner) -> None:
    turn = engine.start_session(learner.scope)
    voice = engine.voice_turn(
        learner.scope, turn.session_id, "6", learner.scope.voice_ref
    )
    assert voice.audio.startswith(b"RIFF")
    other = engine.create_learner("Other", "13_17", True)
    other_scope = engine.learner_scope(other.learner_token)
    with pytest.raises(ScopeError):
        engine.voice_turn(
            learner.scope, turn.session_id, "6", other_scope.voice_ref
        )
    assert any(
        event.category == "cross_learner_voice"
        for event in engine.store.safety_events(learner.scope)
    )


def test_sessions_are_learner_scoped(engine, learner) -> None:
    turn = engine.start_session(learner.scope)
    other = engine.create_learner("Other", "13_17", True)
    other_scope = engine.learner_scope(other.learner_token)
    with pytest.raises(ScopeError):
        engine.reply(other_scope, turn.session_id, "6")
    with pytest.raises(SessionNotFound):
        engine.reply(learner.scope, "sess_missing", "6")


def test_parent_progress_report_export_delete(engine, learner) -> None:
    turn = engine.start_session(learner.scope)
    engine.reply(learner.scope, turn.session_id, "I'm stuck")
    progress = engine.progress(learner.parent)
    assert progress["learner"]["id"] == learner.credentials.learner_id
    assert "retention" in progress
    pdf = engine.weekly_pdf(learner.parent, 0)
    assert pdf.startswith(b"%PDF-1.4")
    exported = engine.export(learner.parent)
    assert exported["learner"]["id"] == learner.credentials.learner_id
    engine.delete(learner.parent)
    assert engine.store.interaction_count(learner.scope) == 0


def test_access_safety_block_does_not_advance(engine, learner) -> None:
    turn = engine.start_session(learner.scope)
    before = engine.store.mastery_map(learner.scope)
    blocked = engine.reply(learner.scope, turn.session_id, "I want to hurt someone")
    assert blocked.safety_blocked
    assert engine.store.mastery_map(learner.scope) == before
