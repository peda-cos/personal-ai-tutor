from __future__ import annotations

from personal_ai_tutor.bkt import default_params
from personal_ai_tutor.items import ItemFactory
from personal_ai_tutor.policy import PolicyContext, SocraticPolicy


def _policy(engine) -> SocraticPolicy:
    return engine.policy


def _context(engine, concept_id: str, item, text: str, *, mastery=None, step=0, hint=0, hints=0):
    concept = engine.curriculum.concepts[concept_id]
    return PolicyContext(
        concept=concept,
        item=item,
        step_index=step,
        hint_level=hint,
        hints_used=hints,
        attempts=0,
        mastery=mastery,
        mastery_map={},
        due_concepts=(),
        text=text,
        bkt_params=default_params(concept),
    )


def test_demo_flow_and_mastery(engine) -> None:
    policy = _policy(engine)
    item = ItemFactory(engine.curriculum).build_from_equation("3x + 6 = 12")
    mastery = None
    step = 0
    hint = 0
    hints = 0
    first = policy.respond(
        _context(
            engine,
            "two_step_equations",
            item,
            "I don't understand why 3x + 6 = 12 means x = 2",
            mastery=mastery,
            step=step,
            hint=hint,
            hints=hints,
        )
    )
    assert first.signal == "help"
    assert first.response.endswith("?")
    assert "x = 2" not in first.response
    for text in ("6", "6", "2"):
        decision = policy.respond(
            _context(
                engine,
                "two_step_equations",
                item,
                text,
                mastery=mastery,
                step=step,
                hint=hint,
                hints=hints,
            )
        )
        step, hint, hints = decision.step_index, decision.hint_level, decision.hints_used
        if decision.mastery_after is not None:
            mastery = decision.mastery_after
    assert decision.item_solved
    assert decision.bkt_observation is True
    assert mastery is not None and mastery > 0.5
    assert decision.fsrs_grade in {2, 3, 4}


def test_repeated_answer_requests_never_reveal(engine) -> None:
    policy = _policy(engine)
    item = ItemFactory(engine.curriculum).build_from_equation("3x + 6 = 12")
    hint = 0
    hints = 0
    for _ in range(5):
        decision = policy.respond(
            _context(
                engine,
                "two_step_equations",
                item,
                "just tell me the answer",
                step=0,
                hint=hint,
                hints=hints,
            )
        )
        assert "x = 2" not in decision.response
        assert "answer is 2" not in decision.response.lower()
        assert decision.response.endswith("?")
        hint, hints = decision.hint_level, decision.hints_used
        assert decision.bkt_observation is None


def test_help_and_skip_semantics(engine) -> None:
    policy = _policy(engine)
    item = ItemFactory(engine.curriculum).build_from_equation("3x + 6 = 12")
    help_decision = policy.respond(
        _context(engine, "two_step_equations", item, "I'm stuck")
    )
    assert help_decision.bkt_observation is None
    assert help_decision.mastery_after is None
    skip_decision = policy.respond(
        _context(engine, "two_step_equations", item, "next problem")
    )
    assert skip_decision.item_failed
    assert skip_decision.bkt_observation is False
    assert skip_decision.fsrs_grade == 1


def test_selection_respects_prerequisites_and_linear_mode(engine) -> None:
    policy = _policy(engine)
    first = policy.select_concept({})
    assert first is not None and first.id == "number_line"
    next_concept = policy.select_concept(
        {concept_id: 0.9 for concept_id in engine.curriculum.concepts}
    )
    assert next_concept is not None
    linear = policy.select_concept(
        {"number_line": 0.95},
        mode="linear",
        current_id="number_line",
    )
    assert linear is not None and linear.id == "integers_compare"


def test_due_review_wins_when_available(engine) -> None:
    policy = _policy(engine)
    mastery = {concept_id: 0.95 for concept_id in engine.curriculum.concepts}
    selected = policy.select_concept(
        mastery,
        due_concepts=("fractions_equiv",),
        current_id="number_line",
    )
    assert selected is not None and selected.id == "fractions_equiv"


def test_memory_note_is_prepended(engine) -> None:
    policy = _policy(engine)
    item = ItemFactory(engine.curriculum).build_from_equation("3x + 6 = 12")
    concept = engine.curriculum.concepts["two_step_equations"]
    context = PolicyContext(
        concept=concept,
        item=item,
        step_index=0,
        hint_level=0,
        hints_used=0,
        attempts=0,
        mastery=None,
        mastery_map={},
        due_concepts=(),
        text="I'm stuck",
        memory_note="You practiced this before.",
        bkt_params=default_params(concept),
    )
    decision = policy.respond(context)
    assert decision.response.startswith("You practiced this before.")
