from __future__ import annotations

import pytest

from personal_ai_tutor.algebra import check_answer
from personal_ai_tutor.curriculum import Curriculum
from personal_ai_tutor.items import Item, ItemError, ItemFactory


def test_every_concept_self_grades(curriculum: Curriculum, factory: ItemFactory) -> None:
    for concept_id in curriculum.concepts:
        for variant in range(5):
            item = factory.build(concept_id, variant)
            assert item.steps
            for step in item.steps:
                assert check_answer(step.kind, step.answer, step.answer, step.options)
                assert not check_answer(step.kind, step.answer, "999999", step.options)
                assert len(step.hints) == 3
                assert step.hints[0] and step.worked


def test_hints_do_not_reveal_final_answer(curriculum: Curriculum, factory: ItemFactory) -> None:
    for concept_id in curriculum.concepts:
        for variant in range(5):
            item = factory.build(concept_id, variant)
            final = item.final_answer
            for step in item.steps:
                blob = " ".join((*step.hints, step.worked)).lower()
                assert f"answer is {final.lower()}" not in blob
                assert f"x = {final.lower()}" not in blob


def test_items_are_deterministic(factory: ItemFactory) -> None:
    first = factory.build("two_step_equations", 3)
    second = factory.build("two_step_equations", 3)
    assert first == second
    assert first.prompt == second.prompt


def test_graphing_items_have_visuals(factory: ItemFactory) -> None:
    assert factory.build("plotting_points", 0).visual is not None
    assert factory.build("slope", 0).visual is not None
    assert factory.build("coordinate_plane", 0).visual is not None


def test_item_json_roundtrip(factory: ItemFactory) -> None:
    item = factory.build("two_step_equations", 0)
    restored = Item.from_dict(item.to_dict())
    assert restored == item


def test_equation_intake_steps(factory: ItemFactory) -> None:
    item = factory.build_from_equation("3x + 6 = 12")
    assert item.concept_id == "two_step_equations"
    assert [step.answer for step in item.steps] == ["6", "6", "2"]
    both = factory.build_from_equation("5x - 3 = 2x + 9")
    assert both.concept_id == "variables_both_sides"
    assert both.steps[-1].answer == "4"
    negative = factory.build_from_equation("3x - 6 = 12")
    assert [step.answer for step in negative.steps] == ["6", "18", "6"]


def test_unknown_concept_is_explicit(factory: ItemFactory) -> None:
    with pytest.raises(ItemError):
        factory.build("not_a_concept", 0)
