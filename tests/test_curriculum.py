from __future__ import annotations

from pathlib import Path

import pytest

from personal_ai_tutor.curriculum import Curriculum, CurriculumError


def test_curriculum_shape(curriculum: Curriculum) -> None:
    assert len(curriculum.concepts) == 68
    assert len(curriculum.order) == 68
    assert len(curriculum.strands) == 8
    edges = sum(len(concept.prereqs) for concept in curriculum.concepts.values())
    assert edges == 113


def test_every_prerequisite_precedes_concept(curriculum: Curriculum) -> None:
    positions = {concept_id: index for index, concept_id in enumerate(curriculum.order)}
    for concept in curriculum.concepts.values():
        for prereq in concept.prereqs:
            assert prereq in curriculum.concepts
            assert positions[prereq] < positions[concept.id]


def test_eligibility_respects_prerequisites(curriculum: Curriculum) -> None:
    eligible = {concept.id for concept in curriculum.eligible({}, threshold=0.70)}
    assert "number_line" in eligible
    assert "two_step_equations" not in eligible
    mastery = {concept_id: 0.95 for concept_id in curriculum.concepts}
    assert {c.id for c in curriculum.eligible(mastery, 0.70)} == set(curriculum.concepts)


def test_next_concepts_orders_by_mastery(curriculum: Curriculum) -> None:
    mastery = {
        "number_line": 0.9,
        "integers_compare": 0.2,
        "integer_add_sub": 0.4,
        "fractions_equiv": 0.5,
    }
    next_items = curriculum.next_concepts(mastery, threshold=0.70, limit=2)
    assert next_items[0].id == "integers_compare"
    assert all(
        curriculum.concepts[prereq].id == prereq
        for concept in next_items
        for prereq in concept.prereqs
    )


def test_prerequisite_closure_and_depth(curriculum: Curriculum) -> None:
    closure = curriculum.prerequisite_closure("two_step_equations")
    assert {"one_step_add_sub", "one_step_mul_div", "variables_expressions"} <= set(closure)
    assert curriculum.depth("number_line") == 1
    assert curriculum.depth("two_step_equations") > 3


def test_cycle_is_rejected(curriculum: Curriculum) -> None:
    raw = {
        "concepts": [
            {
                "id": "a",
                "title": "A",
                "strand": "S",
                "difficulty": 1,
                "prereqs": ["b"],
                "lesson": "Original lesson.",
                "oer": {
                    "title": "OpenStax",
                    "url": "https://openstax.org/books/elementary-algebra-2e/pages/1-introduction",
                },
                "family": "number_line",
                "family_mode": "plot",
            },
            {
                "id": "b",
                "title": "B",
                "strand": "S",
                "difficulty": 1,
                "prereqs": ["a"],
                "lesson": "Original lesson.",
                "oer": {
                    "title": "OpenStax",
                    "url": "https://openstax.org/books/elementary-algebra-2e/pages/1-introduction",
                },
                "family": "number_line",
                "family_mode": "plot",
            },
        ]
    }
    with pytest.raises(CurriculumError):
        Curriculum.from_dict(raw)


def test_missing_file_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(CurriculumError):
        Curriculum.from_path(tmp_path / "missing.json")


def test_cypher_export_has_all_nodes_and_edges(curriculum: Curriculum) -> None:
    cypher = curriculum.export_cypher()
    assert cypher.count("MERGE (c:Concept") == 68
    assert cypher.count("[:PREREQ]") == 113
    assert "CREATE CONSTRAINT" in cypher
    assert "O'Brien" not in cypher
    assert all("openstax.org" in c.oer.url for c in curriculum.concepts.values())
