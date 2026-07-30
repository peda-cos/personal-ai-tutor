"""Algebra curriculum graph: validation, traversal, and Cypher export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class CurriculumError(ValueError):
    """Raised when the curriculum file is missing or inconsistent."""


@dataclass(frozen=True)
class Oer:
    title: str
    url: str


@dataclass(frozen=True)
class Concept:
    id: str
    title: str
    strand: str
    difficulty: int
    prereqs: tuple[str, ...]
    lesson: str
    oer: Oer
    family: str
    family_mode: str


@dataclass(frozen=True)
class Curriculum:
    concepts: dict[str, Concept]
    order: tuple[str, ...]

    @classmethod
    def load_default(cls) -> Curriculum:
        path = Path(__file__).parent / "data" / "algebra.json"
        return cls.from_path(path)

    @classmethod
    def from_path(cls, path: Path) -> Curriculum:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise CurriculumError(f"cannot read curriculum at {path}") from exc
        except json.JSONDecodeError as exc:
            raise CurriculumError(f"invalid curriculum JSON at {path}") from exc
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Curriculum:
        concepts_raw = raw.get("concepts")
        if not isinstance(concepts_raw, list):
            raise CurriculumError("curriculum must contain a concepts list")
        concepts: dict[str, Concept] = {}
        for item in concepts_raw:
            if not isinstance(item, dict):
                raise CurriculumError("every concept must be an object")
            try:
                oer_raw = item["oer"]
                concept = Concept(
                    id=str(item["id"]),
                    title=str(item["title"]),
                    strand=str(item["strand"]),
                    difficulty=int(item["difficulty"]),
                    prereqs=tuple(str(p) for p in item["prereqs"]),
                    lesson=str(item["lesson"]),
                    oer=Oer(title=str(oer_raw["title"]), url=str(oer_raw["url"])),
                    family=str(item["family"]),
                    family_mode=str(item["family_mode"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise CurriculumError(f"malformed concept: {item!r}") from exc
            if concept.id in concepts:
                raise CurriculumError(f"duplicate concept id: {concept.id}")
            concepts[concept.id] = concept
        order = cls._topological_order(concepts)
        curriculum = cls(concepts=concepts, order=order)
        curriculum.validate(raw)
        return curriculum

    @staticmethod
    def _topological_order(concepts: Mapping[str, Concept]) -> tuple[str, ...]:
        remaining = list(concepts)
        done: set[str] = set()
        order: list[str] = []
        while remaining:
            ready = [
                cid for cid in remaining if all(p in done for p in concepts[cid].prereqs)
            ]
            if not ready:
                raise CurriculumError("curriculum prerequisite graph contains a cycle")
            for cid in ready:
                order.append(cid)
                done.add(cid)
            remaining = [cid for cid in remaining if cid not in done]
        return tuple(order)

    def validate(self, raw: Mapping[str, Any] | None = None) -> None:
        if not 50 <= len(self.concepts) <= 150:
            raise CurriculumError("curriculum must contain 50-150 concepts")
        for concept in self.concepts.values():
            if not concept.id or not concept.title or not concept.lesson:
                raise CurriculumError(f"concept {concept.id!r} is missing required text")
            if concept.difficulty < 1 or concept.difficulty > 5:
                raise CurriculumError(f"concept {concept.id!r} difficulty must be 1-5")
            if not concept.oer.url.startswith("https://openstax.org/"):
                raise CurriculumError(f"concept {concept.id!r} needs an OpenStax OER URL")
            for prereq in concept.prereqs:
                if prereq not in self.concepts:
                    raise CurriculumError(f"unknown prerequisite {prereq!r}")
                if prereq == concept.id:
                    raise CurriculumError(f"concept {concept.id!r} depends on itself")
            if concept.id not in self.order:
                raise CurriculumError(f"concept {concept.id!r} missing from topological order")
        positions = {cid: i for i, cid in enumerate(self.order)}
        for concept in self.concepts.values():
            for prereq in concept.prereqs:
                if positions[prereq] >= positions[concept.id]:
                    raise CurriculumError(
                        f"prerequisite {prereq!r} must precede {concept.id!r}"
                    )
        if raw is not None:
            strands = raw.get("strands")
            if strands is not None and set(strands) != {
                c.strand for c in self.concepts.values()
            }:
                raise CurriculumError("declared strands do not match concept strands")

    @property
    def strands(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(c.strand for c in self.concepts.values()))

    def eligible(
        self, mastery: Mapping[str, float], threshold: float = 0.70
    ) -> list[Concept]:
        return [
            concept
            for concept in self.concepts.values()
            if all(mastery.get(p, 0.0) >= threshold for p in concept.prereqs)
        ]

    def next_concepts(
        self,
        mastery: Mapping[str, float],
        threshold: float = 0.70,
        limit: int = 3,
    ) -> list[Concept]:
        eligible = self.eligible(mastery, threshold)
        return sorted(eligible, key=lambda c: (mastery.get(c.id, 0.0), c.difficulty, c.id))[
            :limit
        ]

    def prerequisite_closure(self, concept_id: str) -> tuple[str, ...]:
        if concept_id not in self.concepts:
            raise KeyError(concept_id)
        seen: set[str] = set()
        stack = list(self.concepts[concept_id].prereqs)
        while stack:
            cid = stack.pop()
            if cid in seen:
                continue
            seen.add(cid)
            stack.extend(self.concepts[cid].prereqs)
        return tuple(seen)

    def depth(self, concept_id: str) -> int:
        concept = self.concepts[concept_id]
        if not concept.prereqs:
            return 1
        return 1 + max(self.depth(p) for p in concept.prereqs)

    def export_cypher(self) -> str:
        lines = [
            "// Generated by personal_ai_tutor. Import with: cypher-shell -f curriculum.cypher",
            "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE;",
        ]
        for concept in self.concepts.values():
            lines.append(
                "MERGE (c:Concept {id: '%s'}) SET c.title = '%s', c.strand = '%s', "
                "c.difficulty = %d, c.lesson = '%s', c.oer = '%s';"
                % (
                    _cypher_escape(concept.id),
                    _cypher_escape(concept.title),
                    _cypher_escape(concept.strand),
                    concept.difficulty,
                    _cypher_escape(concept.lesson),
                    _cypher_escape(concept.oer.url),
                )
            )
        for concept in self.concepts.values():
            for prereq in concept.prereqs:
                lines.append(
                    "MATCH (a:Concept {id: '%s'}), (b:Concept {id: '%s'}) "
                    "MERGE (a)-[:PREREQ]->(b);"
                    % (_cypher_escape(prereq), _cypher_escape(concept.id))
                )
        return "\n".join(lines) + "\n"


def _cypher_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ")
