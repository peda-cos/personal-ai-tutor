"""Episodic and semantic learner memory with COPPA-aware retention."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .store import (
    Interaction,
    LearnerScope,
    ParentScope,
    SemanticFact,
    Store,
)


@dataclass(frozen=True)
class MemoryProfile:
    recent: tuple[Interaction, ...]
    facts: dict[str, SemanticFact]

    @property
    def preferred_modality(self) -> str | None:
        fact = self.facts.get("preferred_modality")
        return fact.value if fact is not None else None


class MemoryService:
    def __init__(self, store: Store) -> None:
        self.store = store

    def record_interaction(
        self,
        scope: LearnerScope,
        *,
        session_id: str,
        modality: str,
        learner_text: str,
        tutor_text: str,
        concept_id: str | None,
        signal: str,
        correct: bool,
        step_index: int,
        hints_used: int,
        mastery_after: float | None,
        safety_category: str | None = None,
    ) -> None:
        self.store.add_interaction(
            scope,
            session_id=session_id,
            modality=modality,
            learner_text=learner_text,
            tutor_text=tutor_text,
            concept_id=concept_id,
            signal=signal,
            correct=correct,
            step_index=step_index,
            hints_used=hints_used,
            mastery_after=mastery_after,
            safety_category=safety_category,
        )
        if concept_id is not None:
            self._promote(scope, concept_id, correct, modality)

    def _promote(
        self, scope: LearnerScope, concept_id: str, correct: bool, modality: str
    ) -> None:
        mistakes = self.store.count_interactions(
            scope, concept_id=concept_id, correct=False
        )
        if mistakes >= 3:
            self.store.upsert_semantic(
                scope,
                f"mistake:{concept_id}",
                "needs more practice",
                min(0.9, 0.5 + 0.1 * mistakes),
            )
        modality_count = self.store.count_interactions(scope, modality=modality)
        if modality in {"voice", "photo"} and modality_count >= 3:
            self.store.upsert_semantic(
                scope, "preferred_modality", modality, 0.6
            )

    def profile(self, scope: LearnerScope, recent_limit: int = 5) -> MemoryProfile:
        return MemoryProfile(
            recent=tuple(self.store.recent_interactions(scope, recent_limit)),
            facts=self.store.semantic_map(scope),
        )

    def note_for(self, scope: LearnerScope, concept_id: str) -> str | None:
        fact = self.store.semantic_map(scope).get(f"mistake:{concept_id}")
        if fact is None:
            return None
        return "You have practiced this concept before; let's use a scaffold and check each step."

    def export(self, parent: ParentScope) -> dict[str, Any]:
        return self.store.export(parent)

    def delete(self, parent: ParentScope) -> None:
        self.store.delete(parent)

    def purge_expired(self, now: datetime | None = None) -> dict[str, int]:
        return self.store.purge(now)

    def retention_summary(self) -> dict[str, object]:
        return {
            "purposes": [
                "tutoring and step feedback",
                "mastery tracking and spaced review",
                "safety auditing",
                "parent progress reporting",
            ],
            "interactions_days": 365,
            "semantic_days": 365,
            "safety_events_days": 365,
            "audit_days": 730,
            "idle_session_days": 30,
            "inactive_learner_days": 365,
            "parent_rights": [
                "review a JSON export of the learner's data",
                "delete all learner data at any time",
                "see safety event categories, not raw blocked content",
            ],
            "certification": "COPPA-aware design, not a legal compliance certification",
        }
