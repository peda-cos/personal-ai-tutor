"""Socratic tutoring policy as a four-node LangGraph."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from . import bkt
from .algebra import check_answer
from .bkt import BktParams
from .curriculum import Concept, Curriculum
from .items import Item, ItemFactory

HELP_PHRASES = (
    "i don't know",
    "i dont know",
    "idk",
    "i'm stuck",
    "im stuck",
    "stuck",
    "help",
    "confused",
    "i don't understand",
    "i dont understand",
    "not sure",
    "just tell me",
    "give me the answer",
    "give me the answer",
    "show me",
)
SKIP_PHRASES = ("next problem", "skip", "new problem")
_SAFETY_QUESTIONS = {
    "self_harm",
    "guard_unavailable",
}


@dataclass(frozen=True)
class PolicyContext:
    concept: Concept
    item: Item
    step_index: int
    hint_level: int
    hints_used: int
    attempts: int
    mastery: float | None
    mastery_map: Mapping[str, float]
    due_concepts: Sequence[str]
    text: str
    memory_note: str | None = None
    selection_mode: str = "adaptive"
    concept_pool: Sequence[str] | None = None
    bkt_params: BktParams | None = None


@dataclass(frozen=True)
class PolicyDecision:
    signal: str
    response: str
    correct_step: bool
    item_solved: bool
    item_failed: bool
    advance_step: bool
    step_index: int
    hint_level: int
    hints_used: int
    bkt_observation: bool | None
    fsrs_grade: int | None
    mastery_after: float | None


class PolicyState(TypedDict, total=False):
    concept: Concept
    item: Item
    step_index: int
    hint_level: int
    hints_used: int
    attempts: int
    mastery: float | None
    mastery_map: dict[str, float]
    due_concepts: tuple[str, ...]
    text: str
    memory_note: str | None
    selection_mode: str
    concept_pool: tuple[str, ...] | None
    bkt_params: BktParams | None
    signal: str
    correct_step: bool
    item_solved: bool
    item_failed: bool
    advance_step: bool
    bkt_observation: bool | None
    fsrs_grade: int | None
    response: str
    mastery_after: float | None


class SocraticPolicy:
    def __init__(
        self,
        curriculum: Curriculum,
        factory: ItemFactory | None = None,
        mastery_target: float = 0.80,
        prereq_threshold: float = 0.70,
    ) -> None:
        self.curriculum = curriculum
        self.factory = factory or ItemFactory(curriculum)
        self.mastery_target = mastery_target
        self.prereq_threshold = prereq_threshold
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Selection

    def select_concept(
        self,
        mastery_map: Mapping[str, float],
        *,
        due_concepts: Sequence[str] = (),
        mode: str = "adaptive",
        pool: Sequence[str] | None = None,
        current_id: str | None = None,
        exclude_current: bool = False,
    ) -> Concept | None:
        allowed = set(pool) if pool is not None else set(self.curriculum.concepts)
        if mode == "linear":
            ordered = [cid for cid in self.curriculum.order if cid in allowed]
            if not ordered:
                return None
            if current_id is None:
                return self.curriculum.concepts[ordered[0]]
            try:
                index = ordered.index(current_id)
            except ValueError:
                return self.curriculum.concepts[ordered[0]]
            return self.curriculum.concepts[ordered[(index + 1) % len(ordered)]]

        eligible = [
            concept
            for concept in self.curriculum.eligible(
                mastery_map, self.prereq_threshold
            )
            if concept.id in allowed
        ]
        if exclude_current:
            eligible = [c for c in eligible if c.id != current_id]
        if not eligible:
            return None
        due = [cid for cid in due_concepts if cid in {c.id for c in eligible}]
        if due:
            due_set = set(due)
            eligible = sorted(
                eligible,
                key=lambda c: (
                    0 if c.id in due_set else 1,
                    due.index(c.id) if c.id in due_set else 0,
                    mastery_map.get(c.id, 0.0),
                    c.difficulty,
                    c.id,
                ),
            )
            return eligible[0]
        return min(
            eligible,
            key=lambda c: (mastery_map.get(c.id, 0.0), c.difficulty, c.id),
        )

    def next_item(self, concept_id: str, item_index: int) -> Item:
        return self.factory.build(concept_id, item_index)

    # ------------------------------------------------------------------
    # Graph nodes

    def _read_signal(self, state: PolicyState) -> PolicyState:
        text = state["text"].strip()
        item = state["item"]
        step = item.steps[state["step_index"]]
        lowered = text.lower()
        if any(phrase in lowered for phrase in SKIP_PHRASES):
            return {
                "signal": "skip",
                "correct_step": False,
                "item_failed": True,
                "bkt_observation": False,
                "fsrs_grade": 1,
            }
        if not text or any(phrase in lowered for phrase in HELP_PHRASES):
            return {
                "signal": "help",
                "correct_step": False,
                "item_failed": False,
                "bkt_observation": None,
            }
        if check_answer(step.kind, step.answer, text, step.options):
            final = state["step_index"] == len(item.steps) - 1
            return {
                "signal": "correct",
                "correct_step": True,
                "advance_step": not final,
                "item_solved": final,
                "bkt_observation": True if final else None,
            }
        # A learner who jumps to the final answer may skip the remaining steps.
        if state["step_index"] < len(item.steps) - 1:
            final_step = item.steps[-1]
            if check_answer(final_step.kind, final_step.answer, text, final_step.options):
                return {
                    "signal": "correct",
                    "correct_step": True,
                    "item_solved": True,
                    "bkt_observation": True,
                }
        if not re.search(r"[0-9xX=+\-*/^()]", text):
            return {
                "signal": "off_topic",
                "correct_step": False,
                "item_failed": False,
                "bkt_observation": None,
            }
        return {
            "signal": "incorrect",
            "correct_step": False,
            "item_failed": False,
            "bkt_observation": None,
        }

    def _select_concept_node(self, state: PolicyState) -> PolicyState:
        # The engine normally supplies an active item. This node is also used
        # when a caller starts from an empty state.
        if "concept" in state and "item" in state:
            return {}
        current = state.get("concept")
        concept = self.select_concept(
            state.get("mastery_map", {}),
            due_concepts=state.get("due_concepts", ()),
            mode=state.get("selection_mode", "adaptive"),
            pool=state.get("concept_pool"),
            current_id=current.id if current is not None else None,
        )
        if concept is None:
            return {"response": "There is no available concept yet."}
        item = self.factory.build(concept.id, 0)
        return {"concept": concept, "item": item, "step_index": 0}

    def _scaffold(self, state: PolicyState) -> PolicyState:
        item = state["item"]
        step_index = state.get("step_index", 0)
        step = item.steps[step_index]
        signal = state.get("signal", "incorrect")
        hint_level = state.get("hint_level", 0)
        hints_used = state.get("hints_used", 0)
        concept = state["concept"]
        if signal == "correct":
            if state.get("item_solved"):
                response = f"You've got it: {item.final_answer}. That is correct."
                return {"response": response}
            next_step = item.steps[step_index + 1]
            return {
                "response": f"Nice work. {next_step.question}",
                "advance_step": True,
                "step_index": step_index + 1,
                "hint_level": 0,
            }
        if signal == "skip":
            return {
                "response": "Let's try a different problem.",
                "item_failed": True,
            }
        if signal == "off_topic":
            return {
                "response": f"Let's keep working on {concept.title}. {step.question}",
                "hint_level": hint_level,
            }
        if signal in {"help", "incorrect"}:
            if hint_level >= 3:
                if step_index < len(item.steps) - 1:
                    response = f"{step.worked} What is the next step?"
                    return {
                        "response": response,
                        "advance_step": True,
                        "step_index": step_index + 1,
                        "hint_level": 0,
                        "hints_used": hints_used + 1,
                    }
                response = f"{step.worked} Try the final step again; what do you get?"
                hints_used += 1
                return {
                    "response": response,
                    "hint_level": 3,
                    "hints_used": hints_used,
                    "advance_step": False,
                }
            hint = step.hints[min(hint_level, 2)]
            response = f"{hint} What do you think?"
            return {
                "response": response,
                "hint_level": hint_level + 1,
                "hints_used": hints_used + 1,
            }
        return {"response": f"Let's keep going. {step.question}"}

    def _update_mastery(self, state: PolicyState) -> PolicyState:
        observation = state.get("bkt_observation")
        mastery = state.get("mastery")
        concept = state["concept"]
        params = state.get("bkt_params") or bkt.default_params(concept)
        mastery_after = mastery
        grade: int | None = None
        if observation is not None:
            prior = params.p_init if mastery is None else mastery
            mastery_after = bkt.update(prior, observation, params)
            if observation:
                hints = state.get("hints_used", 0)
                grade = 4 if hints <= 1 else 3 if hints <= 3 else 2
            else:
                grade = 1
        response = state.get("response", "")
        if not state.get("item_solved") and state.get("signal") not in {"skip", "off_topic"}:
            response = _guard_no_reveal(response, state["item"], state.get("signal", ""))
        return {
            "response": response,
            "mastery_after": mastery_after,
            "fsrs_grade": grade,
        }

    def _build_graph(self) -> object:
        graph = StateGraph(PolicyState)
        graph.add_node("read_signal", self._read_signal)
        graph.add_node("select_concept", self._select_concept_node)
        graph.add_node("scaffold", self._scaffold)
        graph.add_node("update_mastery", self._update_mastery)
        graph.add_edge(START, "read_signal")
        graph.add_edge("read_signal", "select_concept")
        graph.add_edge("select_concept", "scaffold")
        graph.add_edge("scaffold", "update_mastery")
        graph.add_edge("update_mastery", END)
        return graph.compile()

    def respond(self, context: PolicyContext) -> PolicyDecision:
        state: PolicyState = {
            "concept": context.concept,
            "item": context.item,
            "step_index": context.step_index,
            "hint_level": context.hint_level,
            "hints_used": context.hints_used,
            "attempts": context.attempts,
            "mastery": context.mastery,
            "mastery_map": dict(context.mastery_map),
            "due_concepts": tuple(context.due_concepts),
            "text": context.text,
            "memory_note": context.memory_note,
            "selection_mode": context.selection_mode,
            "concept_pool": tuple(context.concept_pool) if context.concept_pool else None,
            "bkt_params": context.bkt_params,
        }
        result = self._graph.invoke(state)  # type: ignore[attr-defined]
        response = str(result.get("response", "")).strip()
        if context.memory_note and result.get("signal") not in {"skip", "off_topic"}:
            response = f"{context.memory_note} {response}"
        item = result["item"]
        step_index = int(result.get("step_index", context.step_index))
        hint_level = int(result.get("hint_level", context.hint_level))
        hints_used = int(result.get("hints_used", context.hints_used))
        step_index = min(step_index, len(item.steps) - 1)
        return PolicyDecision(
            signal=str(result.get("signal", "incorrect")),
            response=response,
            correct_step=bool(result.get("correct_step", False)),
            item_solved=bool(result.get("item_solved", False)),
            item_failed=bool(result.get("item_failed", False)),
            advance_step=bool(result.get("advance_step", False)),
            step_index=step_index,
            hint_level=hint_level,
            hints_used=hints_used,
            bkt_observation=result.get("bkt_observation"),
            fsrs_grade=result.get("fsrs_grade"),
            mastery_after=result.get("mastery_after", context.mastery),
        )


def _guard_no_reveal(response: str, item: Item, signal: str) -> str:
    if signal == "correct":
        return response
    final = item.final_answer
    lowered = response.lower()
    if f"answer is {final.lower()}" in lowered or f"x = {final.lower()}" in lowered:
        return "Let's use a scaffold instead of skipping ahead. What would you try first?"
    if not response.rstrip().endswith("?") and signal not in _SAFETY_QUESTIONS:
        response = response.rstrip() + " What do you think?"
    return response
