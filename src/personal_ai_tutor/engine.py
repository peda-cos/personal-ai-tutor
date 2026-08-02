"""Tutor engine: safety, sessions, policy, persistence, and parent access."""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Callable, Sequence

from . import fsrs
from .algebra import check_answer
from .bkt import BktParams, default_params
from .config import Settings
from .curriculum import Concept, Curriculum
from .db import Clock, SystemClock
from .items import ItemFactory, ItemError
from .llm import ClaudeScaffolder
from .memory import MemoryService
from .photo import HttpVlmOcr, ImageError, SyntheticOcr, decode_image
from .policy import PolicyContext, PolicyDecision, SocraticPolicy
from .safety import LlamaGuard, ProviderError, SafetyService, SafetyVerdict
from .voice import build_tts
from .store import (
    Credentials,
    LearnerScope,
    ParentScope,
    ScopeError,
    Session,
    Store,
    authenticate_learner,
    authenticate_parent,
    create_learner,
)


class SessionNotFound(LookupError):
    """Raised when a learner asks for a session that does not exist."""


class PhotoRecognitionError(ValueError):
    """Raised when a photo cannot be recognized with enough confidence."""


@dataclass(frozen=True)
class Turn:
    session_id: str
    message: str
    concept_id: str
    concept_title: str
    item_prompt: str
    visual: dict[str, object] | None
    step_question: str
    mastery_before: float | None
    mastery_after: float | None
    signal: str
    solved: bool
    safety_blocked: bool
    hint_level: int


@dataclass(frozen=True)
class VoiceTurn:
    turn: Turn
    audio: bytes


@dataclass(frozen=True)
class TutorReply:
    turn: Turn
    decision: PolicyDecision | None = None


_BARE_ANSWER = re.compile(r"^\s*[a-zA-Z]\s*=\s*-?\d+(?:\.\d+)?(?:/\d+)?\s*$")
_EQUATION_CANDIDATE = re.compile(
    r"(?<![A-Za-z0-9_])[0-9xXyY(+-][-+*/().\d\sxXyY^]*=[-+*/().\d\sxXyY^]+"
)


class TutorEngine:
    def __init__(
        self,
        settings: Settings | None = None,
        curriculum: Curriculum | None = None,
        selection_mode: str = "adaptive",
        concept_pool: Sequence[str] | None = None,
        clock: Clock | None = None,
        bkt_params_for: Callable[[str], BktParams] | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.curriculum = curriculum or Curriculum.load_default()
        self.factory = ItemFactory(self.curriculum)
        self.clock = clock or SystemClock()
        self.store = Store(self.settings.db_path, self.clock)
        self.memory = MemoryService(self.store)
        guard = None
        if self.settings.guard in {"llama_guard", "both"}:
            if not self.settings.guard_url:
                raise ValueError("TUTOR_GUARD_URL is required for Llama Guard mode")
            guard = LlamaGuard(
                self.settings.guard_url,
                self.settings.guard_model,
                self.settings.guard_api_key,
            )
        self.safety = SafetyService(
            self.store, guard=guard, fail_closed=self.settings.guard_fail_closed
        )
        self.policy = SocraticPolicy(
            self.curriculum,
            self.factory,
            mastery_target=self.settings.mastery_target,
            prereq_threshold=self.settings.prereq_threshold,
        )
        self.selection_mode = selection_mode
        self.concept_pool = tuple(concept_pool) if concept_pool else None
        if self.concept_pool is not None:
            unknown = [cid for cid in self.concept_pool if cid not in self.curriculum.concepts]
            if unknown:
                raise ValueError(f"unknown concept pool entries: {unknown}")
        self.bkt_params_for = bkt_params_for or (
            lambda cid: default_params(self.curriculum.concepts[cid])
        )
        self.ocr: HttpVlmOcr | SyntheticOcr
        if self.settings.ocr == "http":
            if not self.settings.ocr_url:
                raise ValueError("TUTOR_OCR_URL is required for HTTP OCR mode")
            self.ocr = HttpVlmOcr(
                self.settings.ocr_url,
                self.settings.ocr_model,
                self.settings.ocr_api_key,
            )
        else:
            self.ocr = SyntheticOcr()
        self.tts = build_tts(self.settings)
        self.scaffolder: ClaudeScaffolder | None = None
        if self.settings.llm == "claude":
            if not self.settings.anthropic_api_key or not self.settings.claude_model:
                raise ValueError(
                    "ANTHROPIC_API_KEY and TUTOR_CLAUDE_MODEL are required for Claude mode"
                )
            self.scaffolder = ClaudeScaffolder(
                self.settings.anthropic_api_key,
                self.settings.claude_model,
            )

    # ------------------------------------------------------------------
    # Credentials

    def create_learner(
        self, display_name: str, age_band: str, consent: bool
    ) -> Credentials:
        return create_learner(self.store, display_name, age_band, consent)

    def learner_scope(self, token: str) -> LearnerScope:
        return authenticate_learner(self.store, token)

    def parent_scope(self, parent_key: str) -> ParentScope:
        return authenticate_parent(self.store, parent_key)

    # ------------------------------------------------------------------
    # Sessions

    def start_session(self, scope: LearnerScope) -> Turn:
        now = self.clock.now()
        mastery = self.store.mastery_map(scope)
        due = self._due_concepts(scope, now)
        concept = self.policy.select_concept(
            mastery,
            due_concepts=due,
            mode=self.selection_mode,
            pool=self.concept_pool,
        )
        if concept is None:
            raise SessionNotFound("no concept is available")
        item = self.policy.next_item(concept.id, 0)
        session = Session(
            id="sess_" + secrets.token_hex(8),
            learner_id=scope.learner_id,
            concept_id=concept.id,
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
        self.store.save_session(scope, session)
        return self._turn_for(
            session,
            f"Let's work on {concept.title}. {item.prompt} {item.steps[0].question}",
            signal="start",
            solved=False,
            safety_blocked=False,
        )

    def reply(
        self,
        scope: LearnerScope,
        session_id: str,
        text: str,
        modality: str = "text",
    ) -> Turn:
        session = self.store.get_session(scope, session_id)
        if session is None:
            raise SessionNotFound("session not found")
        verdict = self.safety.check_input(scope, text)
        if not verdict.allowed:
            self._record_blocked(scope, session, verdict, modality)
            return self._turn_for(
                session,
                verdict.response or "I can't help with that.",
                signal="blocked",
                solved=False,
                safety_blocked=True,
            )
        redacted = verdict.redacted_text
        equation = self._extract_equation(redacted)
        current_step = session.item.steps[
            min(session.step_index, len(session.item.steps) - 1)
        ]
        step_is_equation_answer = (
            current_step.kind == "expression"
            and check_answer(
                current_step.kind, current_step.answer, redacted, current_step.options
            )
        )
        if (
            equation is not None
            and not step_is_equation_answer
            and not self._same_as_current(equation, session)
        ):
            return self._start_problem(scope, session, equation, modality, verdict)
        mastery_before = self.store.mastery_map(scope).get(session.concept_id)
        exhausted = session.attempts >= 8
        decision = self._respond(
            scope,
            session,
            "skip" if exhausted else redacted,
            modality,
        )
        message = decision.response
        if exhausted:
            message = (
                "We have spent enough time on this one. Let's try a fresh problem. "
                + message
            )
        if verdict.response:
            message = f"{verdict.response} {message}".strip()
        output = self.safety.check_output(scope, message)
        if not output.allowed:
            message = output.response or message
            blocked = True
        else:
            message = output.redacted_text
            blocked = False
        solved = decision.item_solved or decision.item_failed
        if solved:
            session, new_prompt = self._advance(scope, session, decision)
            message = f"{message}\n\nNext: {new_prompt}"
        else:
            session = self._save_decision(scope, session, decision)
            if session.consecutive_wrong >= 3:
                stepped = self._step_back(scope, session)
                if stepped is not None:
                    session, step_prompt = stepped
                    message = f"{message}\n\nLet's step back. {step_prompt}"
        return self._turn_for(
            session,
            message,
            signal=decision.signal,
            solved=decision.item_solved,
            safety_blocked=blocked,
            mastery_before=mastery_before,
        )

    def _respond(
        self,
        scope: LearnerScope,
        session: Session,
        text: str,
        modality: str,
    ) -> PolicyDecision:
        concept = self.curriculum.concepts[session.concept_id]
        mastery_map = self.store.mastery_map(scope)
        due = self._due_concepts(scope, self.clock.now())
        decision = self.policy.respond(
            PolicyContext(
                concept=concept,
                item=session.item,
                step_index=session.step_index,
                hint_level=session.hint_level,
                hints_used=session.hints_used,
                attempts=session.attempts,
                mastery=mastery_map.get(concept.id),
                mastery_map=mastery_map,
                due_concepts=due,
                text=text,
                memory_note=self.memory.note_for(scope, concept.id),
                selection_mode=self.selection_mode,
                concept_pool=self.concept_pool,
                bkt_params=self.bkt_params_for(concept.id),
            )
        )
        if self.scaffolder is not None and decision.signal in {"help", "incorrect"}:
            try:
                step = session.item.steps[
                    min(session.step_index, len(session.item.steps) - 1)
                ]
                rewritten = self.scaffolder.scaffold(
                    concept.title,
                    step.question,
                    decision.response,
                    session.item.final_answer,
                )
                decision = replace(decision, response=rewritten)
            except ProviderError:
                self.store.audit(
                    scope.learner_id,
                    scope.learner_id,
                    "llm_fallback",
                    False,
                    "Claude scaffolding failed; deterministic hint used",
                )
        if decision.bkt_observation is not None:
            row = self.store.mastery_row(scope, concept.id)
            attempts = (row.attempts if row else 0) + 1
            correct = (row.correct if row else 0) + (
                1 if decision.bkt_observation else 0
            )
            self.store.set_mastery(
                scope,
                concept.id,
                decision.mastery_after or 0.0,
                attempts,
                correct,
            )
            grade = decision.fsrs_grade or 1
            previous = self.store.review_map(scope).get(concept.id)
            self.store.set_review(
                scope,
                concept.id,
                fsrs.review(previous, grade, self.clock.now()),
            )
        self.memory.record_interaction(
            scope,
            session_id=session.id,
            modality=modality,
            learner_text=text,
            tutor_text=decision.response,
            concept_id=concept.id,
            signal=decision.signal,
            correct=bool(decision.correct_step),
            step_index=decision.step_index,
            hints_used=decision.hints_used,
            mastery_after=decision.mastery_after,
        )
        return decision

    def _save_decision(
        self, scope: LearnerScope, session: Session, decision: PolicyDecision
    ) -> Session:
        updated = Session(
            id=session.id,
            learner_id=session.learner_id,
            concept_id=session.concept_id,
            item=session.item,
            step_index=decision.step_index,
            hint_level=decision.hint_level,
            hints_used=decision.hints_used,
            item_index=session.item_index,
            attempts=session.attempts + 1,
            consecutive_wrong=(
                0
                if decision.signal == "correct"
                else session.consecutive_wrong + 1
                if decision.signal in {"incorrect", "help"}
                else session.consecutive_wrong
            ),
            status=session.status,
            created_at=session.created_at,
            updated_at=self.clock.now(),
        )
        self.store.save_session(scope, updated)
        return updated

    def _advance(
        self, scope: LearnerScope, session: Session, decision: PolicyDecision
    ) -> tuple[Session, str]:
        mastery = self.store.mastery_map(scope)
        concept = self.curriculum.concepts[session.concept_id]
        next_concept: Concept | None = None
        next_index = session.item_index + 1
        if self.selection_mode == "linear":
            next_concept = self.policy.select_concept(
                mastery,
                mode="linear",
                pool=self.concept_pool,
                current_id=concept.id,
            )
            next_index = 0
        elif decision.item_solved and (decision.mastery_after or 0.0) >= self.settings.mastery_target:
            next_concept = self.policy.select_concept(
                mastery,
                due_concepts=self._due_concepts(scope, self.clock.now()),
                mode="adaptive",
                pool=self.concept_pool,
                current_id=concept.id,
                exclude_current=True,
            )
            next_index = 0
        elif session.consecutive_wrong >= 3 and not decision.item_solved:
            weak_prereqs = [
                self.curriculum.concepts[p]
                for p in concept.prereqs
                if mastery.get(p, 0.0) < self.settings.prereq_threshold
            ]
            if weak_prereqs:
                next_concept = min(
                    weak_prereqs, key=lambda c: (mastery.get(c.id, 0.0), c.difficulty)
                )
                next_index = 0
        if next_concept is None:
            next_concept = concept
        item = self.policy.next_item(next_concept.id, next_index)
        now = self.clock.now()
        updated = Session(
            id=session.id,
            learner_id=session.learner_id,
            concept_id=next_concept.id,
            item=item,
            step_index=0,
            hint_level=0,
            hints_used=0,
            item_index=next_index,
            attempts=0,
            consecutive_wrong=0,
            status="active",
            created_at=session.created_at,
            updated_at=now,
        )
        self.store.save_session(scope, updated)
        return updated, f"{item.prompt} {item.steps[0].question}"

    def _start_problem(
        self,
        scope: LearnerScope,
        session: Session,
        equation: str,
        modality: str,
        verdict: SafetyVerdict,
    ) -> Turn:
        try:
            item = self.factory.build_from_equation(equation)
        except (ItemError, ValueError):
            return self.reply(scope, session.id, "I couldn't read that equation. Can you type it again?", modality)
        concept = self.curriculum.concepts[item.concept_id]
        updated = Session(
            id=session.id,
            learner_id=session.learner_id,
            concept_id=concept.id,
            item=item,
            step_index=0,
            hint_level=0,
            hints_used=0,
            item_index=0,
            attempts=0,
            consecutive_wrong=0,
            status="active",
            created_at=session.created_at,
            updated_at=self.clock.now(),
        )
        self.store.save_session(scope, updated)
        message = (
            f"Let's look at {item.prompt.rstrip('.')}. {item.steps[0].question}"
        )
        if verdict.response:
            message = f"{verdict.response} {message}"
        output = self.safety.check_output(scope, message)
        if not output.allowed:
            message = output.response or message
        return self._turn_for(
            updated,
            message,
            signal="problem",
            solved=False,
            safety_blocked=not output.allowed,
        )

    def photo_turn(
        self,
        scope: LearnerScope,
        session_id: str,
        image_bytes: bytes,
        media_type: str = "image/png",
    ) -> Turn:
        session = self.store.get_session(scope, session_id)
        if session is None:
            raise SessionNotFound("session not found")
        try:
            image = decode_image(image_bytes, media_type)
            result = self.ocr.recognize(image)
            self.factory.build_from_equation(result.text)
        except (ImageError, ProviderError, ItemError, ValueError) as exc:
            raise PhotoRecognitionError(
                "I couldn't read that equation; try a clearer printed photo."
            ) from exc
        verdict = self.safety.check_input(scope, result.text)
        if not verdict.allowed:
            self._record_blocked(scope, session, verdict, "photo")
            return self._turn_for(
                session,
                verdict.response or "I can't help with that.",
                signal="blocked",
                solved=False,
                safety_blocked=True,
            )
        return self._start_problem(scope, session, result.text, "photo", verdict)

    def voice_turn(
        self,
        scope: LearnerScope,
        session_id: str,
        transcript: str,
        speaker_ref: str,
        barge_in: bool = False,
    ) -> VoiceTurn:
        session = self.store.get_session(scope, session_id)
        if session is None:
            raise SessionNotFound("session not found")
        if speaker_ref != scope.voice_ref:
            self.store.audit(
                scope.learner_id,
                scope.learner_id,
                "cross_learner_voice_ingest",
                False,
                "speaker label did not match the authenticated learner",
            )
            self.store.add_safety_event(
                scope, "voice", "cross_learner_voice", True, transcript
            )
            raise ScopeError("voice clip does not belong to this learner")
        turn = self.reply(scope, session_id, transcript, modality="voice")
        audio = b"".join(self.tts.synthesize(turn.message))
        return VoiceTurn(turn=turn, audio=audio)

    def _record_blocked(
        self,
        scope: LearnerScope,
        session: Session,
        verdict: SafetyVerdict,
        modality: str,
    ) -> None:
        concept = self.curriculum.concepts[session.concept_id]
        self.memory.record_interaction(
            scope,
            session_id=session.id,
            modality=modality,
            learner_text="[blocked]",
            tutor_text=verdict.response or "[blocked]",
            concept_id=concept.id,
            signal="blocked",
            correct=False,
            step_index=session.step_index,
            hints_used=session.hints_used,
            mastery_after=None,
            safety_category=verdict.category,
        )

    def _step_back(
        self, scope: LearnerScope, session: Session
    ) -> tuple[Session, str] | None:
        concept = self.curriculum.concepts[session.concept_id]
        mastery = self.store.mastery_map(scope)
        weak = [
            self.curriculum.concepts[prereq]
            for prereq in concept.prereqs
            if mastery.get(prereq, 0.0) < self.settings.prereq_threshold
        ]
        if not weak:
            return None
        target = min(weak, key=lambda item: (mastery.get(item.id, 0.0), item.difficulty))
        item = self.policy.next_item(target.id, 0)
        updated = Session(
            id=session.id,
            learner_id=session.learner_id,
            concept_id=target.id,
            item=item,
            step_index=0,
            hint_level=0,
            hints_used=0,
            item_index=0,
            attempts=0,
            consecutive_wrong=0,
            status="active",
            created_at=session.created_at,
            updated_at=self.clock.now(),
        )
        self.store.save_session(scope, updated)
        return updated, f"{target.title}: {item.prompt} {item.steps[0].question}"

    def _turn_for(
        self,
        session: Session,
        message: str,
        *,
        signal: str,
        solved: bool,
        safety_blocked: bool,
        mastery_before: float | None = None,
    ) -> Turn:
        concept = self.curriculum.concepts[session.concept_id]
        row = self.store.mastery_row(
            LearnerScope(session.learner_id, ""), concept.id
        )
        return Turn(
            session_id=session.id,
            message=message,
            concept_id=concept.id,
            concept_title=concept.title,
            item_prompt=session.item.prompt,
            visual=session.item.visual,
            step_question=session.item.steps[min(session.step_index, len(session.item.steps) - 1)].question,
            mastery_before=mastery_before,
            mastery_after=row.p if row is not None else None,
            signal=signal,
            solved=solved,
            safety_blocked=safety_blocked,
            hint_level=session.hint_level,
        )

    def _due_concepts(self, scope: LearnerScope, now: datetime) -> tuple[str, ...]:
        reviews = self.store.review_map(scope)
        due = [cid for cid, state in reviews.items() if state.due_at <= now]
        return tuple(sorted(due, key=lambda cid: reviews[cid].due_at))

    def _extract_equation(self, text: str) -> str | None:
        if _BARE_ANSWER.match(text):
            return None
        candidates = [m.group(0).strip() for m in _EQUATION_CANDIDATE.finditer(text)]
        valid: list[str] = []
        for candidate in candidates:
            if _BARE_ANSWER.match(candidate):
                continue
            if not re.search(r"[a-zA-Z]", candidate):
                continue
            try:
                self.factory.build_from_equation(candidate)
            except (ItemError, ValueError):
                continue
            valid.append(candidate)
        if not valid:
            return None
        return max(valid, key=len)

    def _same_as_current(self, equation: str, session: Session) -> bool:
        compact = re.sub(r"\s+", "", equation)
        prompt = re.sub(r"\s+", "", session.item.prompt)
        return compact in prompt

    # ------------------------------------------------------------------
    # Parent-facing data

    def progress(self, parent: ParentScope) -> dict[str, object]:
        scope = LearnerScope(parent.learner_id, "")
        learner = self.store.learner_row(parent)
        mastery = self.store.mastery_map(scope)
        rows = []
        for cid, p in sorted(mastery.items(), key=lambda item: item[1]):
            concept = self.curriculum.concepts.get(cid)
            if concept is not None:
                row = self.store.mastery_row(scope, cid)
                rows.append(
                    {
                        "concept_id": cid,
                        "title": concept.title,
                        "mastery": p,
                        "attempts": row.attempts if row else 0,
                        "correct": row.correct if row else 0,
                    }
                )
        next_concepts = [
            {"id": c.id, "title": c.title}
            for c in self.curriculum.next_concepts(
                mastery, self.settings.prereq_threshold, limit=3
            )
        ]
        return {
            "learner": {
                "id": learner["id"],
                "display_name": learner["display_name"],
                "age_band": learner["age_band"],
            },
            "mastery": rows,
            "next_concepts": next_concepts,
            "due_reviews": list(self._due_concepts(scope, self.clock.now())),
            "interaction_count": self.store.interaction_count(scope),
            "safety_events": [
                {
                    "ts": event.ts.isoformat(),
                    "direction": event.direction,
                    "category": event.category,
                    "blocked": event.blocked,
                }
                for event in self.store.safety_events(scope)
            ],
            "retention": self.memory.retention_summary(),
        }

    def export(self, parent: ParentScope) -> dict[str, object]:
        return self.memory.export(parent)

    def delete(self, parent: ParentScope) -> None:
        self.memory.delete(parent)

    def weekly_pdf(self, parent: ParentScope, week: int = 0) -> bytes:
        from .reports import build_weekly_report, render_pdf

        report = build_weekly_report(
            self.store,
            self.curriculum,
            parent,
            week,
            self.memory,
        )
        return render_pdf(report)
