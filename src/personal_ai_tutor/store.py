"""Scope-enforcing SQLite persistence and credential authentication."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .db import Clock, SystemClock, connect, from_iso, init_schema, to_iso
from .fsrs import ReviewState
from .items import Item


class ConsentRequired(ValueError):
    """Raised when an under-13 learner is created without consent."""


class AuthError(PermissionError):
    """Raised when a learner token or parent key is invalid."""


class ScopeError(PermissionError):
    """Raised when a caller tries to reach another learner's data."""


@dataclass(frozen=True)
class LearnerScope:
    learner_id: str
    voice_ref: str


@dataclass(frozen=True)
class ParentScope:
    learner_id: str


@dataclass(frozen=True)
class Credentials:
    learner_id: str
    learner_token: str
    parent_key: str
    voice_ref: str


@dataclass(frozen=True)
class MasteryRow:
    concept_id: str
    p: float
    attempts: int
    correct: int
    updated_at: datetime


@dataclass(frozen=True)
class Session:
    id: str
    learner_id: str
    concept_id: str
    item: Item
    step_index: int
    hint_level: int
    hints_used: int
    item_index: int
    attempts: int
    consecutive_wrong: int
    status: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class Interaction:
    id: int
    ts: datetime
    modality: str
    learner_text: str
    tutor_text: str
    concept_id: str | None
    signal: str
    correct: bool
    step_index: int
    hints_used: int
    mastery_after: float | None
    safety_category: str | None


@dataclass(frozen=True)
class SemanticFact:
    key: str
    value: str
    confidence: float
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class SafetyEvent:
    id: int
    ts: datetime
    direction: str
    category: str
    blocked: bool
    excerpt_hash: str


@dataclass(frozen=True)
class AuditRow:
    id: int
    ts: datetime
    actor: str
    learner_id: str | None
    action: str
    allowed: bool
    detail: str


def _hash_secret(secret: str, salt_hex: str) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256", secret.encode("utf-8"), bytes.fromhex(salt_hex), 200_000
    )
    return digest.hex()


def _lookup(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def create_learner(
    store: Store,
    display_name: str,
    age_band: str,
    consent: bool,
) -> Credentials:
    if age_band not in {"under_13", "13_17", "18_plus"}:
        raise ValueError("age_band must be under_13, 13_17, or 18_plus")
    clean_name = " ".join(display_name.strip().split())[:60]
    if not clean_name:
        raise ValueError("display_name is required")
    if age_band == "under_13" and not consent:
        raise ConsentRequired("a parent or guardian must consent for under-13 learners")
    learner_id = "ln_" + secrets.token_hex(6)
    learner_token = "lt_" + secrets.token_urlsafe(32)
    parent_key = "pk_" + secrets.token_urlsafe(32)
    voice_ref = "vr_" + secrets.token_hex(8)
    token_salt = secrets.token_hex(16)
    parent_salt = secrets.token_hex(16)
    now = store.clock.now()
    with store.connection() as conn:
        conn.execute(
            """
            INSERT INTO learners (
                id, display_name, age_band, voice_ref,
                token_lookup, token_salt, token_hash,
                parent_lookup, parent_salt, parent_hash, consent_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                learner_id,
                clean_name,
                age_band,
                voice_ref,
                _lookup(learner_token),
                token_salt,
                _hash_secret(learner_token, token_salt),
                _lookup(parent_key),
                parent_salt,
                _hash_secret(parent_key, parent_salt),
                to_iso(now) if consent else None,
                to_iso(now),
            ),
        )
        conn.commit()
    return Credentials(learner_id, learner_token, parent_key, voice_ref)


def authenticate_learner(store: Store, token: str) -> LearnerScope:
    with store.connection() as conn:
        row = conn.execute(
            "SELECT * FROM learners WHERE token_lookup = ?", (_lookup(token),)
        ).fetchone()
    if row is None:
        store.audit("anonymous", None, "learner_auth_failed", False, "unknown token")
        raise AuthError("invalid learner token")
    if not hmac.compare_digest(
        _hash_secret(token, row["token_salt"]), row["token_hash"]
    ):
        store.audit("anonymous", row["id"], "learner_auth_failed", False, "bad token")
        raise AuthError("invalid learner token")
    return LearnerScope(learner_id=row["id"], voice_ref=row["voice_ref"])


def authenticate_parent(store: Store, parent_key: str) -> ParentScope:
    with store.connection() as conn:
        row = conn.execute(
            "SELECT * FROM learners WHERE parent_lookup = ?", (_lookup(parent_key),)
        ).fetchone()
    if row is None:
        store.audit("anonymous", None, "parent_auth_failed", False, "unknown key")
        raise AuthError("invalid parent key")
    if not hmac.compare_digest(
        _hash_secret(parent_key, row["parent_salt"]), row["parent_hash"]
    ):
        store.audit("anonymous", row["id"], "parent_auth_failed", False, "bad key")
        raise AuthError("invalid parent key")
    return ParentScope(learner_id=row["id"])


class Store:
    def __init__(self, db_path: Path | str, clock: Clock | None = None) -> None:
        self.path = Path(db_path)
        self.clock = clock or SystemClock()
        with self.connection() as conn:
            init_schema(conn)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = connect(self.path)
        try:
            yield conn
        finally:
            conn.close()

    def audit(
        self,
        actor: str,
        learner_id: str | None,
        action: str,
        allowed: bool,
        detail: str,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO audit (ts, actor, learner_id, action, allowed, detail) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    to_iso(self.clock.now()),
                    actor,
                    learner_id,
                    action,
                    int(allowed),
                    detail[:500],
                ),
            )
            conn.commit()

    def learner_row(self, scope: LearnerScope | ParentScope) -> dict[str, Any]:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM learners WHERE id = ?", (scope.learner_id,)
            ).fetchone()
        if row is None:
            raise ScopeError("learner not found")
        return dict(row)

    def get_session(self, scope: LearnerScope, session_id: str) -> Session | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        if row["learner_id"] != scope.learner_id:
            self.audit(
                scope.learner_id,
                scope.learner_id,
                "scope_violation",
                False,
                f"session {session_id}",
            )
            raise ScopeError("session does not belong to this learner")
        return self._session_from_row(row)

    def latest_session(self, scope: LearnerScope) -> Session | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE learner_id = ? "
                "ORDER BY updated_at DESC LIMIT 1",
                (scope.learner_id,),
            ).fetchone()
        return self._session_from_row(row) if row is not None else None

    def save_session(self, scope: LearnerScope, session: Session) -> None:
        if session.learner_id != scope.learner_id:
            self.audit(scope.learner_id, scope.learner_id, "scope_violation", False, "save session")
            raise ScopeError("session does not belong to this learner")
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    id, learner_id, concept_id, item_json, step_index, hint_level,
                    hints_used, item_index, attempts, consecutive_wrong, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    concept_id=excluded.concept_id, item_json=excluded.item_json,
                    step_index=excluded.step_index, hint_level=excluded.hint_level,
                    hints_used=excluded.hints_used, item_index=excluded.item_index,
                    attempts=excluded.attempts, consecutive_wrong=excluded.consecutive_wrong,
                    status=excluded.status, updated_at=excluded.updated_at
                """,
                (
                    session.id,
                    session.learner_id,
                    session.concept_id,
                    json.dumps(session.item.to_dict(), separators=(",", ":")),
                    session.step_index,
                    session.hint_level,
                    session.hints_used,
                    session.item_index,
                    session.attempts,
                    session.consecutive_wrong,
                    session.status,
                    to_iso(session.created_at),
                    to_iso(session.updated_at),
                ),
            )
            conn.commit()

    def _session_from_row(self, row: sqlite3.Row) -> Session:
        try:
            item = Item.from_dict(json.loads(row["item_json"]))
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            raise ScopeError("stored session item is corrupt") from exc
        return Session(
            id=row["id"],
            learner_id=row["learner_id"],
            concept_id=row["concept_id"],
            item=item,
            step_index=int(row["step_index"]),
            hint_level=int(row["hint_level"]),
            hints_used=int(row["hints_used"]),
            item_index=int(row["item_index"]),
            attempts=int(row["attempts"]),
            consecutive_wrong=int(row["consecutive_wrong"]),
            status=row["status"],
            created_at=from_iso(row["created_at"]),
            updated_at=from_iso(row["updated_at"]),
        )

    def mastery_map(self, scope: LearnerScope) -> dict[str, float]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT concept_id, p FROM mastery WHERE learner_id = ?",
                (scope.learner_id,),
            ).fetchall()
        return {row["concept_id"]: float(row["p"]) for row in rows}

    def mastery_row(self, scope: LearnerScope, concept_id: str) -> MasteryRow | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM mastery WHERE learner_id = ? AND concept_id = ?",
                (scope.learner_id, concept_id),
            ).fetchone()
        if row is None:
            return None
        return MasteryRow(
            concept_id=row["concept_id"],
            p=float(row["p"]),
            attempts=int(row["attempts"]),
            correct=int(row["correct"]),
            updated_at=from_iso(row["updated_at"]),
        )

    def set_mastery(
        self,
        scope: LearnerScope,
        concept_id: str,
        p: float,
        attempts: int,
        correct: int,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO mastery (learner_id, concept_id, p, attempts, correct, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(learner_id, concept_id) DO UPDATE SET
                    p=excluded.p, attempts=excluded.attempts,
                    correct=excluded.correct, updated_at=excluded.updated_at
                """,
                (
                    scope.learner_id,
                    concept_id,
                    max(0.0, min(1.0, p)),
                    attempts,
                    correct,
                    to_iso(self.clock.now()),
                ),
            )
            conn.commit()

    def review_map(self, scope: LearnerScope) -> dict[str, ReviewState]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM reviews WHERE learner_id = ?", (scope.learner_id,)
            ).fetchall()
        return {
            row["concept_id"]: ReviewState(
                stability=float(row["stability"]),
                difficulty=float(row["difficulty"]),
                due_at=from_iso(row["due_at"]),
                last_review=from_iso(row["last_review"]),
                reps=int(row["reps"]),
                lapses=int(row["lapses"]),
            )
            for row in rows
        }

    def set_review(self, scope: LearnerScope, concept_id: str, state: ReviewState) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO reviews (
                    learner_id, concept_id, stability, difficulty, due_at,
                    last_review, reps, lapses
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(learner_id, concept_id) DO UPDATE SET
                    stability=excluded.stability, difficulty=excluded.difficulty,
                    due_at=excluded.due_at, last_review=excluded.last_review,
                    reps=excluded.reps, lapses=excluded.lapses
                """,
                (
                    scope.learner_id,
                    concept_id,
                    state.stability,
                    state.difficulty,
                    to_iso(state.due_at),
                    to_iso(state.last_review),
                    state.reps,
                    state.lapses,
                ),
            )
            conn.commit()

    def add_interaction(
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
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO interactions (
                    learner_id, session_id, ts, modality, learner_text, tutor_text,
                    concept_id, signal, correct, step_index, hints_used,
                    mastery_after, safety_category
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scope.learner_id,
                    session_id,
                    to_iso(self.clock.now()),
                    modality,
                    learner_text[:4000],
                    tutor_text[:4000],
                    concept_id,
                    signal,
                    int(correct),
                    step_index,
                    hints_used,
                    mastery_after,
                    safety_category,
                ),
            )
            conn.commit()

    def recent_interactions(self, scope: LearnerScope, limit: int = 5) -> list[Interaction]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM interactions WHERE learner_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (scope.learner_id, limit),
            ).fetchall()
        return [self._interaction_from_row(row) for row in reversed(rows)]

    def count_interactions(
        self,
        scope: LearnerScope,
        *,
        concept_id: str | None = None,
        correct: bool | None = None,
        modality: str | None = None,
        signal: str | None = None,
    ) -> int:
        clauses = ["learner_id = ?"]
        params: list[object] = [scope.learner_id]
        if concept_id is not None:
            clauses.append("concept_id = ?")
            params.append(concept_id)
        if correct is not None:
            clauses.append("correct = ?")
            params.append(int(correct))
        if modality is not None:
            clauses.append("modality = ?")
            params.append(modality)
        if signal is not None:
            clauses.append("signal = ?")
            params.append(signal)
        sql = "SELECT COUNT(*) AS n FROM interactions WHERE " + " AND ".join(clauses)
        with self.connection() as conn:
            row = conn.execute(sql, tuple(params)).fetchone()
        return int(row["n"])

    def interaction_count(self, scope: LearnerScope) -> int:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM interactions WHERE learner_id = ?",
                (scope.learner_id,),
            ).fetchone()
        return int(row["n"])

    def all_interactions(self, scope: LearnerScope, limit: int = 1000) -> list[Interaction]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM interactions WHERE learner_id = ? "
                "ORDER BY id ASC LIMIT ?",
                (scope.learner_id, limit),
            ).fetchall()
        return [self._interaction_from_row(row) for row in rows]

    def _interaction_from_row(self, row: sqlite3.Row) -> Interaction:
        return Interaction(
            id=int(row["id"]),
            ts=from_iso(row["ts"]),
            modality=row["modality"],
            learner_text=row["learner_text"],
            tutor_text=row["tutor_text"],
            concept_id=row["concept_id"],
            signal=row["signal"],
            correct=bool(row["correct"]),
            step_index=int(row["step_index"]),
            hints_used=int(row["hints_used"]),
            mastery_after=row["mastery_after"],
            safety_category=row["safety_category"],
        )

    def semantic_map(self, scope: LearnerScope) -> dict[str, SemanticFact]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM semantic WHERE learner_id = ?", (scope.learner_id,)
            ).fetchall()
        return {
            row["key"]: SemanticFact(
                key=row["key"],
                value=row["value"],
                confidence=float(row["confidence"]),
                created_at=from_iso(row["created_at"]),
                updated_at=from_iso(row["updated_at"]),
            )
            for row in rows
        }

    def upsert_semantic(
        self, scope: LearnerScope, key: str, value: str, confidence: float
    ) -> None:
        now = to_iso(self.clock.now())
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO semantic (learner_id, key, value, confidence, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(learner_id, key) DO UPDATE SET
                    value=excluded.value, confidence=excluded.confidence,
                    updated_at=excluded.updated_at
                """,
                (scope.learner_id, key, value, max(0.0, min(1.0, confidence)), now, now),
            )
            conn.commit()

    def add_safety_event(
        self,
        scope: LearnerScope,
        direction: str,
        category: str,
        blocked: bool,
        excerpt: str,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO safety_events (
                    learner_id, ts, direction, category, blocked, excerpt_hash
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    scope.learner_id,
                    to_iso(self.clock.now()),
                    direction,
                    category,
                    int(blocked),
                    hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
                ),
            )
            conn.commit()

    def safety_events(self, scope: LearnerScope) -> list[SafetyEvent]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM safety_events WHERE learner_id = ? ORDER BY id DESC",
                (scope.learner_id,),
            ).fetchall()
        return [
            SafetyEvent(
                id=int(row["id"]),
                ts=from_iso(row["ts"]),
                direction=row["direction"],
                category=row["category"],
                blocked=bool(row["blocked"]),
                excerpt_hash=row["excerpt_hash"],
            )
            for row in rows
        ]

    def audit_rows(self) -> list[AuditRow]:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM audit ORDER BY id DESC").fetchall()
        return [
            AuditRow(
                id=int(row["id"]),
                ts=from_iso(row["ts"]),
                actor=row["actor"],
                learner_id=row["learner_id"],
                action=row["action"],
                allowed=bool(row["allowed"]),
                detail=row["detail"],
            )
            for row in rows
        ]

    def export(self, parent: ParentScope) -> dict[str, Any]:
        scope = LearnerScope(parent.learner_id, "")
        learner = self.learner_row(parent)
        mastery = [
            {
                "concept_id": row["concept_id"],
                "p": row["p"],
                "attempts": row["attempts"],
                "correct": row["correct"],
                "updated_at": row["updated_at"],
            }
            for row in self._rows(
                "SELECT * FROM mastery WHERE learner_id = ? ORDER BY concept_id",
                (parent.learner_id,),
            )
        ]
        reviews = [
            {
                "concept_id": row["concept_id"],
                "stability": row["stability"],
                "difficulty": row["difficulty"],
                "due_at": row["due_at"],
                "last_review": row["last_review"],
                "reps": row["reps"],
                "lapses": row["lapses"],
            }
            for row in self._rows(
                "SELECT * FROM reviews WHERE learner_id = ? ORDER BY concept_id",
                (parent.learner_id,),
            )
        ]
        return {
            "learner": {
                "id": learner["id"],
                "display_name": learner["display_name"],
                "age_band": learner["age_band"],
                "created_at": learner["created_at"],
                "consent_at": learner["consent_at"],
            },
            "mastery": mastery,
            "reviews": reviews,
            "interactions": [
                {
                    "id": item.id,
                    "ts": to_iso(item.ts),
                    "modality": item.modality,
                    "learner_text": item.learner_text,
                    "tutor_text": item.tutor_text,
                    "concept_id": item.concept_id,
                    "signal": item.signal,
                    "correct": item.correct,
                    "mastery_after": item.mastery_after,
                }
                for item in self.all_interactions(scope, 10000)
            ],
            "semantic": [
                {
                    "key": fact.key,
                    "value": fact.value,
                    "confidence": fact.confidence,
                    "updated_at": to_iso(fact.updated_at),
                }
                for fact in self.semantic_map(scope).values()
            ],
            "safety_events": [
                {
                    "ts": to_iso(event.ts),
                    "direction": event.direction,
                    "category": event.category,
                    "blocked": event.blocked,
                }
                for event in self.safety_events(scope)
            ],
            "retention": {
                "interactions_days": 365,
                "semantic_days": 365,
                "safety_days": 365,
                "audit_days": 730,
                "idle_session_days": 30,
                "inactive_learner_days": 365,
            },
        }

    def _rows(self, sql: str, params: tuple[object, ...]) -> list[sqlite3.Row]:
        with self.connection() as conn:
            return conn.execute(sql, params).fetchall()

    def delete(self, parent: ParentScope) -> None:
        with self.connection() as conn:
            for table in (
                "sessions",
                "mastery",
                "reviews",
                "interactions",
                "semantic",
                "safety_events",
                "audit",
            ):
                conn.execute(f"DELETE FROM {table} WHERE learner_id = ?", (parent.learner_id,))
            conn.execute("DELETE FROM learners WHERE id = ?", (parent.learner_id,))
            conn.commit()

    def purge(self, now: datetime | None = None) -> dict[str, int]:
        current = now or self.clock.now()
        interaction_cutoff = to_iso(current - timedelta(days=365))
        audit_cutoff = to_iso(current - timedelta(days=730))
        session_cutoff = to_iso(current - timedelta(days=30))
        learner_cutoff = to_iso(current - timedelta(days=365))
        counts = {
            "interactions": 0,
            "semantic": 0,
            "safety_events": 0,
            "audit": 0,
            "sessions": 0,
            "learners": 0,
        }
        with self.connection() as conn:
            counts["interactions"] = conn.execute(
                "DELETE FROM interactions WHERE ts < ?", (interaction_cutoff,)
            ).rowcount
            counts["semantic"] = conn.execute(
                "DELETE FROM semantic WHERE updated_at < ?", (interaction_cutoff,)
            ).rowcount
            counts["safety_events"] = conn.execute(
                "DELETE FROM safety_events WHERE ts < ?", (interaction_cutoff,)
            ).rowcount
            counts["audit"] = conn.execute(
                "DELETE FROM audit WHERE ts < ?", (audit_cutoff,)
            ).rowcount
            counts["sessions"] = conn.execute(
                "DELETE FROM sessions WHERE updated_at < ?", (session_cutoff,)
            ).rowcount
            stale = conn.execute(
                """
                SELECT l.id FROM learners l
                WHERE l.created_at < ?
                  AND NOT EXISTS (
                      SELECT 1 FROM interactions i
                      WHERE i.learner_id = l.id AND i.ts >= ?
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM sessions s
                      WHERE s.learner_id = l.id AND s.updated_at >= ?
                  )
                """,
                (learner_cutoff, learner_cutoff, learner_cutoff),
            ).fetchall()
            for row in stale:
                for table in (
                    "sessions",
                    "mastery",
                    "reviews",
                    "interactions",
                    "semantic",
                    "safety_events",
                    "audit",
                ):
                    conn.execute(f"DELETE FROM {table} WHERE learner_id = ?", (row["id"],))
                conn.execute("DELETE FROM learners WHERE id = ?", (row["id"],))
                counts["learners"] += 1
            conn.commit()
        return counts
