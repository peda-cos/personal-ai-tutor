"""SQLite connection, schema, and clock helpers."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

SCHEMA = """
CREATE TABLE IF NOT EXISTS learners (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    age_band TEXT NOT NULL,
    voice_ref TEXT NOT NULL,
    token_lookup TEXT NOT NULL UNIQUE,
    token_salt TEXT NOT NULL,
    token_hash TEXT NOT NULL,
    parent_lookup TEXT NOT NULL UNIQUE,
    parent_salt TEXT NOT NULL,
    parent_hash TEXT NOT NULL,
    consent_at TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    learner_id TEXT NOT NULL REFERENCES learners(id) ON DELETE CASCADE,
    concept_id TEXT NOT NULL,
    item_json TEXT NOT NULL,
    step_index INTEGER NOT NULL DEFAULT 0,
    hint_level INTEGER NOT NULL DEFAULT 0,
    hints_used INTEGER NOT NULL DEFAULT 0,
    item_index INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    consecutive_wrong INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_learner ON sessions(learner_id, updated_at);

CREATE TABLE IF NOT EXISTS mastery (
    learner_id TEXT NOT NULL REFERENCES learners(id) ON DELETE CASCADE,
    concept_id TEXT NOT NULL,
    p REAL NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    correct INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (learner_id, concept_id)
);

CREATE TABLE IF NOT EXISTS reviews (
    learner_id TEXT NOT NULL REFERENCES learners(id) ON DELETE CASCADE,
    concept_id TEXT NOT NULL,
    stability REAL NOT NULL,
    difficulty REAL NOT NULL,
    due_at TEXT NOT NULL,
    last_review TEXT NOT NULL,
    reps INTEGER NOT NULL DEFAULT 0,
    lapses INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (learner_id, concept_id)
);

CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_id TEXT NOT NULL REFERENCES learners(id) ON DELETE CASCADE,
    session_id TEXT,
    ts TEXT NOT NULL,
    modality TEXT NOT NULL,
    learner_text TEXT NOT NULL,
    tutor_text TEXT NOT NULL,
    concept_id TEXT,
    signal TEXT NOT NULL,
    correct INTEGER NOT NULL DEFAULT 0,
    step_index INTEGER NOT NULL DEFAULT 0,
    hints_used INTEGER NOT NULL DEFAULT 0,
    mastery_after REAL,
    safety_category TEXT
);
CREATE INDEX IF NOT EXISTS interactions_learner_ts ON interactions(learner_id, ts);

CREATE TABLE IF NOT EXISTS semantic (
    learner_id TEXT NOT NULL REFERENCES learners(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (learner_id, key)
);

CREATE TABLE IF NOT EXISTS safety_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_id TEXT NOT NULL REFERENCES learners(id) ON DELETE CASCADE,
    ts TEXT NOT NULL,
    direction TEXT NOT NULL,
    category TEXT NOT NULL,
    blocked INTEGER NOT NULL,
    excerpt_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS safety_learner_ts ON safety_events(learner_id, ts);

CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    actor TEXT NOT NULL,
    learner_id TEXT,
    action TEXT NOT NULL,
    allowed INTEGER NOT NULL,
    detail TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS audit_ts ON audit(ts);
"""


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class FrozenClock:
    """Deterministic clock for tests and the efficacy simulation."""

    def __init__(self, start: datetime) -> None:
        self._now = start.astimezone(UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, **kwargs: float) -> datetime:
        from datetime import timedelta

        self._now = self._now + timedelta(**kwargs)
        return self._now

    def set(self, value: datetime) -> None:
        self._now = value.astimezone(UTC)


def to_iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def connect(path: Path | str) -> sqlite3.Connection:
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def init_db(path: Path | str) -> None:
    with connect(path) as conn:
        init_schema(conn)
