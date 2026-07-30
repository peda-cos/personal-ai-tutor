from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from personal_ai_tutor.config import Settings
from personal_ai_tutor.curriculum import Curriculum
from personal_ai_tutor.db import FrozenClock
from personal_ai_tutor.engine import TutorEngine
from personal_ai_tutor.items import ItemFactory
from personal_ai_tutor.store import Credentials, LearnerScope, ParentScope


@dataclass
class DemoLearner:
    credentials: Credentials
    scope: LearnerScope
    parent: ParentScope


@pytest.fixture(scope="session")
def curriculum() -> Curriculum:
    return Curriculum.load_default()


@pytest.fixture(scope="session")
def factory(curriculum: Curriculum) -> ItemFactory:
    return ItemFactory(curriculum)


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock(datetime(2026, 1, 1, tzinfo=UTC))


@pytest.fixture
def engine(tmp_path: Path, clock: FrozenClock) -> TutorEngine:
    return TutorEngine(
        Settings(db_path=tmp_path / "tutor.db", guard="rules", tts="tone", llm="off"),
        clock=clock,
    )


@pytest.fixture
def learner(engine: TutorEngine) -> DemoLearner:
    credentials = engine.create_learner("Test Learner", "13_17", True)
    return DemoLearner(
        credentials=credentials,
        scope=engine.learner_scope(credentials.learner_token),
        parent=engine.parent_scope(credentials.parent_key),
    )
