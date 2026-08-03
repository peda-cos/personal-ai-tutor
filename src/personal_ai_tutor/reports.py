"""Weekly parent-facing progress reports and PDF rendering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from .curriculum import Curriculum
from .db import from_iso, to_iso
from .memory import MemoryService
from .pdf import PDFBuilder
from .store import LearnerScope, ParentScope, Store


@dataclass(frozen=True)
class TopicSummary:
    concept_id: str
    title: str
    attempts: int
    correct: int


@dataclass(frozen=True)
class MasterySummary:
    concept_id: str
    title: str
    mastery: float


@dataclass(frozen=True)
class WeeklyReport:
    learner_id: str
    learner_name: str
    week_label: str
    week_window: str
    interactions: int
    topics: tuple[TopicSummary, ...]
    mastery: tuple[MasterySummary, ...]
    trajectory: tuple[tuple[str, float], ...]
    next_steps: tuple[str, ...]
    safety_events: tuple[tuple[str, str, bool], ...]
    retention: dict[str, object]


def build_weekly_report(
    store: Store,
    curriculum: Curriculum,
    parent: ParentScope,
    week: int,
    memory: MemoryService | None = None,
) -> WeeklyReport:
    learner = store.learner_row(parent)
    scope = LearnerScope(parent.learner_id, "")
    start = from_iso(str(learner["created_at"])) + timedelta(days=7 * week)
    end = start + timedelta(days=7)
    interactions = [
        item
        for item in store.all_interactions(scope, 10000)
        if start <= item.ts < end
    ]
    topics: dict[str, TopicSummary] = {}
    for item in interactions:
        if not item.concept_id:
            continue
        concept = curriculum.concepts.get(item.concept_id)
        if concept is None:
            continue
        existing = topics.get(item.concept_id)
        topics[item.concept_id] = TopicSummary(
            concept_id=item.concept_id,
            title=concept.title,
            attempts=(existing.attempts if existing else 0) + 1,
            correct=(existing.correct if existing else 0) + (1 if item.correct else 0),
        )
    mastery_map = store.mastery_map(scope)
    mastery = tuple(
        MasterySummary(
            concept_id=cid,
            title=curriculum.concepts[cid].title,
            mastery=value,
        )
        for cid, value in sorted(mastery_map.items(), key=lambda pair: pair[1])
        if cid in curriculum.concepts
    )
    trajectory = tuple(
        (to_iso(item.ts), item.mastery_after)
        for item in interactions
        if item.mastery_after is not None
    )
    next_steps = tuple(
        concept.title
        for concept in curriculum.next_concepts(
            mastery_map, limit=3
        )
    )
    safety = tuple(
        (to_iso(event.ts), event.category, event.blocked)
        for event in store.safety_events(scope)
        if start <= event.ts < end
    )
    retention = (
        memory.retention_summary()
        if memory is not None
        else {
            "interactions_days": 365,
            "semantic_days": 365,
            "safety_events_days": 365,
            "audit_days": 730,
            "idle_session_days": 30,
            "inactive_learner_days": 365,
            "certification": "COPPA-aware design, not a legal compliance certification",
        }
    )
    return WeeklyReport(
        learner_id=str(learner["id"]),
        learner_name=str(learner["display_name"]),
        week_label=f"Week {week + 1}",
        week_window=f"day {7 * week} to day {7 * week + 7}",
        interactions=len(interactions),
        topics=tuple(topics.values()),
        mastery=mastery,
        trajectory=trajectory,
        next_steps=next_steps,
        safety_events=safety,
        retention=retention,
    )


def render_pdf(report: WeeklyReport) -> bytes:
    builder = PDFBuilder(f"Weekly progress - {report.learner_name} - {report.week_label}")
    builder.heading("Personal AI Tutor - Weekly Progress")
    builder.text(f"Learner: {report.learner_name}", bold=True, size=12)
    builder.text(f"{report.week_label} ({report.week_window})")
    builder.text(
        "This report summarizes practice, mastery estimates, and safety signals. "
        "It is a simulated/offline demo unless real learner data was ingested."
    )
    builder.gap(6)
    builder.rule()
    builder.heading("Activity")
    builder.text(f"Tutor interactions this week: {report.interactions}")
    if report.topics:
        for topic in report.topics:
            builder.text(
                f"{topic.title}: {topic.attempts} attempts, {topic.correct} correct"
            )
    else:
        builder.text("No interactions were recorded in this window.")
    builder.gap(6)
    builder.heading("Mastery")
    if report.mastery:
        for row in report.mastery:
            builder.bar(row.title, row.mastery)
    else:
        builder.text("No mastery estimates yet.")
    builder.gap(6)
    builder.heading("Mastery trajectory")
    if report.trajectory:
        for timestamp, value in report.trajectory[-12:]:
            builder.text(f"{timestamp[:10]}: {value:.0%}")
    else:
        builder.text("No mastery snapshots in this window.")
    builder.gap(6)
    builder.heading("Recommended next steps")
    if report.next_steps:
        for title in report.next_steps:
            builder.text(f"- {title}")
    else:
        builder.text("Continue with the current concept and review queue.")
    builder.gap(6)
    builder.heading("Safety and privacy")
    if report.safety_events:
        for timestamp, category, blocked in report.safety_events:
            builder.text(
                f"{timestamp[:10]}: {category} ({'blocked' if blocked else 'redacted'})"
            )
    else:
        builder.text("No safety events in this window.")
    builder.text(
        "Retention: interactions and semantic memory 365 days; safety events 365 days; "
        "audit 730 days; inactive learner data purged after 365 days. "
        "Parents can export or delete learner data at any time."
    )
    builder.text("COPPA-aware design, not a legal compliance certification.")
    return builder.render()
