from __future__ import annotations

from personal_ai_tutor.pdf import PDFBuilder
from personal_ai_tutor.reports import build_weekly_report, render_pdf


def test_weekly_report_and_pdf(engine, learner) -> None:
    turn = engine.start_session(learner.scope)
    engine.reply(learner.scope, turn.session_id, "I'm stuck")
    report = build_weekly_report(
        engine.store, engine.curriculum, learner.parent, 0, engine.memory
    )
    assert report.learner_name == "Test Learner"
    assert report.week_label == "Week 1"
    assert report.interactions >= 1
    assert report.retention["interactions_days"] == 365
    pdf = render_pdf(report)
    assert pdf.startswith(b"%PDF-1.4")
    assert pdf.rstrip().endswith(b"%%EOF")
    assert b"Weekly Progress" in pdf
    assert b"Test Learner" in pdf
    assert pdf.count(b"/Type /Page ") >= 1


def test_reports_do_not_leak_other_learners(engine, learner) -> None:
    other = engine.create_learner("Other Learner", "13_17", True)
    turn = engine.start_session(learner.scope)
    engine.reply(learner.scope, turn.session_id, "I'm stuck")
    report = build_weekly_report(
        engine.store, engine.curriculum, learner.parent, 0, engine.memory
    )
    pdf = render_pdf(report)
    assert b"Other Learner" not in pdf
    assert other.learner_id.encode() not in pdf


def test_week_two_report_is_valid(engine, learner) -> None:
    report = build_weekly_report(
        engine.store, engine.curriculum, learner.parent, 1, engine.memory
    )
    assert report.week_label == "Week 2"
    assert render_pdf(report).startswith(b"%PDF")


def test_pdf_writes_one_footer_per_page() -> None:
    builder = PDFBuilder("Long report")
    for index in range(80):
        builder.text(f"Row {index}: mastery trajectory and practice notes.")
    pdf = builder.render()
    pages = pdf.count(b"/Type /Page ")
    assert pages > 1
    assert pdf.count(b"simulated/offline report") == pages
