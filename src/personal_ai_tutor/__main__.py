"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from .api import TutorAPI, serve
from .config import Settings
from .curriculum import Curriculum
from .db import init_db
from .engine import TutorEngine
from .memory import MemoryService
from .safety import SafetyService
from .store import Store
from .study import ingest_csv, run_study, write_study


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    if getattr(args, "db", None):
        settings = replace(settings, db_path=Path(args.db))
    return settings


def _engine(args: argparse.Namespace, **kwargs: object) -> TutorEngine:
    return TutorEngine(_settings(args), **kwargs)  # type: ignore[arg-type]


def _cmd_init(args: argparse.Namespace) -> int:
    init_db(_settings(args).db_path)
    print(f"initialized {_settings(args).db_path}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    engine = _engine(args)
    api = TutorAPI(engine)
    server = serve(api, args.host, args.port)
    host, port = server.server_address[:2]
    print(f"Algebra Tutor listening on http://{host!s}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    engine = _engine(args)
    credentials = engine.create_learner("Demo Learner", "13_17", True)
    scope = engine.learner_scope(credentials.learner_token)
    parent = engine.parent_scope(credentials.parent_key)
    turn = engine.start_session(scope)
    print(f"[start] {turn.concept_title}")
    messages = [
        "I don't understand why 3x + 6 = 12 means x = 2",
        "6",
        "6",
        "2",
    ]
    for message in messages:
        print(f"learner> {message}")
        turn = engine.reply(scope, turn.session_id, message)
        print(f"tutor> {turn.message}")
        if turn.mastery_after is not None:
            print(
                f"[mastery] {turn.concept_id}: "
                f"{turn.mastery_before if turn.mastery_before is not None else 'prior'} "
                f"-> {turn.mastery_after:.3f}"
            )
    if args.report_out:
        path = Path(args.report_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(engine.weekly_pdf(parent, 0))
        print(f"wrote {path}")
    return 0


def _cmd_study_run(args: argparse.Namespace) -> int:
    result = run_study(
        n_per_arm=args.learners,
        seed=args.seed,
        sessions=args.sessions,
        items_per_session=args.items_per_session,
    )
    if args.out:
        write_study(result, Path(args.out))
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def _cmd_study_ingest(args: argparse.Namespace) -> int:
    result = ingest_csv(Path(args.csv))
    if args.out:
        write_study(result, Path(args.out))
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def _cmd_safety_audit(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="tutor-audit-") as tmp:
        service = SafetyService(Store(Path(tmp) / "audit.db"))
        result = service.audit_cases()
    payload = {
        "total": result.total,
        "passed": result.passed,
        "pass_rate": result.pass_rate,
        "false_positives": result.false_positives,
        "false_negatives": result.false_negatives,
        "results": [
            {
                "text": item.text,
                "expected": "allow" if item.expected else "block",
                "actual": "allow" if item.actual else "block",
                "category": item.category,
            }
            for item in result.results
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0 if result.false_negatives == 0 and result.false_positives == 0 else 1


def _cmd_report(args: argparse.Namespace) -> int:
    engine = _engine(args)
    parent = engine.parent_scope(args.parent_key)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(engine.weekly_pdf(parent, args.week))
    print(f"wrote {path}")
    return 0


def _cmd_export_cypher(args: argparse.Namespace) -> int:
    text = Curriculum.load_default().export_cypher()
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")
    else:
        print(text, end="")
    return 0


def _cmd_retention(args: argparse.Namespace) -> int:
    store = Store(_settings(args).db_path)
    counts = MemoryService(store).purge_expired()
    print(json.dumps(counts, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="personal-ai-tutor")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="create the local SQLite database")
    init.add_argument("--db")
    init.set_defaults(func=_cmd_init)

    serve_cmd = sub.add_parser("serve", help="run the local web app")
    serve_cmd.add_argument("--db")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8000)
    serve_cmd.set_defaults(func=_cmd_serve)

    demo = sub.add_parser("demo", help="run the Socratic demo transcript")
    demo.add_argument("--db")
    demo.add_argument("--report-out")
    demo.set_defaults(func=_cmd_demo)

    study = sub.add_parser("study", help="efficacy study tooling")
    study_sub = study.add_subparsers(dest="study_command", required=True)
    study_run = study_sub.add_parser("run", help="run the deterministic simulation")
    study_run.add_argument("--out")
    study_run.add_argument("--seed", type=int, default=20261123)
    study_run.add_argument("--learners", type=int, default=10)
    study_run.add_argument("--sessions", type=int, default=6)
    study_run.add_argument("--items-per-session", type=int, default=5)
    study_run.set_defaults(func=_cmd_study_run)
    study_ingest = study_sub.add_parser("ingest", help="analyze real pre/post CSV data")
    study_ingest.add_argument("--csv", required=True)
    study_ingest.add_argument("--out")
    study_ingest.set_defaults(func=_cmd_study_ingest)

    audit = sub.add_parser("safety-audit", help="run the curated safety fixture audit")
    audit.set_defaults(func=_cmd_safety_audit)

    report = sub.add_parser("report", help="write a weekly parent PDF")
    report.add_argument("--db")
    report.add_argument("--parent-key", required=True)
    report.add_argument("--week", type=int, choices=[0, 1], default=0)
    report.add_argument("--out", required=True)
    report.set_defaults(func=_cmd_report)

    cypher = sub.add_parser("export-cypher", help="export the curriculum for Neo4j")
    cypher.add_argument("--out")
    cypher.set_defaults(func=_cmd_export_cypher)

    retention = sub.add_parser("retention", help="apply the retention policy")
    retention.add_argument("--db")
    retention.set_defaults(func=_cmd_retention)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # CLI boundary: concise error, no traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
