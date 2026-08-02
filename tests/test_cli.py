from __future__ import annotations

from pathlib import Path

from personal_ai_tutor.__main__ import main


def test_cli_init_demo_and_report(tmp_path: Path, capsys) -> None:
    db = tmp_path / "cli.db"
    assert main(["init", "--db", str(db)]) == 0
    report = tmp_path / "report.pdf"
    assert main(["demo", "--db", str(db), "--report-out", str(report)]) == 0
    output = capsys.readouterr().out
    assert "tutor>" in output
    assert report.read_bytes().startswith(b"%PDF")


def test_cli_cypher_and_audit(tmp_path: Path, capsys) -> None:
    cypher = tmp_path / "curriculum.cypher"
    assert main(["export-cypher", "--out", str(cypher)]) == 0
    assert "MERGE (c:Concept" in cypher.read_text(encoding="utf-8")
    assert main(["safety-audit"]) == 0
    assert '"false_negatives": 0' in capsys.readouterr().out


def test_cli_study_run_and_ingest(tmp_path: Path, capsys) -> None:
    output = tmp_path / "study.json"
    assert (
        main(
            [
                "study",
                "run",
                "--learners",
                "2",
                "--out",
                str(output),
            ]
        )
        == 0
    )
    assert output.exists()
    assert '"simulated": true' in output.read_text(encoding="utf-8")
    csv = tmp_path / "real.csv"
    csv.write_text(
        "learner_id,pre,post,arm\na,10,20,adaptive\nb,11,21,adaptive\n"
        "c,9,12,baseline\nd,10,13,baseline\n",
        encoding="utf-8",
    )
    result = tmp_path / "real.json"
    assert main(["study", "ingest", "--csv", str(csv), "--out", str(result)]) == 0
    assert '"simulated": false' in result.read_text(encoding="utf-8")


def test_cli_retention(tmp_path: Path) -> None:
    db = tmp_path / "retention.db"
    assert main(["retention", "--db", str(db)]) == 0
