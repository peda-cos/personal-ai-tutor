#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

uv sync --frozen
uv run ruff check src tests
uv run mypy src
uv run pytest -q

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

uv run python -m personal_ai_tutor demo \
  --db "$work/demo.db" \
  --report-out "$work/report.pdf" >"$work/demo.txt"
grep -q "You've got it" "$work/demo.txt"
test "$(head -c 5 "$work/report.pdf")" = "%PDF-"

uv run python -m personal_ai_tutor study run \
  --out "$work/study.json" \
  --seed 20261123 >/dev/null
uv run python -m personal_ai_tutor safety-audit >"$work/safety.json"
uv run python -m personal_ai_tutor export-cypher --out "$work/curriculum.cypher" >/dev/null

test "$(grep -c 'MERGE (c:Concept' "$work/curriculum.cypher")" -eq 68
test "$(grep -c '\[:PREREQ\]' "$work/curriculum.cypher")" -eq 113
uv run python -m personal_ai_tutor retention --db "$work/demo.db" >/dev/null

echo "verify.sh: OK"
