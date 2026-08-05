# Personal AI Tutor — adaptive algebra with memory and safety

A subject-specific K–12 algebra tutor with a Socratic step policy, Bayesian knowledge tracing
(BKT), FSRS-6 spaced repetition, a 68-concept prerequisite graph, learner-scoped episodic and
semantic memory, age-appropriate safety filtering, voice and photo-math input, a reproducible
simulated efficacy study, and weekly PDF progress reports.

The whole core runs offline. Paid or GPU-backed services (Claude, Neo4j, LiveKit, Whisper,
Cartesia, dots.mocr/PaliGemma, Llama Guard 4) sit behind optional adapters; the learner model,
graph engine, grading, memory, safety baseline, study, and PDF reports are local implementations.

## Architecture

```
learner browser (vanilla JS, no build step)
  |
  +-- text ----------> TutorAPI (stdlib http.server, auth, body limits)
  +-- photo ---------> PNG/PGM decode -> synthetic OCR or HTTP VLM
  +-- voice ---------> browser ASR / whisper.cpp -> VoiceSession (barge-in)
                         |
                         v
                  TutorEngine
                    +-- SafetyService: rules, PII redaction, output guard,
                    |     optional Llama Guard 4 (fail-closed)
                    +-- SocraticPolicy: LangGraph
                    |     read_signal -> select_concept -> scaffold -> update_mastery
                    +-- BKT mastery + FSRS-6 review scheduling (SQLite)
                    +-- MemoryService: episodic/semantic, learner scope, retention
                    +-- Curriculum: 68-concept algebra DAG + OpenStax OER links
                    +-- PDF weekly report
                         |
                         v
                  SQLite (WAL, hashed credentials, scoped access)
```

The subject is algebra: it provides a deeper prerequisite graph than intro Python and makes the
photo-math path meaningful. The web app is vanilla JavaScript served by Python, not a second
TypeScript port, because the capstone languages list Python first and the brief asks for one
primary-language implementation.

## Quickstart

Prerequisites: Python 3.14 and [uv](https://docs.astral.sh/uv/). No global installs.

```bash
# clone or copy this repository, then:
cd personal-ai-tutor
./verify.sh                    # installs into .venv, runs lint, typecheck, tests, CLI checks
uv run python -m personal_ai_tutor serve --port 8000
# open http://127.0.0.1:8000
```

The first `uv sync` needs network access for pinned wheels; after that the tutor, tests, demo,
study, safety audit, and PDF generation run offline. `verify.sh` is safe to run repeatedly and
uses temporary databases.

Useful commands:

```bash
uv run python -m personal_ai_tutor demo --db /tmp/demo.db --report-out /tmp/report.pdf
uv run python -m personal_ai_tutor study run --seed 20261123 --out /tmp/study.json
uv run python -m personal_ai_tutor study ingest --csv /path/to/pre-post.csv
uv run python -m personal_ai_tutor safety-audit
uv run python -m personal_ai_tutor export-cypher --out /tmp/curriculum.cypher
uv run python -m personal_ai_tutor retention --db /tmp/demo.db
```

The browser app creates a learner and shows a learner token and parent key once. The learner
token authenticates tutoring; the parent key opens progress, PDF, export, and deletion. The
token is kept in `sessionStorage`; the parent key is not persisted. This is a local demo
credential flow, not a production identity system.

## What runs offline vs. real providers

| Capability | Offline default (tested) | Real adapter (documented, not exercised by tests) | Configuration |
|---|---|---|---|
| Tutor policy | Deterministic Socratic templates + LangGraph | Claude Messages API scaffold rewrite with prompt caching | `TUTOR_LLM=claude`, `ANTHROPIC_API_KEY`, `TUTOR_CLAUDE_MODEL` |
| Curriculum graph | In-process DAG + SQLite persistence | Neo4j import from generated Cypher | `export-cypher`, then `cypher-shell -f curriculum.cypher` |
| ASR | Scripted fixtures / browser speech recognition | whisper.cpp CLI (`whisper-cli -m … -f … -oj`) | `TUTOR_ASR=whisper`, `WHISPER_BIN`, `WHISPER_MODEL` |
| TTS | Deterministic tone WAV (not speech) | Cartesia `/tts/bytes` (current Sonic model) | `TUTOR_TTS=cartesia`, `CARTESIA_API_KEY`, `CARTESIA_VOICE_ID`, `CARTESIA_MODEL` |
| Photo OCR | Built-in clean-render OCR (PNG/PGM) | OpenAI-compatible VLM endpoint (dots.mocr or PaliGemma serving) | `TUTOR_OCR=http`, `TUTOR_OCR_URL`, `TUTOR_OCR_MODEL`, `TUTOR_OCR_API_KEY` |
| Safety | Age-appropriate rules + PII redaction + output guard | Llama Guard 4 classifier over an OpenAI-compatible endpoint | `TUTOR_GUARD=llama_guard`, `TUTOR_GUARD_URL`, `TUTOR_GUARD_MODEL` |
| Voice transport | `VoiceSession` event loop with real barge-in semantics | LiveKit worker is not bundled; wire the event loop to a transport | see limitations |
| Efficacy | Simulated learners, fixed seed | Real pre/post CSV ingestion | `study ingest --csv …` |

Provider version notes: the capstone names Sonic-2, but Cartesia's current documentation lists
sonic-2 as sunset, so the adapter defaults to `sonic-3.6` and exposes the model ID through
`CARTESIA_MODEL`. dots.ocr was renamed dots.mocr; the HTTP adapter takes the model name from
env. Claude model IDs also drift, so `TUTOR_CLAUDE_MODEL` is required rather than hardcoded.
The adapters are small request/response boundaries; provider docs are linked inline below.

## Socratic tutoring

Every generated item is a small step plan. The learner sees the problem and one leading
question, not the solution. The policy reads the answer as correct, incorrect, stuck, skip, or
off-topic, then chooses a scaffold. Hints are three leading questions followed by a worked
example with different numbers. The final answer is never stated before the learner supplies
it. If a learner asks for the answer repeatedly, the tutor refuses and continues scaffolding.
An item is abandoned after a bounded number of turns rather than looping forever.

The demo transcript from `python -m personal_ai_tutor demo` (local, deterministic):

```
learner> I don't understand why 3x + 6 = 12 means x = 2
tutor>   Let's look at Solve 3x + 6 = 12. What number would you subtract on both sides to start?
learner> 6
tutor>   Nice work. After that step, what is the variable term equal to?
learner> 6
tutor>   Nice work. What is x?
learner> 2
tutor>   You've got it: 2. That is correct.
         Next: Solve 4x + 8 = 32. What number would you subtract from both sides to start?
[mastery] two_step_equations: prior -> 0.718
```

The policy is a LangGraph `StateGraph` with the four nodes from the capstone. Node bodies are
pure functions over the current item, mastery estimate, due reviews, and text; persistence is
handled by the engine after the graph returns. This keeps the policy unit-testable and makes
the graph replaceable without changing the learner model.

## Learner model

**Bayesian knowledge tracing** tracks one mastery probability per concept:

$$P(L_n \mid \text{correct}) = \frac{P(L_{n-1})(1-S)}{P(L_{n-1})(1-S)+(1-P(L_{n-1}))G}$$

$$P(L_n \mid \text{incorrect}) = \frac{P(L_{n-1})S}{P(L_{n-1})S+(1-P(L_{n-1}))(1-G)}$$

followed by the learning transition $P(L_n)=P(L_n\mid\text{obs})+(1-P(L_n\mid\text{obs}))T$.
The implementation uses concept-specific priors based on difficulty, `S=0.10`, `G=0.20`, and
`T=0.12`; item completion is the observation. Help requests do not count as incorrect.

**FSRS-6** schedules reviews using the official 21 weights and DSR equations. The scheduler
returns stability, difficulty, due date, reps, and lapses per concept. It is a scheduler, not an
optimizer: no per-user weight fitting is performed. FSRS-6 was chosen over FSRS-7 because the
capstone cites the 2024-era FSRS and FSRS-6 has compact, auditable equations.

The next concept is the eligible frontier concept with the lowest mastery, weighted by
difficulty and ID for deterministic tie-breaking. Prerequisites must reach 0.70 mastery.
Due reviews take priority. A concept advances when mastery reaches 0.80; learners with low
mastery usually need several items to get there. Three consecutive struggle turns can move the
learner to a weak prerequisite.

## Curriculum graph

- **68 concepts**, **113 prerequisite edges**, **8 strands**, maximum depth 15.
- Runs from the number line and integer operations through expressions, linear equations,
  inequalities, graphing, systems, exponents, polynomials, factoring, quadratics, rational
  expressions, functions, and sequences.
- Each concept carries a short original lesson, difficulty, family/mode for the item bank, and
  an OpenStax Elementary/Intermediate Algebra 2e section link. No OpenStax text is copied;
  the links are OER references and the lesson summaries are original.
- The graph is validated for duplicate IDs, unknown prerequisites, cycles, missing text, and
  non-topological ordering. `export-cypher` emits idempotent `MERGE` statements for a Neo4j
  instance; the app itself uses the in-process graph.

## Memory, privacy, and COPPA-aware retention

Every interaction is written to a learner-scoped episodic store. Three mistakes on a concept
promote a semantic fact (`mistake:<concept>`); three voice/photo turns promote a preferred
modality. The tutor retrieves only its own learner's facts and at most five recent interactions.
It never injects another learner's text.

Credentials are high-entropy tokens stored as SHA-256 lookup hashes plus PBKDF2-HMAC-SHA256
hashes with per-row salts. All reads/writes go through `LearnerScope`; a cross-learner session
access attempts an audit record and fails. Parent operations go through `ParentScope` after
parent-key verification. PII in learner text is redacted before storage, and safety events store
only a category and an excerpt hash.

Retention policy:

- interactions, semantic facts, and safety events: 365 days;
- audit records: 730 days;
- idle sessions: 30 days;
- inactive learners: 365 days (full cascade);
- parents can export JSON or delete all learner data at any time.

The engine applies this purge at startup, and `python -m personal_ai_tutor retention` reruns it on
demand. This is a COPPA-aware design with a written retention policy and parent
access/deletion surface. It is not a legal compliance certification. The under-13 consent
checkbox is a demo, not verifiable parental consent.

## Safety

The local guard covers self-harm, sexual content, violence, weapons, drugs, hate, privacy/PII,
prompt-injection attempts, and specialized medical/legal advice. Self-harm gets a supportive
response that directs the learner to a trusted adult or local emergency services and deliberately
does not continue the math. Other blocked categories get a refusal and a redirect. PII is
redacted and logged without the raw value. Every tutor response also passes the output guard.

The curated audit fixture contains 24 unsafe prompts and 20 benign/PII cases. Measured locally
with `safety-audit`: **44/44 correct, 0 false positives, 0 false negatives**. This is a curated
fixture, not a general safety benchmark. Llama Guard 4 is an optional classifier adapter; if it
is enabled and unreachable, the default policy is fail-closed. The adapter's category mapping is
covered with an injected transport in tests; model weights were not run locally.

## Multimodal input

- **Text:** typed answers and pasted equations. An equation in a learner message starts a new
  multi-step problem; a bare `x = 2` is treated as an answer, not a new problem.
- **Photo:** PNG/PGM decode (PNG filters 0–4, RGB/RGBA/gray), thresholding, column segmentation,
  and nearest-neighbor matching against a built-in 5×7 glyph font. This is real image processing
  for clean printed renders, not a fixture lookup. The HTTP adapter sends the PNG to a
  dots.mocr/PaliGemma-compatible endpoint for real photos. Low-confidence images fail with a
  clear message instead of a guessed equation.
- **Voice:** a transport-agnostic event loop with partial/final speech events and explicit
  barge-in; `ToneTts` produces deterministic WAV chunks, `whisper.cpp` is the ASR adapter, and
  Cartesia is the real TTS adapter. The browser uses SpeechRecognition/SpeechSynthesis when
  available and falls back to typed transcripts. Cross-learner voice re-ingestion is rejected
  on speaker-label mismatch, audited, and surfaced as a safety event.

The tested OCR cases are `3x+6=12`, `x^2=16`, `2(x+1)=8`, and `5x-3=2x+9`, all recognized at
1.0 confidence from clean renders. Real photos are an explicit provider boundary.

## Efficacy study

The harness runs the real policy, BKT, FSRS, safety, and SQLite engine for ten simulated
learners per arm. The only manipulated variable is concept selection: the adaptive arm uses
mastery/due/prerequisite logic; the baseline follows a fixed topological order. Pre/post draws
are paired by learner so the comparison is not dominated by sampling noise. Sessions occur on
simulated days 0, 2, 4, 7, 9, and 13, with five solved items per session. The pre/post test is
30 items over the 31-concept linear-equations unit and its prerequisites.

**Measured locally** with:

```bash
uv run python -m personal_ai_tutor study run --seed 20261123 --out /tmp/study.json
```

| Arm | n | Pre mean /30 | Post mean /30 | Gain | 95% bootstrap CI |
|---|---:|---:|---:|---:|---|
| Adaptive BKT selection | 10 | 11.5 | 19.5 | **+8.0** | [6.9, 9.2] |
| Fixed topological baseline | 10 | 11.5 | 15.2 | **+3.7** | [2.2, 5.4] |

**Delta = +4.3 points** (95% CI [2.3, 6.2]); Cohen's *d* = 1.84. Seed `20261123`. Runtime on
the local machine was about 8 seconds. This is simulated evidence that the adaptive sequencing
mechanism can outperform linear sequencing under the stated learner model. It is **not**
evidence of human learning gains. Real study data can be analyzed with
`study ingest --csv pre-post.csv` (required columns: `learner_id,pre,post,arm`).

## Weekly progress reports

`build_weekly_report` summarizes the learner's week: interactions, topics with attempts and
correct counts, mastery estimates, mastery trajectory, recommended next concepts, safety event
categories, and the retention policy. `pdf.py` is a small dependency-free PDF 1.4 writer; it
draws text, rules, mastery bars, and page numbers. Reports contain no raw conversation text.
The parent dashboard downloads week 1 through `/api/report.pdf?week=0`; week 2 is available at
`week=1` and via `python -m personal_ai_tutor report`.

## Verification

`./verify.sh` performs, in order:

1. `uv sync --frozen` (pinned, local `.venv`);
2. `ruff check src tests`;
3. `mypy src` (21 source files);
4. `pytest -q` (99 tests);
5. a deterministic CLI demo and PDF header check;
6. a full simulated study, safety audit, and Cypher export with 68 nodes/113 edges;
7. a retention-policy run.

The script exits 0, uses only a temporary directory, and is safe to run twice. Tests make no
external network calls: API tests call `TutorAPI.handle` in-process and provider tests inject
fake transports. A clean-copy run of `./verify.sh` was also used as the quickstart check.

Measured locally with dependencies cached: verification completes in roughly 22 seconds. The test suite
covers curriculum validation, algebraic equivalence, all item families, BKT/FSRS equations,
scoped persistence, retention boundaries, PII redaction, safety fixtures, policy behavior,
engine flows, voice barge-in, OCR, study statistics, PDF reports, the in-process API, and CLI.

## Limitations and explicit scope boundaries

- **Simulated efficacy:** no human learners were available. The study is a mechanism harness;
  it must not be cited as a human learning result. The CSV ingester is the real-study path.
- **OCR:** the built-in recognizer works on clean renders from its own font. Real photographs
  need the HTTP VLM adapter and a GPU/paid model; no photo accuracy is claimed.
- **Voice:** Cartesia and whisper.cpp adapters are implemented and tested at the request/response
  boundary, but the offline default TTS is a tone, not speech. LiveKit transport is not bundled;
  the tested voice core is the event loop with barge-in. README instructions describe the
  transport boundary rather than shipping untested LiveKit code.
- **Graph backend:** the app uses an in-process graph plus SQLite; the Neo4j adapter is a Cypher
  export, not a live driver. Import it with `cypher-shell`.
- **Safety:** the rule pack is a baseline; Llama Guard 4 is optional and was not executed with
  weights locally. The audit is a curated fixture, not a general benchmark. The project is
  COPPA-aware, not COPPA-certified.
- **Identity:** learner tokens and parent keys are high-entropy local credentials; there is no
  account recovery, email verification, rate limiting, or production consent verification.
- **Not implemented:** the capstone's Spanish language-switch exercise and the modality
  convergence experiment; neither is a Build It step, and both would add unrelated scope.

## Design decisions

- **Algebra, not intro Python:** richer prerequisites and a meaningful equation-photo path.
- **Python + vanilla JS:** one primary implementation; no duplicate TypeScript learner model.
- **Deterministic policy by default:** tests and the study never require an API key. Claude is
  a validated optional scaffold layer, not the source of truth for correctness or safety.
- **Local graph + Cypher export:** testable offline; Neo4j remains a deployment choice.
- **FSRS-6:** the stable 2024-era DSR algorithm referenced by the capstone; no optimizer.
- **Simulated study with paired draws:** honest about simulation while reducing noise; real CSV
  ingestion is provided.
- **Fail-closed optional guard:** a configured safety provider that is unavailable does not
  silently allow content.

## References

1. Capstone specification — https://aiengineeringfromscratch.com/learn/capstone-projects/personal-ai-tutor
2. LASER Institute, *Module 1: Bayesian Knowledge Tracing* — https://laser-institute.github.io/knowledge-tracing/module-1/kt-1-conceptual-overview.html
3. Open Spaced Repetition, *The FSRS Algorithm* — https://github.com/open-spaced-repetition/awesome-fsrs/wiki/The-Algorithm
4. Meta, *Llama Guard 4 Model Card* — https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard4/12B/MODEL_CARD.md
5. 16 CFR § 312.10, *Data retention and deletion requirements* — https://www.law.cornell.edu/cfr/text/16/312.10

Additional sources used inline: OpenStax [Elementary Algebra 2e](https://openstax.org/books/elementary-algebra-2e/pages/1-introduction)
and [Intermediate Algebra 2e](https://openstax.org/books/intermediate-algebra-2e/pages/1-introduction)
(CC BY-NC-SA; links only, no text copied); [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview);
[FSRS-6 equations and weights](https://open-spaced-repetition.github.io/ts-fsrs/guide/model/fsrs-6);
[Cartesia TTS Bytes](https://docs.cartesia.ai/api-reference/tts/bytes) and
[Sonic 3.6](https://docs.cartesia.ai/build-with-cartesia/tts-models/latest);
[whisper.cpp CLI](https://github.com/ggml-org/whisper.cpp);
[dots.mocr](https://github.com/rednote-hilab/dots.mocr); [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching);
and 16 CFR § 312.6 (parental review/deletion rights).

## License

MIT for the code and original lesson summaries, © Pedro Monteiro, 2026. Linked OER content
remains under its own license.
