# Personal AI Tutor — adaptive algebra with memory and safety

A K-12 algebra tutor that asks leading questions instead of dumping answers. It combines a
Socratic step policy, Bayesian knowledge tracing (BKT), FSRS-6 spaced repetition, a
prerequisite curriculum graph, learner-scoped episodic/semantic memory, age-appropriate
safety filters, voice and photo-math input, and a reproducible simulated efficacy study.

The project runs offline by default. Paid or GPU-backed services (Claude, Neo4j, LiveKit,
Whisper, Cartesia, dots.mocr/PaliGemma, Llama Guard 4) are optional adapters with documented
boundaries; the learner model, graph engine, grading, memory, safety baseline, study, and PDF
reports are implemented locally.

## Quickstart

```bash
./verify.sh
uv run python -m personal_ai_tutor serve --port 8000
# open http://127.0.0.1:8000
```

## Status

Implementation in progress. Architecture, measured results, limitations, and references are
added before the final commit.
