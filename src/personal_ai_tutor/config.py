"""Runtime configuration.

All secrets are read from the environment and excluded from repr/log output. The
defaults keep the entire tutor offline and deterministic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

LlmMode = Literal["off", "claude"]
AsrMode = Literal["scripted", "whisper"]
TtsMode = Literal["tone", "cartesia"]
OcrMode = Literal["synthetic", "http"]
GuardMode = Literal["rules", "llama_guard", "both"]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    db_path: Path = Path("var/tutor.db")
    retention_days: int = 365
    audit_retention_days: int = 730
    session_idle_days: int = 30
    mastery_target: float = 0.80
    prereq_threshold: float = 0.70
    llm: LlmMode = "off"
    asr: AsrMode = "scripted"
    tts: TtsMode = "tone"
    ocr: OcrMode = "synthetic"
    guard: GuardMode = "rules"
    guard_fail_closed: bool = True
    anthropic_api_key: str | None = field(default=None, repr=False)
    claude_model: str | None = None
    whisper_bin: str | None = None
    whisper_model: str | None = None
    cartesia_api_key: str | None = field(default=None, repr=False)
    cartesia_voice_id: str | None = None
    cartesia_model: str = "sonic-3.6"
    ocr_url: str | None = None
    ocr_api_key: str | None = field(default=None, repr=False)
    ocr_model: str = "dots.mocr"
    guard_url: str | None = None
    guard_model: str = "llama-guard-4"
    guard_api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> Settings:
        llm = os.environ.get("TUTOR_LLM", "off").strip().lower()
        asr = os.environ.get("TUTOR_ASR", "scripted").strip().lower()
        tts = os.environ.get("TUTOR_TTS", "tone").strip().lower()
        ocr = os.environ.get("TUTOR_OCR", "synthetic").strip().lower()
        guard = os.environ.get("TUTOR_GUARD", "rules").strip().lower()
        if llm not in {"off", "claude"}:
            raise ValueError("TUTOR_LLM must be off or claude")
        if asr not in {"scripted", "whisper"}:
            raise ValueError("TUTOR_ASR must be scripted or whisper")
        if tts not in {"tone", "cartesia"}:
            raise ValueError("TUTOR_TTS must be tone or cartesia")
        if ocr not in {"synthetic", "http"}:
            raise ValueError("TUTOR_OCR must be synthetic or http")
        if guard not in {"rules", "llama_guard", "both"}:
            raise ValueError("TUTOR_GUARD must be rules, llama_guard, or both")
        return cls(
            db_path=Path(os.environ.get("TUTOR_DB", "var/tutor.db")),
            retention_days=_env_int("TUTOR_RETENTION_DAYS", 365),
            audit_retention_days=_env_int("TUTOR_AUDIT_RETENTION_DAYS", 730),
            session_idle_days=_env_int("TUTOR_SESSION_IDLE_DAYS", 30),
            mastery_target=float(os.environ.get("TUTOR_MASTERY_TARGET", "0.80")),
            prereq_threshold=float(os.environ.get("TUTOR_PREREQ_THRESHOLD", "0.70")),
            llm=llm,  # type: ignore[arg-type]
            asr=asr,  # type: ignore[arg-type]
            tts=tts,  # type: ignore[arg-type]
            ocr=ocr,  # type: ignore[arg-type]
            guard=guard,  # type: ignore[arg-type]
            guard_fail_closed=_env_bool("TUTOR_GUARD_FAIL_CLOSED", True),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            claude_model=os.environ.get("TUTOR_CLAUDE_MODEL"),
            whisper_bin=os.environ.get("WHISPER_BIN"),
            whisper_model=os.environ.get("WHISPER_MODEL"),
            cartesia_api_key=os.environ.get("CARTESIA_API_KEY"),
            cartesia_voice_id=os.environ.get("CARTESIA_VOICE_ID"),
            cartesia_model=os.environ.get("CARTESIA_MODEL", "sonic-3.6"),
            ocr_url=os.environ.get("TUTOR_OCR_URL"),
            ocr_api_key=os.environ.get("TUTOR_OCR_API_KEY"),
            ocr_model=os.environ.get("TUTOR_OCR_MODEL", "dots.mocr"),
            guard_url=os.environ.get("TUTOR_GUARD_URL"),
            guard_model=os.environ.get("TUTOR_GUARD_MODEL", "llama-guard-4"),
            guard_api_key=os.environ.get("TUTOR_GUARD_API_KEY"),
        )
