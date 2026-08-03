"""Transport-agnostic voice loop with barge-in, plus real ASR/TTS adapters."""

from __future__ import annotations

import io
import json
import math
import struct
import subprocess
import tempfile
import wave
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .config import Settings
from .safety import ProviderError
from .store import LearnerScope, ScopeError

if TYPE_CHECKING:
    from .engine import TutorEngine


@dataclass(frozen=True)
class VoiceEvent:
    kind: str
    text: str = ""
    speaker_ref: str = ""


@dataclass(frozen=True)
class VoiceReply:
    text: str
    audio: bytes
    interrupted: bool


class TtsAdapter(Protocol):
    def synthesize(self, text: str) -> Iterator[bytes]: ...


class AsrAdapter(Protocol):
    def transcribe(self, wav_path: Path) -> str: ...


class VoiceSession:
    def __init__(
        self,
        engine: TutorEngine,
        scope: LearnerScope,
        session_id: str,
        tts: TtsAdapter,
    ) -> None:
        self.engine = engine
        self.scope = scope
        self.session_id = session_id
        self.tts = tts
        self._interrupted = False
        self.tts_chunks_emitted = 0

    def feed(self, event: VoiceEvent) -> VoiceReply | None:
        if event.kind == "barge_in":
            self.interrupt()
            return None
        if event.kind != "final":
            return None
        if event.speaker_ref != self.scope.voice_ref:
            self.engine.store.audit(
                self.scope.learner_id,
                self.scope.learner_id,
                "cross_learner_voice_ingest",
                False,
                "speaker label did not match the authenticated learner",
            )
            self.engine.store.add_safety_event(
                self.scope,
                "voice",
                "cross_learner_voice",
                True,
                event.text,
            )
            raise ScopeError("voice clip does not belong to this learner")
        turn = self.engine.reply(
            self.scope, self.session_id, event.text, modality="voice"
        )
        audio = b"".join(self.stream_tts(turn.message))
        return VoiceReply(text=turn.message, audio=audio, interrupted=self._interrupted)

    def stream_tts(self, text: str) -> Iterator[bytes]:
        self._interrupted = False
        for chunk in self.tts.synthesize(text):
            if self._interrupted:
                return
            if chunk:
                self.tts_chunks_emitted += 1
                yield chunk

    def interrupt(self) -> None:
        self._interrupted = True


class ScriptedAsr:
    """Deterministic event source for the local voice loop."""

    def __init__(self, events: list[VoiceEvent]) -> None:
        self._events = list(events)
        self._index = 0

    def next_event(self) -> VoiceEvent | None:
        if self._index >= len(self._events):
            return None
        event = self._events[self._index]
        self._index += 1
        return event


class WhisperCppAsr:
    def __init__(self, binary: str, model: str, timeout: float = 30.0) -> None:
        self.binary = binary
        self.model = model
        self.timeout = timeout

    def transcribe(self, wav_path: Path) -> str:
        if not Path(self.binary).exists():
            raise ProviderError("whisper.cpp binary was not found")
        if not Path(self.model).exists():
            raise ProviderError("whisper.cpp model was not found")
        with tempfile.TemporaryDirectory(prefix="tutor-whisper-") as tmp:
            output_base = Path(tmp) / "transcript"
            command = [
                self.binary,
                "-m",
                self.model,
                "-f",
                str(wav_path),
                "-l",
                "en",
                "-oj",
                "-of",
                str(output_base),
                "-np",
            ]
            try:
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    timeout=self.timeout,
                )
            except (subprocess.SubprocessError, OSError) as exc:
                raise ProviderError("whisper.cpp transcription failed") from exc
            json_path = output_base.with_suffix(".json")
            if not json_path.exists():
                raise ProviderError("whisper.cpp produced no transcript JSON")
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ProviderError("whisper.cpp transcript JSON is invalid") from exc
        pieces: list[str] = []
        for item in payload.get("transcription", []):
            if isinstance(item, dict) and item.get("text"):
                pieces.append(str(item["text"]).strip())
        text = " ".join(piece for piece in pieces if piece).strip()
        if not text:
            raise ProviderError("whisper.cpp returned an empty transcript")
        return text


class ToneTts:
    """Deterministic WAV output used for offline integration and tests.

    This is a tone generator, not speech. Cartesia is the real voice adapter.
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate

    def synthesize(self, text: str) -> Iterator[bytes]:
        payload = self._wav(text)
        chunk_size = max(1, len(payload) // 4)
        for offset in range(0, len(payload), chunk_size):
            yield payload[offset : offset + chunk_size]

    def _wav(self, text: str) -> bytes:
        duration = min(2.0, max(0.12, 0.06 * len(text)))
        frames = int(self.sample_rate * duration)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            payload = bytearray()
            for index in range(frames):
                envelope = 0.25 * (1.0 - index / max(1, frames))
                sample = int(
                    12000 * envelope * math.sin(2 * math.pi * 440 * index / self.sample_rate)
                )
                payload.extend(struct.pack("<h", sample))
            handle.writeframes(bytes(payload))
        return buffer.getvalue()


class CartesiaTts:
    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model: str = "sonic-3.6",
        url: str = "https://api.cartesia.ai/tts/bytes",
        version: str = "2026-08-14",
        transport: Callable[[str, dict[str, str], bytes, float], bytes] | None = None,
    ) -> None:
        if not api_key or not voice_id:
            raise ProviderError("CARTESIA_API_KEY and CARTESIA_VOICE_ID are required")
        self.api_key = api_key
        self.voice_id = voice_id
        self.model = model
        self.url = url
        self.version = version
        self.transport = transport or _urllib_post

    def synthesize(self, text: str) -> Iterator[bytes]:
        payload = {
            "model_id": self.model,
            "transcript": text,
            "voice": {"id": self.voice_id},
            "output_format": {
                "container": "wav",
                "encoding": "pcm_s16le",
                "sample_rate": 16000,
            },
            "language": "en",
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Cartesia-Version": self.version,
        }
        try:
            audio = self.transport(
                self.url,
                headers,
                json.dumps(payload).encode("utf-8"),
                10.0,
            )
        except OSError as exc:
            raise ProviderError("Cartesia request failed") from exc
        if not audio.startswith(b"RIFF"):
            raise ProviderError("Cartesia returned non-WAV audio")
        yield audio


def _urllib_post(url: str, headers: dict[str, str], body: bytes, timeout: float) -> bytes:
    import urllib.request

    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return bytes(response.read())


def build_asr(settings: Settings) -> AsrAdapter | None:
    if settings.asr == "whisper":
        if not settings.whisper_bin or not settings.whisper_model:
            raise ProviderError("WHISPER_BIN and WHISPER_MODEL are required")
        return WhisperCppAsr(settings.whisper_bin, settings.whisper_model)
    return None


def build_tts(settings: Settings) -> TtsAdapter:
    if settings.tts == "cartesia":
        if not settings.cartesia_api_key or not settings.cartesia_voice_id:
            raise ProviderError("CARTESIA_API_KEY and CARTESIA_VOICE_ID are required")
        return CartesiaTts(
            settings.cartesia_api_key,
            settings.cartesia_voice_id,
            settings.cartesia_model,
        )
    return ToneTts()
