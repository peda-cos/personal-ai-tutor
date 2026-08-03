from __future__ import annotations

import io
import json
import stat
import wave
from pathlib import Path

import pytest

from personal_ai_tutor.photo import (
    HttpVlmOcr,
    ImageError,
    SyntheticOcr,
    decode_image,
    render_image,
    render_pgm,
    render_png,
)
from personal_ai_tutor.safety import ProviderError
from personal_ai_tutor.voice import (
    CartesiaTts,
    ScriptedAsr,
    ToneTts,
    VoiceEvent,
    VoiceSession,
    WhisperCppAsr,
)


def test_tone_tts_is_valid_wav() -> None:
    chunks = list(ToneTts().synthesize("First sentence. Second sentence."))
    assert len(chunks) > 1
    with wave.open(io.BytesIO(b"".join(chunks)), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getnframes() > 0


def test_barge_in_stops_stream(engine, learner) -> None:
    turn = engine.start_session(learner.scope)
    session = VoiceSession(engine, learner.scope, turn.session_id, ToneTts())
    iterator = session.stream_tts("First sentence. Second sentence. Third sentence.")
    assert next(iterator)
    session.interrupt()
    assert list(iterator) == []


def test_scripted_asr_events() -> None:
    asr = ScriptedAsr(
        [VoiceEvent("final", "six", "vr_test"), VoiceEvent("barge_in")]
    )
    first = asr.next_event()
    assert first is not None and first.kind == "final"
    second = asr.next_event()
    assert second is not None and second.kind == "barge_in"
    assert asr.next_event() is None


def test_whisper_adapter_parses_json(tmp_path: Path) -> None:
    binary = tmp_path / "fake-whisper"
    model = tmp_path / "model.bin"
    model.write_bytes(b"fake")
    binary.write_text(
        """#!/usr/bin/env python3
import json, sys
args = sys.argv
base = args[args.index('-of') + 1]
with open(base + '.json', 'w', encoding='utf-8') as handle:
    json.dump({'transcription': [{'text': ' six '}]}, handle)
""",
        encoding="utf-8",
    )
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF")
    asr = WhisperCppAsr(str(binary), str(model))
    assert asr.transcribe(wav) == "six"
    with pytest.raises(ProviderError):
        WhisperCppAsr(str(tmp_path / "missing"), str(model)).transcribe(wav)


def test_cartesia_adapter_request_and_response() -> None:
    captured = {}

    def transport(url, headers, body, timeout):
        captured.update({"url": url, "headers": headers, "body": json.loads(body)})
        return b"RIFF....WAVE"

    tts = CartesiaTts(
        "test_key_not_real",
        "test_voice_not_real",
        transport=transport,
    )
    audio = b"".join(tts.synthesize("hello"))
    assert audio.startswith(b"RIFF")
    assert captured["headers"]["Cartesia-Version"]
    assert captured["body"]["output_format"]["container"] == "wav"
    with pytest.raises(ProviderError):
        CartesiaTts("", "")


def test_synthetic_ocr_recognizes_clean_equations() -> None:
    for text in ("3x+6=12", "x^2=16", "2(x+1)=8", "5x-3=2x+9"):
        result = SyntheticOcr().recognize(decode_image(render_png(text)))
        assert result.text == text
        assert result.confidence == 1.0
        from_pgm = SyntheticOcr().recognize(decode_image(render_pgm(render_image(text))))
        assert from_pgm.text == text


def test_ocr_rejects_garbage() -> None:
    with pytest.raises(ImageError):
        SyntheticOcr().recognize(render_image(" "))
    with pytest.raises(ImageError):
        decode_image(b"not an image")
    with pytest.raises(ImageError):
        decode_image(b"P5\n1 1\n255\n")


def test_http_ocr_adapter() -> None:
    def transport(url, headers, body, timeout):
        payload = json.loads(body)
        assert payload["messages"][1]["content"][1]["type"] == "image_url"
        return json.dumps(
            {"choices": [{"message": {"content": "```text\n3x+6=12\n```"}}]}
        ).encode()

    ocr = HttpVlmOcr("https://ocr.invalid/v1/chat/completions", transport=transport)
    result = ocr.recognize(render_image("3x+6=12"))
    assert result.text == "3x+6=12"
