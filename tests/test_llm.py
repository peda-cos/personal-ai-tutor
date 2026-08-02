from __future__ import annotations

import json

import pytest

from personal_ai_tutor.llm import ClaudeScaffolder
from personal_ai_tutor.safety import ProviderError


def test_claude_scaffolder_payload_and_validation() -> None:
    captured = {}

    def transport(url, headers, body, timeout):
        captured.update(json.loads(body))
        return json.dumps(
            {"content": [{"type": "text", "text": "What would you subtract first?"}]}
        ).encode()

    scaffolder = ClaudeScaffolder(
        "test_key_not_real", "claude-test-model", transport=transport
    )
    text = scaffolder.scaffold(
        "Two-Step Equations",
        "What is x?",
        "What operation undoes the constant?",
        "2",
    )
    assert text.endswith("?")
    assert captured["cache_control"] == {"type": "ephemeral"}
    assert captured["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_claude_rejects_answer_leak_and_bad_transport() -> None:
    def leak(url, headers, body, timeout):
        return json.dumps(
            {"content": [{"type": "text", "text": "So x = 2?"}]}
        ).encode()

    scaffolder = ClaudeScaffolder(
        "test_key_not_real", "claude-test-model", transport=leak
    )
    with pytest.raises(ProviderError):
        scaffolder.scaffold("Two-Step", "What is x?", "hint", "2")

    def broken(url, headers, body, timeout):
        raise OSError("down")

    broken_scaffolder = ClaudeScaffolder(
        "test_key_not_real", "claude-test-model", transport=broken
    )
    with pytest.raises(ProviderError):
        broken_scaffolder.scaffold("Two-Step", "What is x?", "hint", "2")
