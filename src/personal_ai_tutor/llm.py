"""Optional Claude scaffolding adapter with prompt caching and validation."""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Callable

from .safety import ProviderError

Transport = Callable[[str, dict[str, str], bytes, float], bytes]


class ClaudeScaffolder:
    def __init__(
        self,
        api_key: str,
        model: str,
        url: str = "https://api.anthropic.com/v1/messages",
        version: str = "2026-10-21",
        transport: Transport | None = None,
    ) -> None:
        if not api_key or not model:
            raise ProviderError("ANTHROPIC_API_KEY and TUTOR_CLAUDE_MODEL are required")
        self.api_key = api_key
        self.model = model
        self.url = url
        self.version = version
        self.transport = transport or _urllib_post

    def scaffold(
        self,
        concept_title: str,
        question: str,
        hint: str,
        final_answer: str,
    ) -> str:
        system = (
            "You are a Socratic K-12 algebra tutor. Ask one short leading question. "
            "Never state the final answer or a completed solution. Keep the reply under "
            "70 words and end with a question mark. Do not ask for personal information."
        )
        user = (
            f"Concept: {concept_title}\n"
            f"Current step question: {question}\n"
            f"Planned scaffold: {hint}\n"
            "Rewrite the scaffold as one warm, concrete leading question."
        )
        payload = {
            "model": self.model,
            "max_tokens": 180,
            "cache_control": {"type": "ephemeral"},
            "system": [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.version,
        }
        try:
            raw = self.transport(
                self.url, headers, json.dumps(payload).encode("utf-8"), 12.0
            )
            data = json.loads(raw.decode("utf-8"))
            blocks = data.get("content", [])
            text = " ".join(
                str(block.get("text", ""))
                for block in blocks
                if isinstance(block, dict) and block.get("type") == "text"
            ).strip()
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ProviderError("Claude scaffolding request failed") from exc
        if not _valid_scaffold(text, final_answer):
            raise ProviderError("Claude scaffolding response failed validation")
        return text


def _valid_scaffold(text: str, final_answer: str) -> bool:
    if not text or not text.endswith("?") or len(text) > 700 or len(text) < 8:
        return False
    lowered = text.lower()
    final = final_answer.strip().lower()
    return not (
        final
        and (
            f"answer is {final}" in lowered
            or f"x = {final}" in lowered
            or re.search(rf"\b{re.escape(final)}\b\s*$", lowered)
        )
    )


def _urllib_post(url: str, headers: dict[str, str], body: bytes, timeout: float) -> bytes:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return bytes(response.read())
