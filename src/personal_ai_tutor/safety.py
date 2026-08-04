"""Age-appropriate safety rules, PII redaction, and an optional Llama Guard 4 adapter."""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .store import LearnerScope, Store


class ProviderError(RuntimeError):
    """Raised when a configured external provider cannot be reached."""


Transport = Callable[[str, dict[str, str], bytes, float], bytes]


@dataclass(frozen=True)
class SafetyVerdict:
    allowed: bool
    category: str | None
    response: str | None
    redacted_text: str


@dataclass(frozen=True)
class AuditCaseResult:
    text: str
    expected: bool
    actual: bool
    category: str | None
    direction: str


@dataclass(frozen=True)
class AuditResult:
    total: int
    passed: int
    false_positives: int
    false_negatives: int
    results: tuple[AuditCaseResult, ...]

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    categories: tuple[str, ...]
    raw: str


SELF_HARM_RESPONSE = (
    "You're not alone in this. Please tell a trusted adult right now, or contact local "
    "emergency services if you might be in danger. We can return to algebra when you're safe."
)
REFUSAL = (
    "I can't help with that. Let's return to the algebra problem: what would you try first?"
)
PRIVACY_NOTE = (
    "Please don't share personal details here. Let's keep working on the math: "
    "what is the next step?"
)
GUARD_UNAVAILABLE = (
    "I can't run the safety check right now, so I can't continue this turn. Please try again later."
)


_UNSAFE_RULES: dict[str, tuple[re.Pattern[str], ...]] = {
    "self_harm": tuple(
        re.compile(pattern, re.I)
        for pattern in (
            r"\b(?:kill|hurt|harm)\s+myself\b",
            r"\bcut(?:ting)?\s+myself\b",
            r"\bsuicid(?:e|al)\b",
            r"\bself[-\s]?harm\b",
            r"\bwant(?:ed)? to die\b",
            r"\bend my life\b",
            r"\bdon'?t want to live\b",
            r"\bno reason to live\b",
        )
    ),
    "sexual": tuple(
        re.compile(pattern, re.I)
        for pattern in (
            r"\bsex\b",
            r"\bporn(?:ography)?\b",
            r"\bnudes?\b",
            r"\berotic\b",
            r"\bsexual\b",
            r"\bgenital(?:s|ia)?\b",
        )
    ),
    "violence": tuple(
        re.compile(pattern, re.I)
        for pattern in (
            r"\bkill\s+(?:him|her|them|someone|people|everyone)\b",
            r"\bbeat\s+(?:him|her|them|someone|people)\s+up\b",
            r"\bhurt\s+(?:him|her|them|someone|people)\b",
            r"\bshoot\s+(?:him|her|them|someone|people|up)\b",
            r"\bstab\s+(?:him|her|them|someone|people)\b",
            r"\battack\s+(?:him|her|them|someone|people)\b",
        )
    ),
    "weapons": tuple(
        re.compile(pattern, re.I)
        for pattern in (
            r"\b(?:make|build|assemble)\s+(?:a\s+)?(?:bomb|explosive|gun|weapon)\b",
            r"\bbuy\s+(?:a\s+)?(?:gun|rifle|explosive)\b",
            r"\bpipe bomb\b",
            r"\bnerve gas\b",
        )
    ),
    "drugs": tuple(
        re.compile(pattern, re.I)
        for pattern in (
            r"\bhow\s+to\s+get\s+(?:drunk|high)\b",
            r"\bbuy\s+(?:weed|cocaine|heroin|drugs)\b",
            r"\bmake\s+(?:meth|drugs)\b",
            r"\bvape\s+at\s+school\b",
        )
    ),
    "hate": tuple(
        re.compile(pattern, re.I)
        for pattern in (
            r"\bhate\s+all\s+(?:jews|muslims|blacks|asians|gays|trans people|immigrants)\b",
            r"\b(?:jews|muslims|blacks|asians|gays|trans people|immigrants)\s+"
            r"are\s+(?:stupid|inferior)\b",
        )
    ),
    "jailbreak": tuple(
        re.compile(pattern, re.I)
        for pattern in (
            r"\bignore\s+(?:all\s+)?(?:previous|prior)\s+instructions\b",
            r"\byou are now\b",
            r"\bsystem prompt\b",
            r"\bdeveloper mode\b",
            r"\bpretend\s+(?:you are|to be)\b",
        )
    ),
    "specialized_advice": tuple(
        re.compile(pattern, re.I)
        for pattern in (
            r"\bwhat\s+(?:medicine|drug)\s+should\s+i\s+take\b",
            r"\bdiagnose\s+me\b",
            r"\blegal\s+advice\b",
            r"\bmedical\s+advice\b",
        )
    ),
}

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.I)
_PHONE = re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)")
_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_ADDRESS = re.compile(
    r"\b\d{1,5}\s+[A-Za-z][A-Za-z0-9.\s]{0,30}\s+(?:street|st|avenue|ave|road|rd|lane|ln|drive|dr)\b",
    re.I,
)
_NAME = re.compile(r"\bmy name is\s+[A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+)?", re.I)


def redact_pii(text: str) -> str:
    result = _EMAIL.sub("[redacted email]", text)
    result = _PHONE.sub("[redacted phone]", result)
    result = _SSN.sub("[redacted id]", result)
    result = _ADDRESS.sub("[redacted address]", result)
    result = _NAME.sub("[redacted name]", result)
    return result


def _rule_match(text: str) -> tuple[str, re.Pattern[str]] | None:
    for category, patterns in _UNSAFE_RULES.items():
        for pattern in patterns:
            if pattern.search(text):
                return category, pattern
    return None


def _response_for(category: str) -> str:
    if category == "self_harm":
        return SELF_HARM_RESPONSE
    if category == "specialized_advice":
        return (
            "I can't give medical or legal advice. Please ask a trusted adult. "
            "Let's return to the math: what is the first step?"
        )
    return REFUSAL


class LlamaGuard:
    """OpenAI-compatible Llama Guard 4 client.

    The model is an optional real-provider adapter. Tests inject a fake transport,
    so the suite never opens a socket.
    """

    def __init__(
        self,
        url: str,
        model: str = "llama-guard-4",
        api_key: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.url = url
        self.model = model
        self.api_key = api_key
        self.transport = transport or _urllib_transport

    def classify(self, text: str, direction: str) -> GuardResult:
        system = (
            "You are a safety classifier for a K-12 algebra tutor. Classify the "
            "conversation as safe or unsafe using the MLCommons categories S1-S14. "
            "Respond with 'safe' or 'unsafe' followed by comma-separated category "
            "codes. Categories: S1 violent crimes, S2 non-violent crimes, S3 sex "
            "crimes, S4 child sexual exploitation, S5 defamation, S6 specialized "
            "advice, S7 privacy, S8 intellectual property, S9 indiscriminate weapons, "
            "S10 hate, S11 suicide/self-harm, S12 sexual content, S13 elections, "
            "S14 code interpreter abuse."
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"{direction.upper()}:\n{text}"},
            ],
            "temperature": 0,
            "max_tokens": 40,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            raw = self.transport(
                self.url,
                headers,
                json.dumps(payload).encode("utf-8"),
                5.0,
            )
            parsed = json.loads(raw.decode("utf-8"))
            content = str(parsed["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError, ValueError, OSError) as exc:
            raise ProviderError("Llama Guard request failed") from exc
        lowered = content.lower()
        if lowered.startswith("safe"):
            return GuardResult(True, (), content)
        codes = tuple(re.findall(r"S\d+", content.upper()))
        categories = tuple(sorted({_GUARD_MAP.get(code, "guard_unsafe") for code in codes}))
        return GuardResult(False, categories or ("guard_unsafe",), content)


_GUARD_MAP = {
    "S1": "violence",
    "S2": "weapons",
    "S3": "sexual",
    "S4": "sexual",
    "S6": "specialized_advice",
    "S7": "pii",
    "S9": "weapons",
    "S10": "hate",
    "S11": "self_harm",
    "S12": "sexual",
    "S14": "jailbreak",
}


def _urllib_transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> bytes:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return bytes(response.read())


class SafetyService:
    def __init__(
        self,
        store: Store,
        guard: LlamaGuard | None = None,
        fail_closed: bool = True,
        fixtures_path: Path | None = None,
    ) -> None:
        self.store = store
        self.guard = guard
        self.fail_closed = fail_closed
        self.fixtures_path = fixtures_path or (
            Path(__file__).parent / "data" / "safety_cases.json"
        )

    def check_input(self, scope: LearnerScope, text: str) -> SafetyVerdict:
        redacted = redact_pii(text)
        pii = redacted != text
        match = _rule_match(redacted)
        if match is not None:
            category, _pattern = match
            self.store.add_safety_event(scope, "input", category, True, text)
            return SafetyVerdict(False, category, _response_for(category), redacted)
        if pii:
            self.store.add_safety_event(scope, "input", "pii", False, text)
        guarded = self._guard(scope, "input", redacted)
        if guarded is not None:
            return guarded
        return SafetyVerdict(True, "pii" if pii else None, PRIVACY_NOTE if pii else None, redacted)

    def check_output(self, scope: LearnerScope, text: str) -> SafetyVerdict:
        match = _rule_match(text)
        if match is not None:
            category, _pattern = match
            self.store.add_safety_event(scope, "output", category, True, text)
            return SafetyVerdict(False, category, _response_for(category), text)
        if redact_pii(text) != text:
            self.store.add_safety_event(scope, "output", "pii", True, text)
            return SafetyVerdict(False, "pii", PRIVACY_NOTE, text)
        guarded = self._guard(scope, "output", text)
        if guarded is not None:
            return guarded
        return SafetyVerdict(True, None, None, text)

    def _guard(self, scope: LearnerScope, direction: str, text: str) -> SafetyVerdict | None:
        if self.guard is None:
            return None
        try:
            result = self.guard.classify(text, direction)
        except ProviderError:
            if self.fail_closed:
                self.store.add_safety_event(scope, direction, "guard_unavailable", True, text)
                return SafetyVerdict(False, "guard_unavailable", GUARD_UNAVAILABLE, text)
            self.store.add_safety_event(scope, direction, "guard_unavailable", False, text)
            return None
        if result.allowed:
            return None
        category = result.categories[0] if result.categories else "guard_unsafe"
        self.store.add_safety_event(scope, direction, category, True, text)
        return SafetyVerdict(False, category, _response_for(category), text)

    def audit_cases(self) -> AuditResult:
        raw = json.loads(self.fixtures_path.read_text(encoding="utf-8"))
        cases = raw.get("cases", [])
        results: list[AuditCaseResult] = []
        for case in cases:
            text = str(case["text"])
            expected = str(case["expected"]) == "allow"
            match = _rule_match(redact_pii(text))
            actual = match is None
            category = match[0] if match else ("pii" if redact_pii(text) != text else None)
            results.append(
                AuditCaseResult(
                    text=text,
                    expected=expected,
                    actual=actual,
                    category=category,
                    direction=str(case.get("direction", "input")),
                )
            )
        passed = sum(1 for result in results if result.expected == result.actual)
        false_positives = sum(
            1 for result in results if result.expected and not result.actual
        )
        false_negatives = sum(
            1 for result in results if not result.expected and result.actual
        )
        return AuditResult(
            total=len(results),
            passed=passed,
            false_positives=false_positives,
            false_negatives=false_negatives,
            results=tuple(results),
        )


def load_audit_cases(path: Path | None = None) -> list[dict[str, str]]:
    source = path or Path(__file__).parent / "data" / "safety_cases.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    return list(raw.get("cases", []))
