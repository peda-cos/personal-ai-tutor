"""In-process API handler and a thin stdlib HTTP server."""

from __future__ import annotations

import base64
import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .engine import (
    PhotoRecognitionError,
    SessionNotFound,
    Turn,
    TutorEngine,
    VoiceTurn,
)
from .reports import build_weekly_report, render_pdf
from .safety import ProviderError
from .store import AuthError, ConsentRequired, LearnerScope, ParentScope, ScopeError
from .study import StudyResult, run_study


@dataclass(frozen=True)
class Response:
    status: int
    headers: dict[str, str]
    body: bytes


class TutorAPI:
    def __init__(
        self,
        engine: TutorEngine,
        *,
        study_runner: Callable[[], StudyResult] = run_study,
        static_dir: Path | None = None,
    ) -> None:
        self.engine = engine
        self.study_runner = study_runner
        self.static_dir = static_dir or Path(__file__).parent / "static"
        self._study_lock = threading.Lock()
        self._study_cache: StudyResult | None = None

    def handle(
        self,
        method: str,
        path: str,
        headers: Mapping[str, str],
        body: bytes,
    ) -> Response:
        normalized_headers = {key.lower(): value for key, value in headers.items()}
        parsed = urlparse(path)
        route = parsed.path
        try:
            if method == "GET" and route in {"/", "/app.js", "/styles.css"}:
                return self._static(route)
            if method == "GET" and route == "/api/health":
                return self._json(200, {"status": "ok"})
            if method == "POST" and route == "/api/learners":
                return self._create_learner(body)
            if method == "POST" and route == "/api/session":
                scope = self._learner_scope(normalized_headers)
                turn = self.engine.start_session(scope)
                return self._json(200, self._turn_payload(turn))
            if method == "POST" and route == "/api/chat":
                scope = self._learner_scope(normalized_headers)
                payload = self._json_body(body)
                text = str(payload.get("text", ""))
                if len(text) > 2000:
                    return self._error(413, "text_too_long", "text is limited to 2000 characters")
                turn = self.engine.reply(
                    scope,
                    str(payload.get("session_id", "")),
                    text,
                    modality="text",
                )
                return self._json(200, self._turn_payload(turn))
            if method == "POST" and route == "/api/photo":
                scope = self._learner_scope(normalized_headers)
                payload = self._json_body(body)
                encoded = str(payload.get("image_b64", ""))
                try:
                    image_bytes = base64.b64decode(encoded, validate=True)
                except (ValueError, TypeError) as exc:
                    raise PhotoRecognitionError("image_b64 is not valid base64") from exc
                turn = self.engine.photo_turn(
                    scope,
                    str(payload.get("session_id", "")),
                    image_bytes,
                    str(payload.get("media_type", "image/png")),
                )
                return self._json(200, self._turn_payload(turn))
            if method == "POST" and route == "/api/voice":
                scope = self._learner_scope(normalized_headers)
                payload = self._json_body(body)
                speaker = normalized_headers.get("x-speaker-ref", "")
                result = self.engine.voice_turn(
                    scope,
                    str(payload.get("session_id", "")),
                    str(payload.get("transcript", "")),
                    speaker,
                    bool(payload.get("barge_in", False)),
                )
                return self._json(200, self._voice_payload(result))
            if method == "POST" and route == "/api/tts":
                scope = self._learner_scope(normalized_headers)
                payload = self._json_body(body)
                text = str(payload.get("text", ""))[:2000]
                audio = b"".join(self.engine.tts.synthesize(text))
                return Response(200, {"Content-Type": "audio/wav"}, audio)
            if method == "GET" and route == "/api/progress":
                parent = self._parent_scope(normalized_headers)
                return self._json(200, self.engine.progress(parent))
            if method == "GET" and route == "/api/report.pdf":
                parent = self._parent_scope(normalized_headers)
                week = int(parse_qs(parsed.query).get("week", ["0"])[0])
                if week not in {0, 1}:
                    return self._error(400, "bad_week", "week must be 0 or 1")
                report = build_weekly_report(
                    self.engine.store,
                    self.engine.curriculum,
                    parent,
                    week,
                    self.engine.memory,
                )
                return Response(
                    200,
                    {
                        "Content-Type": "application/pdf",
                        "Content-Disposition": "attachment; filename=weekly-report.pdf",
                    },
                    render_pdf(report),
                )
            if method == "GET" and route == "/api/export":
                parent = self._parent_scope(normalized_headers)
                return self._json(200, self.engine.export(parent))
            if method == "DELETE" and route == "/api/memory":
                parent = self._parent_scope(normalized_headers)
                self.engine.delete(parent)
                return Response(204, {}, b"")
            if method == "GET" and route == "/api/study":
                return self._json(200, self._study().to_dict())
            return self._error(404, "not_found", "route not found")
        except AuthError as exc:
            return self._error(401, "unauthorized", str(exc))
        except ScopeError as exc:
            return self._error(403, "forbidden", str(exc))
        except SessionNotFound as exc:
            return self._error(404, "session_not_found", str(exc))
        except ConsentRequired as exc:
            return self._error(403, "consent_required", str(exc))
        except PhotoRecognitionError as exc:
            return self._error(422, "photo_recognition_failed", str(exc))
        except ProviderError as exc:
            return self._error(503, "provider_unavailable", str(exc))
        except (ValueError, KeyError, TypeError) as exc:
            return self._error(400, "bad_request", str(exc))
        except Exception:
            return self._error(500, "internal_error", "unexpected server error")

    def _learner_scope(self, headers: Mapping[str, str]) -> LearnerScope:
        authorization = headers.get("authorization", "")
        if not authorization.lower().startswith("bearer "):
            raise AuthError("missing learner bearer token")
        return self.engine.learner_scope(authorization.split(" ", 1)[1].strip())

    def _parent_scope(self, headers: Mapping[str, str]) -> ParentScope:
        key = headers.get("x-parent-key", "")
        if not key:
            raise AuthError("missing parent key")
        return self.engine.parent_scope(key)

    def _create_learner(self, body: bytes) -> Response:
        payload = self._json_body(body)
        display_name = payload.get("display_name", "")
        age_band = payload.get("age_band", "13_17")
        consent = payload.get("consent", False)
        if not isinstance(display_name, str):
            raise ValueError("display_name must be a string")
        if not isinstance(age_band, str):
            raise ValueError("age_band must be a string")
        if not isinstance(consent, bool):
            raise ValueError("consent must be true or false")
        credentials = self.engine.create_learner(
            display_name,
            age_band,
            consent,
        )
        return self._json(
            201,
            {
                "learner_id": credentials.learner_id,
                "learner_token": credentials.learner_token,
                "parent_key": credentials.parent_key,
                "voice_ref": credentials.voice_ref,
            },
        )

    def _json_body(self, body: bytes) -> dict[str, object]:
        if len(body) > 6 * 1024 * 1024:
            raise ValueError("request body is too large")
        if not body:
            return {}
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def _study(self) -> StudyResult:
        with self._study_lock:
            if self._study_cache is None:
                self._study_cache = self.study_runner()
            return self._study_cache

    def _static(self, route: str) -> Response:
        name = {"/": "index.html", "/app.js": "app.js", "/styles.css": "styles.css"}[route]
        path = self.static_dir / name
        if not path.exists():
            return self._error(404, "not_found", "static asset missing")
        content_type = {
            "index.html": "text/html; charset=utf-8",
            "app.js": "text/javascript; charset=utf-8",
            "styles.css": "text/css; charset=utf-8",
        }[name]
        headers = {"Content-Type": content_type}
        if name == "index.html":
            headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' data:; connect-src 'self'"
            )
        return Response(200, headers, path.read_bytes())

    def _turn_payload(self, turn: Turn) -> dict[str, object]:
        return {
            "session_id": turn.session_id,
            "message": turn.message,
            "concept_id": turn.concept_id,
            "concept_title": turn.concept_title,
            "item_prompt": turn.item_prompt,
            "visual": turn.visual,
            "step_question": turn.step_question,
            "mastery_before": turn.mastery_before,
            "mastery_after": turn.mastery_after,
            "signal": turn.signal,
            "solved": turn.solved,
            "safety_blocked": turn.safety_blocked,
            "hint_level": turn.hint_level,
        }

    def _voice_payload(self, result: VoiceTurn) -> dict[str, object]:
        payload = self._turn_payload(result.turn)
        payload["tts_b64"] = base64.b64encode(result.audio).decode("ascii")
        return payload

    def _json(self, status: int, payload: object) -> Response:
        return Response(
            status,
            {"Content-Type": "application/json; charset=utf-8"},
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        )

    def _error(self, status: int, code: str, message: str) -> Response:
        return self._json(
            status,
            {"error": {"code": code, "message": message}},
        )


class _Handler(BaseHTTPRequestHandler):
    server: TutorHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch("DELETE")

    def _dispatch(self, method: str) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            length = 0
        if length > 6 * 1024 * 1024:
            payload = b'{"error":{"code":"too_large","message":"request body is too large"}}'
            self._send(Response(413, {"Content-Type": "application/json"}, payload))
            return
        body = self.rfile.read(length) if length > 0 else b""
        headers = {key.lower(): value for key, value in self.headers.items()}
        response = self.server.api.handle(method, self.path, headers, body)
        self._send(response)

    def _send(self, response: Response) -> None:
        self.send_response(response.status)
        for key, value in response.headers.items():
            self.send_header(key, value)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Length", str(len(response.body)))
        self.end_headers()
        if response.body:
            self.wfile.write(response.body)

    def log_message(self, format: str, *args: object) -> None:
        return


class TutorHTTPServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], api: TutorAPI) -> None:
        self.api = api
        super().__init__(address, _Handler)


def serve(api: TutorAPI, host: str = "127.0.0.1", port: int = 8000) -> TutorHTTPServer:
    server = TutorHTTPServer((host, port), api)
    return server
