from __future__ import annotations

import base64
import json

import pytest

from personal_ai_tutor.api import TutorAPI
from personal_ai_tutor.photo import render_png
from personal_ai_tutor.study import compare_gains


@pytest.fixture
def api(engine) -> TutorAPI:
    result = compare_gains([5, 6, 7], [2, 3, 4])
    return TutorAPI(engine, study_runner=lambda: result)


def _json_body(payload: dict[str, object]) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _create(api: TutorAPI) -> dict[str, object]:
    response = api.handle(
        "POST",
        "/api/learners",
        {},
        _json_body(
            {"display_name": "API Learner", "age_band": "13_17", "consent": True}
        ),
    )
    assert response.status == 201
    return json.loads(response.body)


def test_static_and_health(api: TutorAPI) -> None:
    index = api.handle("GET", "/", {}, b"")
    assert index.status == 200
    assert b"Algebra Tutor" in index.body
    assert "Content-Security-Policy" in index.headers
    health = api.handle("GET", "/api/health", {}, b"")
    assert json.loads(health.body)["status"] == "ok"


def test_create_session_chat_and_auth(api: TutorAPI) -> None:
    credentials = _create(api)
    token = str(credentials["learner_token"])
    assert api.handle("POST", "/api/session", {}, b"{}").status == 401
    session = api.handle(
        "POST", "/api/session", {"authorization": f"Bearer {token}"}, b"{}"
    )
    assert session.status == 200
    turn = json.loads(session.body)
    session_id = turn["session_id"]
    response = api.handle(
        "POST",
        "/api/chat",
        {"authorization": f"Bearer {token}"},
        _json_body(
            {
                "session_id": session_id,
                "text": "I don't understand why 3x + 6 = 12 means x = 2",
            }
        ),
    )
    assert response.status == 200
    payload = json.loads(response.body)
    assert payload["concept_id"] == "two_step_equations"
    assert payload["message"].endswith("?")
    assert "x = 2" not in payload["message"]
    for answer in ("6", "6", "2"):
        response = api.handle(
            "POST",
            "/api/chat",
            {"authorization": f"Bearer {token}"},
            _json_body({"session_id": session_id, "text": answer}),
        )
        assert response.status == 200
    payload = json.loads(response.body)
    assert payload["solved"]
    assert payload["mastery_after"] > 0.5


def test_cross_learner_access_is_denied(api: TutorAPI) -> None:
    first = _create(api)
    second = _create(api)
    token_a = str(first["learner_token"])
    token_b = str(second["learner_token"])
    session = json.loads(
        api.handle(
            "POST", "/api/session", {"authorization": f"Bearer {token_a}"}, b"{}"
        ).body
    )
    response = api.handle(
        "POST",
        "/api/chat",
        {"authorization": f"Bearer {token_b}"},
        _json_body({"session_id": session["session_id"], "text": "6"}),
    )
    assert response.status == 403


def test_parent_routes(api: TutorAPI) -> None:
    credentials = _create(api)
    parent_key = str(credentials["parent_key"])
    assert api.handle("GET", "/api/progress", {}, b"").status == 401
    progress = api.handle(
        "GET", "/api/progress", {"x-parent-key": parent_key}, b""
    )
    assert progress.status == 200
    data = json.loads(progress.body)
    assert data["learner"]["id"] == credentials["learner_id"]
    pdf = api.handle(
        "GET", "/api/report.pdf?week=0", {"x-parent-key": parent_key}, b""
    )
    assert pdf.status == 200
    assert pdf.body.startswith(b"%PDF")
    exported = api.handle(
        "GET", "/api/export", {"x-parent-key": parent_key}, b""
    )
    assert exported.status == 200
    deleted = api.handle(
        "DELETE", "/api/memory", {"x-parent-key": parent_key}, b""
    )
    assert deleted.status == 204
    assert (
        api.handle("GET", "/api/export", {"x-parent-key": parent_key}, b"").status
        == 401
    )


def test_photo_voice_and_study(api: TutorAPI) -> None:
    credentials = _create(api)
    token = str(credentials["learner_token"])
    voice_ref = str(credentials["voice_ref"])
    turn = json.loads(
        api.handle(
            "POST", "/api/session", {"authorization": f"Bearer {token}"}, b"{}"
        ).body
    )
    photo = api.handle(
        "POST",
        "/api/photo",
        {"authorization": f"Bearer {token}"},
        _json_body(
            {
                "session_id": turn["session_id"],
                "image_b64": base64.b64encode(render_png("3x+6=12")).decode(),
                "media_type": "image/png",
            }
        ),
    )
    assert photo.status == 200
    assert "3x+6=12" in json.loads(photo.body)["item_prompt"]
    voice = api.handle(
        "POST",
        "/api/voice",
        {"authorization": f"Bearer {token}", "x-speaker-ref": voice_ref},
        _json_body({"session_id": turn["session_id"], "transcript": "6"}),
    )
    assert voice.status == 200
    assert json.loads(voice.body)["tts_b64"]
    study = api.handle("GET", "/api/study", {}, b"")
    assert study.status == 200
    assert "adaptive" in json.loads(study.body)


def test_bad_requests_and_consent(api: TutorAPI) -> None:
    malformed = api.handle("POST", "/api/learners", {}, b"{not json")
    assert malformed.status == 400
    consent = api.handle(
        "POST",
        "/api/learners",
        {},
        _json_body({"display_name": "Kid", "age_band": "under_13", "consent": False}),
    )
    assert consent.status == 403
    string_consent = api.handle(
        "POST",
        "/api/learners",
        {},
        _json_body({"display_name": "Kid", "age_band": "under_13", "consent": "false"}),
    )
    assert string_consent.status == 400
    missing = api.handle("POST", "/api/chat", {}, b"{}")
    assert missing.status == 401
    invalid_photo = api.handle(
        "POST",
        "/api/photo",
        {},
        b"{}",
    )
    assert invalid_photo.status == 401
