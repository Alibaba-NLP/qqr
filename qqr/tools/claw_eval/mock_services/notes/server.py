"""Mock Notes API service for agent evaluation (FastAPI on port 9105)."""

from __future__ import annotations

import json
import copy
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Mock Notes API")

from mock_services._base import add_error_injection, resolve_task_fixture
add_error_injection(app)

FIXTURES_PATH = (
    Path(os.environ["NOTES_FIXTURES"])
    if os.environ.get("NOTES_FIXTURES")
    else resolve_task_fixture("T013zh_meeting_notes", "fixtures", "notes", "meetings.json")
)

_notes: list[dict[str, Any]] = []
_audit_log: list[dict[str, Any]] = []
_shared: list[dict[str, Any]] = []


def _load_fixtures() -> None:
    global _notes
    with open(FIXTURES_PATH) as f:
        _notes = json.load(f)


_load_fixtures()


def _log_call(endpoint: str, request_body: dict[str, Any], response_body: Any) -> None:
    _audit_log.append({
        "endpoint": endpoint,
        "request_body": request_body,
        "response_body": response_body,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


class ListRequest(BaseModel):
    max_results: int = 10
    tag: str | None = None
    days_back: int | None = None
    date_from: str | None = None
    date_to: str | None = None


class GetRequest(BaseModel):
    note_id: str


class ShareRequest(BaseModel):
    note_id: str
    recipients: list[str]


@app.post("/notes/list")
def list_notes(req: ListRequest | None = None) -> dict[str, Any]:
    if req is None:
        req = ListRequest()

    # Compute days_back cutoff if needed
    cutoff_date: str | None = None
    if req.days_back is not None:
        mock_today = os.environ.get("MOCK_TODAY")
        if mock_today:
            today = datetime.strptime(mock_today, "%Y-%m-%d")
        else:
            today = datetime.now(timezone.utc)
        cutoff = today - timedelta(days=req.days_back)
        cutoff_date = cutoff.strftime("%Y-%m-%d")

    filtered = []
    for note in _notes:
        if req.tag and req.tag not in note.get("tags", []):
            continue
        created = note.get("created_at", "")
        if req.date_from and created < req.date_from:
            continue
        if req.date_to and created > req.date_to:
            continue
        if cutoff_date and created < cutoff_date:
            continue
        filtered.append(note)

    results = []
    for note in filtered[:req.max_results]:
        results.append({
            "note_id": note["note_id"],
            "title": note["title"],
            "created_at": note["created_at"],
            "participants": note.get("participants", []),
            "duration_minutes": note.get("duration_minutes", 0),
        })
    resp = {"notes": results, "total": len(results)}
    _log_call("/notes/list", req.model_dump(), resp)
    return resp


@app.post("/notes/get")
def get_note(req: GetRequest) -> dict[str, Any]:
    for note in _notes:
        if note["note_id"] == req.note_id:
            resp = copy.deepcopy(note)
            _log_call("/notes/get", req.model_dump(), resp)
            return resp
    resp = {"error": f"Note {req.note_id} not found"}
    _log_call("/notes/get", req.model_dump(), resp)
    return resp


@app.post("/notes/share")
def share_note(req: ShareRequest) -> dict[str, Any]:
    record = {
        "note_id": req.note_id,
        "recipients": req.recipients,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _shared.append(record)
    resp = {"status": "shared", "record": record}
    _log_call("/notes/share", req.model_dump(), resp)
    return resp


@app.get("/notes/audit")
def get_audit() -> dict[str, Any]:
    return {"calls": _audit_log, "shared": _shared}


@app.post("/notes/reset")
def reset_state() -> dict[str, str]:
    global _audit_log, _shared
    _audit_log = []
    _shared = []
    _load_fixtures()
    return {"status": "reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "9105")))
