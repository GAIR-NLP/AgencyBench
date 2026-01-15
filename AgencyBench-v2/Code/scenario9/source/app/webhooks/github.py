import hashlib
import hmac
import json
import secrets
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request

from app.settings import require_env
from app.sessions.service import create_or_get_session, update_issue_info
from app.logs.service import log
from app.github.service import async_comment_ack


router = APIRouter(prefix="/webhooks", tags=["webhooks"])


def _verify_github_signature(secret: str, body: bytes, signature_header: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return secrets.compare_digest(expected, signature_header)


@router.post("/github")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    signature_header: Optional[str] = Header(None, alias="X-Hub-Signature-256"),
    event: Optional[str] = Header(None, alias="X-GitHub-Event"),
):
    if not signature_header:
        raise HTTPException(status_code=401, detail="Missing signature header")

    secret = require_env("WEBHOOK_SECRET")

    raw_body = await request.body()
    if not _verify_github_signature(secret, raw_body, signature_header):
        raise HTTPException(status_code=401, detail="Invalid signature")

    if not event:
        raise HTTPException(status_code=400, detail="Missing X-GitHub-Event header")

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # Minimal handling for supported events
    if event == "ping":
        return {"ok": True, "pong": True}

    if event == "issues":
        action = payload.get("action")
        if action == "assigned":
            issue = payload.get("issue") or {}
            repository = payload.get("repository", {})
            repo_full = repository.get("full_name")
            issue_number = issue.get("number")
            if not repo_full or issue_number is None:
                raise HTTPException(status_code=400, detail="Missing repository or issue_number")
            session_id = create_or_get_session(repo_full, int(issue_number))
            log(session_id=session_id, level="info", message="issues.assigned received", source="webhook")
            issue_title = str(issue.get("title") or "")
            issue_body = str(issue.get("body") or "")
            update_issue_info(session_id, issue_title, issue_body)
            async_comment_ack(background_tasks, session_id=session_id, repository=repo_full, issue_number=int(issue_number))
            # Agent execution removed - part of task 2
            log(session_id=session_id, level="info", message=f"Session created for {repo_full}#{issue_number} (agent execution not implemented yet)", source="webhook")
            return {"ok": True, "received": "issues.assigned", "session_id": session_id}
        return {"ok": True, "ignored_action": action}

    if event == "issue_comment":
        action = payload.get("action")
        if action == "created":
            issue = payload.get("issue") or {}
            repository = payload.get("repository", {})
            repo_full = repository.get("full_name")
            issue_number = issue.get("number")
            if not repo_full or issue_number is None:
                raise HTTPException(status_code=400, detail="Missing repository or issue_number")
            session_id = create_or_get_session(repo_full, int(issue_number))
            log(session_id=session_id, level="info", message="issue_comment.created received", source="webhook")
            return {"ok": True, "received": "issue_comment.created", "session_id": session_id}
        return {"ok": True, "ignored_action": action}

    return {"ok": True, "ignored_event": event}


