import uuid
from typing import Optional

from app.db.mongo import get_db
from app.utils.time import now_utc_iso


def create_or_get_session(repository: str, issue_number: int) -> str:
    """Create a new session for repo+issue if not exists; return session_id.
    Enforces the unique index (repository, issue_number).
    """
    try:
        db = get_db()
        sessions = db["agent_sessions"]
        existing = sessions.find_one({"repository": repository, "issue_number": issue_number})
        if existing and existing.get("session_id"):
            return existing["session_id"]

        session_id = str(uuid.uuid4())
        now = now_utc_iso()
        doc = {
            "session_id": session_id,
            "issue_number": issue_number,
            "repository": repository,
            "status": "pending",
            "start_time": now,
            "updated_at": now,
            "created_at": now,
        }
        sessions.insert_one(doc)
        return session_id
    except Exception:
        # Fallback: return a session_id without DB storage
        return str(uuid.uuid4())


def update_session_status(session_id: str, status: str) -> None:
    db = get_db()
    db["agent_sessions"].update_one(
        {"session_id": session_id},
        {"$set": {"status": status, "updated_at": now_utc_iso()}},
    )


def set_session_sandbox_id(session_id: str, sandbox_id: str) -> None:
    db = get_db()
    db["agent_sessions"].update_one(
        {"session_id": session_id},
        {"$set": {"sandbox_id": sandbox_id, "updated_at": now_utc_iso()}},
    )


def mark_session_end(session_id: str, status: str) -> None:
    db = get_db()
    db["agent_sessions"].update_one(
        {"session_id": session_id},
        {"$set": {"status": status, "end_time": now_utc_iso(), "updated_at": now_utc_iso()}},
    )


def update_issue_info(session_id: str, title: str, body: str) -> None:
    try:
        db = get_db()
        db["agent_sessions"].update_one(
            {"session_id": session_id},
            {"$set": {"issue_title": title, "issue_body": body, "updated_at": now_utc_iso()}},
        )
    except Exception:
        # Fallback: just print to console if DB fails
        print(f"Failed to update issue info for session {session_id}: {title}")


def update_pr_info(session_id: str, pr_url: str, branch: str) -> None:
    """Update session with PR URL and branch information"""
    try:
        db = get_db()
        db["agent_sessions"].update_one(
            {"session_id": session_id},
            {"$set": {"pr_url": pr_url, "branch": branch, "updated_at": now_utc_iso()}},
        )
    except Exception:
        # Fallback: just print to console if DB fails
        print(f"Failed to update PR info for session {session_id}: {pr_url}")


