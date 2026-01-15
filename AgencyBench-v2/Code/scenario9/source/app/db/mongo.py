from typing import Optional

from pymongo import ASCENDING, MongoClient

from app.settings import get_env, require_env


_client: Optional[MongoClient] = None


def get_mongo_client() -> MongoClient:
    global _client
    if _client is None:
        mongo_url = require_env("MONGODB_URL")
        _client = MongoClient(mongo_url, appname="agent-mvp")
    return _client


def get_database_name() -> str:
    return get_env("MONGODB_DB", "agent_mvp") or "agent_mvp"


def get_db():
    return get_mongo_client()[get_database_name()]


def init_indexes() -> None:
    db = get_db()
    sessions = db["agent_sessions"]
    logs = db["agent_logs"]

    # agent_sessions indexes (skip if no permission)
    try:
        sessions.create_index([("session_id", ASCENDING)], unique=True, name="session_id_unique")
        sessions.create_index(
            [("repository", ASCENDING), ("issue_number", ASCENDING)],
            unique=True,
            name="repo_issue_unique",
        )
        sessions.create_index([("status", ASCENDING)], name="status_idx")
    except Exception:
        pass  # Skip if no permission

    # agent_logs indexes (skip if no permission)
    try:
        logs.create_index([("session_id", ASCENDING)], name="logs_session_id_idx")
        logs.create_index([("timestamp", ASCENDING)], name="logs_timestamp_idx")
    except Exception:
        pass  # Skip if no permission


