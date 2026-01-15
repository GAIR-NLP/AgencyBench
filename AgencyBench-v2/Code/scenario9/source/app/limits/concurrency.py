from app.db.mongo import get_db
from app.settings import get_env


def get_max_concurrent() -> int:
    value = get_env("MAX_CONCURRENT_AGENTS", "2")
    try:
        return int(value or 2)
    except Exception:
        return 2


def current_running_count() -> int:
    db = get_db()
    return db["agent_sessions"].count_documents({"status": "running"})


def can_start_new_agent() -> bool:
    return current_running_count() < get_max_concurrent()


