from typing import Literal, Optional

from app.logs.db_logger import write_log
from app.logs.models import AgentLog
from app.utils.time import now_utc_iso


def log(
    *,
    session_id: str,
    level: Literal["info", "debug", "error", "warning"],
    message: str,
    source: Literal["agent", "webhook", "github_api"],
) -> None:
    entry: AgentLog = {
        "session_id": session_id,
        "timestamp": now_utc_iso(),
        "level": level,
        "message": message,
        "source": source,
    }
    try:
        write_log(entry)
    except Exception:
        # Fallback: just print to console if DB fails
        print(f"[{entry['timestamp']}] {level.upper()} {source}: {message}")


