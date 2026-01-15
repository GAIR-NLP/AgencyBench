from typing import Optional

from app.db.mongo import get_db
from app.logs.models import AgentLog


def write_log(log: AgentLog) -> None:
    db = get_db()
    db["agent_logs"].insert_one(dict(log))


