from typing import Optional

from fastapi import BackgroundTasks

from app.github.cli import comment_issue
from app.logs.service import log


def async_comment_ack(
    tasks: BackgroundTasks,
    *,
    session_id: str,
    repository: str,
    issue_number: int,
    message: str = "收到任务，开始处理",
) -> None:
    def _do():
        try:
            comment_issue(repository, issue_number, message)
            log(session_id=session_id, level="info", message="Commented ack to issue", source="github_api")
        except Exception as e:
            log(session_id=session_id, level="error", message=f"Comment failed: {e}", source="github_api")

    tasks.add_task(_do)


