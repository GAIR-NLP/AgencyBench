from typing import Literal, Optional, TypedDict


LogLevel = Literal["info", "debug", "error", "warning"]


class AgentLog(TypedDict, total=False):
    session_id: str
    timestamp: str
    level: LogLevel
    message: str
    source: Literal["agent", "webhook", "github_api"]


