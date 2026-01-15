from datetime import datetime


def now_utc_iso() -> str:
    """Return current UTC time as ISO-8601 string without microseconds, e.g. 2024-01-01T00:00:00Z."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


