import os
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


def require_env(variable_name: str) -> str:
    value = os.getenv(variable_name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {variable_name}")
    return value


def get_env(variable_name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(variable_name, default)


def get_llm_model() -> str:
    return get_env("LLM_MODEL", "gpt-4o-mini") or "gpt-4o-mini"


