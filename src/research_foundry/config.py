from __future__ import annotations

import os
from pydantic import BaseModel, Field


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Settings(BaseModel):
    """Runtime settings for OpenAI-backed research workflows."""

    openai_api_key: str | None = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    deep_research_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_DEEP_RESEARCH_MODEL", "o3-deep-research")
    )
    fast_deep_research_model: str = Field(
        default_factory=lambda: os.getenv(
            "OPENAI_FAST_DEEP_RESEARCH_MODEL", "o4-mini-deep-research"
        )
    )
    frontier_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_FRONTIER_MODEL", "gpt-5.5")
    )
    reviewer_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_REVIEWER_MODEL", "gpt-5.5-pro")
    )
    background_research: bool = Field(
        default_factory=lambda: _env_bool("RESEARCH_FOUNDRY_BACKGROUND", True)
    )
    max_wait_seconds: int = Field(
        default_factory=lambda: int(os.getenv("RESEARCH_FOUNDRY_MAX_WAIT_SECONDS", "900"))
    )
    poll_seconds: float = Field(
        default_factory=lambda: float(os.getenv("RESEARCH_FOUNDRY_POLL_SECONDS", "5"))
    )
    output_dir: str = Field(default_factory=lambda: os.getenv("RESEARCH_FOUNDRY_OUTPUT_DIR", "runs"))

