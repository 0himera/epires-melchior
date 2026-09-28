"""Configuration management for Melchior."""

import shutil
import os
from pydantic import BaseModel, Field


def _default_llm_mode() -> str:
    if os.getenv("MELCHIOR_LLM_MODE"):
        return os.getenv("MELCHIOR_LLM_MODE")
    if os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
        return "api"
    if shutil.which("opencode"):
        return "opencode"
    return "mock"


class Settings(BaseModel):
    jev_mode: str = Field(
        default_factory=lambda: os.getenv(
            "MELCHIOR_JEV_MODE",
            "api" if os.getenv("JEV_API_KEY") else "mock",
        )
    )
    jev_api_key: str | None = Field(default_factory=lambda: os.getenv("JEV_API_KEY"))
    jev_api_url: str = Field(
        default_factory=lambda: os.getenv(
            "JEV_API_URL", "https://api.typesafe.ai/v1/systemone"
        )
    )

    llm_mode: str = Field(default_factory=_default_llm_mode)
    llm_model: str = Field(
        default_factory=lambda: os.getenv("MELCHIOR_LLM_MODEL", "gpt-4o")
    )
    opencode_model: str | None = Field(
        default_factory=lambda: os.getenv("MELCHIOR_OPENCODE_MODEL")
    )
    llm_base_url: str = Field(
        default_factory=lambda: os.getenv(
            "OPENAI_BASE_URL",
            os.getenv("MELCHIOR_LLM_BASE_URL", "https://api.openai.com/v1"),
        )
    )
    openai_api_key: str | None = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    anthropic_api_key: str | None = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )

    default_timeout_seconds: int = 180
    pilot_timeout_seconds: int = 60


config = Settings()
