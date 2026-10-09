"""Runtime settings: models per agent and run limits, read from the environment / .env."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

AGENT_NAMES = ("profile_analyst", "item_writer", "distractor_designer", "alignment_reviewer", "quality_verifier")


@dataclass(frozen=True)
class Limits:
    max_steps: int = 80  # LLM + tool calls per run, across all agents
    run_timeout_s: float = 600.0
    llm_timeout_s: float = 90.0
    llm_retries: int = 3  # retries of a transient LLM failure, on top of the first attempt
    retry_backoff_s: float = 1.0  # first pause; doubles on every retry
    max_revisions: int = 2  # rewrites of one item before its slot is given up
    sandbox_timeout_s: float = 5.0
    similarity_threshold: float = 0.8
    min_blind_confidence: float = 0.5


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    models: dict[str, str]
    limits: Limits = field(default_factory=Limits)
    runs_dir: Path = Path("runs")


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def load_settings() -> Settings:
    load_dotenv()
    default_model = os.getenv("KIM_MODEL_DEFAULT", "gpt-6-luna")
    models = {name: os.getenv(f"KIM_MODEL_{name.upper()}", default_model) for name in AGENT_NAMES}
    # The verifier runs on a different model so it does not share the writer's blind spots.
    models["quality_verifier"] = os.getenv("KIM_MODEL_QUALITY_VERIFIER", "gpt-6.1-sol")
    defaults = Limits()
    limits = Limits(
        max_steps=_int("KIM_MAX_STEPS", defaults.max_steps),
        run_timeout_s=_float("KIM_RUN_TIMEOUT_S", defaults.run_timeout_s),
        llm_timeout_s=_float("KIM_LLM_TIMEOUT_S", defaults.llm_timeout_s),
        llm_retries=_int("KIM_LLM_RETRIES", defaults.llm_retries),
        max_revisions=_int("KIM_MAX_REVISIONS", defaults.max_revisions),
    )
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        models=models,
        limits=limits,
        runs_dir=Path(os.getenv("KIM_RUNS_DIR", "runs")),
    )
