import asyncio
import json
from collections import Counter
from pathlib import Path

import pytest

from kim_agents.config import AGENT_NAMES, Limits, Settings
from kim_agents.llm import LLMRequest, LLMResponse
from kim_agents.offline_llm import OfflineLLM

PROFILES = Path(__file__).parent.parent / "fixtures" / "profiles"


class ScriptedLLM:
    """OfflineLLM plus scripted faults and response rewrites per task.

    A fault entry is a list (exceptions or raw text, consumed in order) or a callable payload → exception | None.
    """

    def __init__(self, faults: dict | None = None, transform: dict | None = None, delay: float = 0.0):
        self.inner = OfflineLLM()
        self.faults = faults or {}
        self.transform = transform or {}
        self.delay = delay
        self.calls = Counter()

    async def complete(self, request: LLMRequest, timeout: float) -> LLMResponse:
        self.calls[request.task] += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        queue = self.faults.get(request.task)
        if callable(queue):
            if (fault := queue(request.payload)) is not None:
                raise fault
        elif queue:
            fault = queue.pop(0)
            if isinstance(fault, BaseException):
                raise fault
            return LLMResponse(text=fault)
        resp = await self.inner.complete(request, timeout)
        if request.task in self.transform:
            data = self.transform[request.task](request.payload, json.loads(resp.text))
            resp = LLMResponse(text=json.dumps(data, ensure_ascii=False))
        return resp


@pytest.fixture
def make_settings(tmp_path):
    def make(**limits) -> Settings:
        return Settings(
            openai_api_key=None,
            models={name: "scripted" for name in AGENT_NAMES},
            limits=Limits(retry_backoff_s=0, **limits),
            runs_dir=tmp_path / "runs",
        )

    return make


def profile(name: str) -> str:
    return str(PROFILES / name)


def log_records(settings: Settings, run_id: str) -> list[dict]:
    path = settings.runs_dir / run_id / "calls.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
