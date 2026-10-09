"""Agent base: tool whitelist (≤ 5), step budget, logged LLM calls with retries and one repair attempt."""

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from ..calllog import CallLog
from ..config import Settings
from ..llm import FatalLLMError, LLMClient, LLMRequest, TransientLLMError
from ..messages import Envelope
from ..state import StateStore
from ..tools import Toolbox

MAX_TOOLS_PER_AGENT = 5


class AgentError(Exception):
    def __init__(self, agent: str, message: str):
        super().__init__(f"{agent}: {message}")
        self.agent = agent


class StepLimitExceeded(Exception):
    pass


@dataclass
class RunContext:
    run_id: str
    settings: Settings
    llm: LLMClient
    log: CallLog
    state: StateStore
    toolbox: Toolbox
    on_message: Callable[[Envelope], None] | None = None
    steps: int = 0
    inject_wrong_key: bool = False
    injected: bool = False
    accepted_texts: dict[str, str] = field(default_factory=dict)  # item_id → text, for in-run duplicate checks

    def consume_step(self, agent: str) -> None:
        self.steps += 1
        if self.steps > self.settings.limits.max_steps:
            self.log.event("step_limit", agent=agent, steps=self.steps)
            raise StepLimitExceeded(f"превышен лимит шагов ({self.settings.limits.max_steps})")

    def send(self, sender: str, recipient: str, body: BaseModel, attempt: int = 1) -> Envelope:
        env = Envelope.wrap(self.run_id, sender, recipient, body, attempt)
        self.state.put_message(env)
        self.log.event("message", sender=sender, recipient=recipient, type=env.type, msg_id=env.msg_id)
        if self.on_message:
            self.on_message(env)
        return env


class Agent:
    name: str = "agent"
    tool_names: tuple[str, ...] = ()

    def __init__(self, ctx: RunContext):
        if len(self.tool_names) > MAX_TOOLS_PER_AGENT:
            raise ValueError(f"{self.name}: больше {MAX_TOOLS_PER_AGENT} инструментов")
        self.ctx = ctx
        self._tools = {n: ctx.toolbox.get(n) for n in self.tool_names}

    async def use_tool(self, name: str, **kwargs):
        if name not in self._tools:
            raise AgentError(self.name, f"инструмент {name} не разрешён этому агенту")
        self.ctx.consume_step(self.name)
        started = time.perf_counter()
        try:
            result = await asyncio.to_thread(self._tools[name], **kwargs)
        except Exception as e:
            self.ctx.log.call(agent=self.name, kind="tool", name=name, duration_ms=_ms(started), status="error", error=str(e))
            raise
        self.ctx.log.call(agent=self.name, kind="tool", name=name, duration_ms=_ms(started), status="ok")
        return result

    async def ask_llm[T: BaseModel](
        self,
        task: str,
        *,
        system: str,
        payload: dict,
        schema: type[T],
        validate: Callable[[T], None] | None = None,
    ) -> T:
        """One logical LLM request. Transient failures are retried with backoff; an invalid answer gets one repair."""
        limits = self.ctx.settings.limits
        system_full = (
            f"{system}\n\nОтвечай только JSON-объектом, соответствующим этой JSON Schema:\n"
            f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
        )
        user = json.dumps(payload, ensure_ascii=False, indent=1)
        repair_note = ""
        retries = repairs = 0
        attempt = 0
        while True:
            attempt += 1
            self.ctx.consume_step(self.name)
            request = LLMRequest(
                agent=self.name, task=task, model=self.ctx.settings.models[self.name],
                system=system_full, user=user + repair_note, payload=payload,
            )
            started = time.perf_counter()
            try:
                resp = await asyncio.wait_for(self.ctx.llm.complete(request, limits.llm_timeout_s), limits.llm_timeout_s + 5)
            except (TransientLLMError, TimeoutError) as e:
                retries += 1
                gave_up = retries > limits.llm_retries
                self._log_llm(task, started, "error" if gave_up else "retry", attempt, error=str(e) or type(e).__name__)
                if gave_up:
                    raise AgentError(self.name, f"LLM недоступна после {limits.llm_retries} повторов ({e or type(e).__name__})") from e
                await asyncio.sleep(limits.retry_backoff_s * 2 ** (retries - 1))
                continue
            except FatalLLMError as e:
                self._log_llm(task, started, "error", attempt, error=str(e))
                raise AgentError(self.name, f"запрос к LLM отклонён: {e}") from e
            try:
                result = schema.model_validate(json.loads(resp.text))
                if validate:
                    validate(result)
            except (ValueError, ValidationError) as e:
                repairs += 1
                self._log_llm(task, started, "invalid", attempt, resp.tokens_in, resp.tokens_out, error=str(e)[:500])
                if repairs > 1:
                    raise AgentError(self.name, f"ответ LLM не прошёл проверку дважды: {str(e)[:300]}") from e
                repair_note = f"\n\nПредыдущий ответ отклонён: {str(e)[:800]}\nИсправь ошибки и верни корректный JSON."
                continue
            self._log_llm(task, started, "ok", attempt, resp.tokens_in, resp.tokens_out)
            return result

    def _log_llm(self, task, started, status, attempt, tokens_in=None, tokens_out=None, error=None) -> None:
        self.ctx.log.call(
            agent=self.name, kind="llm", name=task, duration_ms=_ms(started), status=status,
            tokens_in=tokens_in, tokens_out=tokens_out, error=error, attempt=attempt,
        )


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
