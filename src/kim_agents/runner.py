"""Prototype orchestration. In the target architecture this is the Orchestrator agent; here it is plain code:
it routes messages and applies limits, and does no subject work and no LLM calls of its own."""

import asyncio
import hashlib
from datetime import datetime
from uuid import uuid4

from .agents.alignment_reviewer import AlignmentReviewer
from .agents.base import AgentError, RunContext, StepLimitExceeded
from .agents.distractor_designer import DistractorDesigner
from .agents.item_writer import ItemWriter
from .agents.profile_analyst import ProfileAnalyst
from .agents.quality_verifier import QualityVerifier
from .calllog import CallLog
from .config import Settings
from .llm import LLMClient
from .messages import (
    AlignmentBatch,
    BlueprintCell,
    CellResult,
    GenerationResult,
    ItemBatch,
    ItemDraftBatch,
    ItemWithOptions,
    RevisionRequest,
    VerdictBatch,
)
from .render import render_markdown
from .state import StateStore
from .tools import ProfileError, Toolbox

ORCHESTRATOR = "orchestrator"


def _fingerprint(item: ItemWithOptions) -> str:
    d = item.draft
    return hashlib.sha256(f"{d.stem}|{d.question}|{d.correct_answer}".encode()).hexdigest()


class Pipeline:
    def __init__(self, ctx: RunContext, language: str):
        self.ctx = ctx
        self.writer = ItemWriter(ctx, language)
        self.designer = DistractorDesigner(ctx, language)
        self.reviewer = AlignmentReviewer(ctx)
        self.verifier = QualityVerifier(ctx)

    def _save(self, item: ItemWithOptions, status: str, verdict=None) -> None:
        self.ctx.state.put_item(
            self.ctx.run_id, item.item_id, item.draft.cell_id, status, item.draft.attempt,
            item.model_dump(mode="json"), verdict.model_dump(mode="json") if verdict else None,
        )

    def _maybe_inject_wrong_key(self, items: list[ItemWithOptions]) -> None:
        """Fault injection for the demo: move the key of the first item to a wrong option."""
        if not self.ctx.inject_wrong_key or self.ctx.injected or not items:
            return
        item = items[0]
        wrong = next(o.id for o in item.options if o.id != item.key_id)
        self.ctx.log.event("fault_injection", item_id=item.item_id, true_key=item.key_id, injected_key=wrong)
        item.key_id = wrong
        self.ctx.injected = True

    async def process_cell(self, cell: BlueprintCell, result: CellResult, inject: bool) -> None:
        ctx, limits = self.ctx, self.ctx.settings.limits
        ctx.state.put_cell(ctx.run_id, cell.cell_id, "in_progress", cell.model_dump(mode="json"))
        ctx.send(ORCHESTRATOR, self.writer.name, cell)
        drafts = await self.writer.write(cell)
        ctx.send(self.writer.name, self.designer.name, ItemDraftBatch(cell_id=cell.cell_id, items=drafts))
        pending = await self.designer.design(cell, drafts)
        if inject:
            self._maybe_inject_wrong_key(pending)
        for item in pending:
            self._save(item, "drafted")

        while pending:
            attempt = max(i.draft.attempt for i in pending)
            ctx.send(self.designer.name, self.reviewer.name, ItemBatch(cell_id=cell.cell_id, items=pending), attempt)
            alignment = await self.reviewer.review(cell, pending)
            ctx.send(self.reviewer.name, ORCHESTRATOR, AlignmentBatch(cell_id=cell.cell_id, verdicts=alignment), attempt)
            rejected = [(item, v, self.reviewer.name) for item, v in zip(pending, alignment, strict=True) if v.decision == "revise"]
            aligned = [item for item, v in zip(pending, alignment, strict=True) if v.decision == "accept"]

            if aligned:
                ctx.send(self.reviewer.name, self.verifier.name, ItemBatch(cell_id=cell.cell_id, items=aligned), attempt)
                verdicts = await self.verifier.verify(cell, aligned)
                ctx.send(self.verifier.name, ORCHESTRATOR, VerdictBatch(cell_id=cell.cell_id, verdicts=verdicts), attempt)
                for item, verdict in zip(aligned, verdicts, strict=True):
                    if verdict.decision == "accept":
                        result.accepted.append(item)
                        ctx.accepted_texts[item.item_id] = f"{item.draft.stem} {item.draft.question}"
                        self._save(item, "accepted", verdict)
                    else:
                        rejected.append((item, verdict, self.verifier.name))

            next_round = []
            for item, verdict, reviewer in rejected:
                if item.draft.attempt > limits.max_revisions:
                    self._save(item, "rejected", verdict)
                    result.unfilled_reasons.append(
                        f"{item.item_id}: не принято после {limits.max_revisions} доработок — {' '.join(verdict.issues)}"
                    )
                    ctx.log.event("revision_limit", item_id=item.item_id)
                    continue
                self._save(item, "revise", verdict)
                ctx.send(reviewer, self.writer.name,
                         RevisionRequest(cell_id=cell.cell_id, item=item, issues=verdict.issues), item.draft.attempt + 1)
                draft = await self.writer.revise(cell, item, verdict.issues)
                ctx.send(self.writer.name, self.designer.name, ItemDraftBatch(cell_id=cell.cell_id, items=[draft]), draft.attempt)
                [revised] = await self.designer.design(cell, [draft])
                if _fingerprint(revised) == _fingerprint(item):
                    # Loop guard: a rewrite that changes nothing would be rejected forever.
                    result.unfilled_reasons.append(f"{item.item_id}: доработка не изменила задание")
                    ctx.log.event("loop_guard", item_id=item.item_id)
                    continue
                self._save(revised, "drafted")
                next_round.append(revised)
            pending = next_round

        result.status = "filled" if len(result.accepted) >= cell.count else ("partial" if result.accepted else "unfilled")
        ctx.state.put_cell(ctx.run_id, cell.cell_id, result.status, cell.model_dump(mode="json"))


def new_run_id() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:6]


async def generate(
    profile_path: str,
    *,
    settings: Settings,
    llm: LLMClient,
    run_id: str | None = None,
    inject_wrong_key: bool = False,
    on_message=None,
) -> GenerationResult:
    run_id = run_id or new_run_id()
    run_dir = settings.runs_dir / run_id
    log = CallLog(run_dir / "calls.jsonl", run_id)
    state = StateStore(settings.runs_dir / "kim.sqlite")
    ctx = RunContext(
        run_id=run_id, settings=settings, llm=llm, log=log, state=state,
        toolbox=Toolbox(sandbox_timeout_s=settings.limits.sandbox_timeout_s),
        on_message=on_message, inject_wrong_key=inject_wrong_key,
    )
    state.start_run(run_id)
    log.event("run_started", profile=profile_path, models=settings.models)

    blueprint = None
    cells: dict[str, CellResult] = {}
    problems: list[str] = []
    try:
        async with asyncio.timeout(settings.limits.run_timeout_s):
            analyst = ProfileAnalyst(ctx)
            profile, blueprint = await analyst.run(profile_path)
            state.set_teacher(run_id, profile.teacher_id)
            ctx.send(analyst.name, ORCHESTRATOR, blueprint)
            pipeline = Pipeline(ctx, profile.language)
            cells = {c.cell_id: CellResult(cell_id=c.cell_id, status="unfilled") for c in blueprint.cells}
            outcomes = await asyncio.gather(
                *(pipeline.process_cell(c, cells[c.cell_id], inject=(n == 0)) for n, c in enumerate(blueprint.cells)),
                return_exceptions=True,
            )
            for cell, outcome in zip(blueprint.cells, outcomes, strict=True):
                if isinstance(outcome, BaseException):
                    if not isinstance(outcome, (AgentError, StepLimitExceeded)):
                        raise outcome
                    cells[cell.cell_id].status = "partial" if cells[cell.cell_id].accepted else "failed"
                    cells[cell.cell_id].unfilled_reasons.append(str(outcome))
                    problems.append(f"{cell.cell_id}: {outcome}")
    except ProfileError as e:
        problems.append(str(e))
    except (AgentError, StepLimitExceeded) as e:
        problems.append(str(e))
    except TimeoutError:
        log.event("run_timeout", seconds=settings.limits.run_timeout_s)
        problems.append(f"превышено время прогона ({settings.limits.run_timeout_s:.0f} с)")

    accepted = sum(len(c.accepted) for c in cells.values())
    expected = blueprint.items_total if blueprint else 0
    if accepted and accepted >= expected and not problems:
        status, message = "ok", f"Тест собран: {accepted} заданий."
    elif accepted:
        status, message = "partial", f"Собрано {accepted} из {expected} заданий. " + " ".join(problems)
    else:
        status, message = "failed", "Тест не собран. " + (" ".join(problems) or "Ни одно задание не прошло проверку.")
    for c in cells.values():
        if c.status in ("partial", "unfilled", "failed") and c.unfilled_reasons and not problems:
            message += f" {c.cell_id}: {c.status}."

    result = GenerationResult(run_id=run_id, status=status, message=message.strip(), blueprint=blueprint,
                              cells=list(cells.values()))
    state.finish_run(run_id, status, result.message)
    state.close()
    log.event("run_finished", status=status, steps=ctx.steps)
    (run_dir / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    if blueprint:
        (run_dir / "test.md").write_text(render_markdown(result), encoding="utf-8")
    return result

