"""Quality Verifier: solves items without the key, recomputes practical answers in code, looks for duplicates."""

from pydantic import BaseModel, Field

from ..messages import BlueprintCell, ItemWithOptions, OptionId, QualityVerdict
from .base import Agent
from .values import same_value

SYSTEM = """Ты — Quality Verifier, независимый эксперт по цифровым компетенциям педагогов (DigCompEdu).
Ключ к заданиям тебе НЕ показан: реши каждое задание сам.
Для каждого задания верни:
- answer — вариант, который ты считаешь лучшим;
- confidence — уверенность от 0 до 1;
- defensible_alternatives — другие варианты, которые квалифицированный эксперт тоже мог бы обоснованно назвать лучшими
  (пустой список, если лучший вариант однозначен);
- blocking_issues — только серьёзные проблемы, из-за которых задание нельзя использовать:
  непонятное условие, неправдоподобный кейс, нет однозначного ответа. Стиль и мелочи не указывай.
- python_code — только для item_type "practical": код на Python (стандартная библиотека, без import os/файлов/сети),
  который по переменной artifact (dict с ключами columns и rows) вычисляет ответ на вопрос и присваивает его переменной answer
  (число или строку — как значение, а не букву варианта). Для "sjt" — null."""


class BlindSolution(BaseModel):
    item_id: str
    answer: OptionId
    confidence: float = Field(ge=0, le=1)
    defensible_alternatives: list[OptionId] = []
    blocking_issues: list[str] = []
    python_code: str | None = None


class BlindBatch(BaseModel):
    items: list[BlindSolution]


class QualityVerifier(Agent):
    name = "quality_verifier"
    tool_names = ("similarity_search", "run_python")

    async def verify(self, cell: BlueprintCell, items: list[ItemWithOptions]) -> list[QualityVerdict]:
        by_id = {i.item_id: i for i in items}

        def validate(batch: BlindBatch) -> None:
            if sorted(s.item_id for s in batch.items) != sorted(by_id):
                raise ValueError(f"нужны решения ровно для заданий {sorted(by_id)}")
            for s in batch.items:
                if by_id[s.item_id].draft.item_type == "practical" and not s.python_code:
                    raise ValueError(f"{s.item_id}: для practical нужен python_code")

        batch = await self.ask_llm(
            "blind_solve", system=SYSTEM, payload={"items": [i.blind_view() for i in items]}, schema=BlindBatch, validate=validate
        )
        solutions = {s.item_id: s for s in batch.items}

        similarity = await self.use_tool(
            "similarity_search",
            texts={i.item_id: f"{i.draft.stem} {i.draft.question}" for i in items},
            extra_corpus=dict(self.ctx.accepted_texts),
        )
        jobs = [
            {"id": i.item_id, "code": solutions[i.item_id].python_code, "artifact": i.draft.artifact.model_dump()}
            for i in items
            if i.draft.item_type == "practical"
        ]
        recomputed = await self.use_tool("run_python", jobs=jobs) if jobs else {}
        return [self._judge(by_id[s], solutions[s], similarity[s], recomputed.get(s)) for s in by_id]

    def _judge(self, item: ItemWithOptions, sol: BlindSolution, sim: dict, run: dict | None) -> QualityVerdict:
        limits = self.ctx.settings.limits
        issues = []
        key_match = sol.answer == item.key_id
        if not key_match:
            issues.append(f"Слепое решение выбрало {sol.answer}, а ключ — {item.key_id}: ключ неверен или неоднозначен.")
        alternatives = [a for a in sol.defensible_alternatives if a != sol.answer]
        if alternatives:
            issues.append(f"Варианты {', '.join(alternatives)} тоже можно защитить как лучшие — нужен однозначный лучший ответ.")
        if key_match and sol.confidence < limits.min_blind_confidence:
            issues.append(f"Лучший вариант неочевиден даже эксперту (уверенность {sol.confidence:.2f}).")
        issues += [f"Проверяющий: {x}" for x in sol.blocking_issues]

        recomputed_value = recomputed_match = None
        if run is not None:
            if run["ok"]:
                recomputed_value = run["value"]
                recomputed_match = same_value(recomputed_value, item.key_option().value)
                if not recomputed_match:
                    issues.append(
                        f"Пересчёт по таблице дал {recomputed_value}, а ключ {item.key_id} означает {item.key_option().value}."
                    )
            else:
                # The verifier's own code failed: that says nothing about the item, so it is not a reason to reject it.
                self.ctx.log.event("recompute_failed", item_id=item.item_id, error=run["error"])

        if sim["score"] >= limits.similarity_threshold:
            issues.append(f"Задание слишком похоже на {sim['match_id']} (сходство {sim['score']:.2f}).")

        return QualityVerdict(
            item_id=item.item_id,
            decision="revise" if issues else "accept",
            blind_answer=sol.answer,
            blind_confidence=sol.confidence,
            key_match=key_match,
            defensible_alternatives=alternatives,
            recomputed_value=recomputed_value,
            recomputed_match=recomputed_match,
            max_similarity=sim["score"],
            similar_to=sim["match_id"],
            issues=issues,
        )
