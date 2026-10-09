"""Alignment Reviewer: the methodologist. Checks that an item measures its competence at its level,
and not something else, such as arithmetic or common sense."""

from pydantic import BaseModel

from ..messages import AlignmentVerdict, BlueprintCell, ItemWithOptions
from .base import Agent

SYSTEM = """Ты — Alignment Reviewer, методист по цифровым компетенциям педагогов (DigCompEdu).
Ты НЕ проверяешь правильность ключа — это делает другой агент. Ты проверяешь, что задание измеряет
именно заявленную компетенцию на заявленном уровне. Дескриптор компетенции и уровня — в descriptor.

Для каждого задания верни:
- competence_match — задание проверяет именно эту компетенцию (а не соседнюю или общую грамотность);
- level_match — сложность и характер решения соответствуют целевому уровню дескриптора;
- requires_competence — без этой компетенции задание НЕ решить. false, если хватает общих умений:
  арифметики, внимательного чтения таблицы, здравого смысла, «социально желательного» выбора по форме ответа;
- solvable_by — если requires_competence = false: каким общим умением решается задание; иначе null;
- issues — конкретные замечания для автора: что изменить, чтобы задание проверяло компетенцию.

Для "practical" особенно важно: задание на анализ данных должно требовать педагогической интерпретации
(какой показатель выбрать, какой вывод допустим, какое решение принять), а не только подсчёта.
Вопрос вида «сколько строк удовлетворяют условию» при готовом условии в кейсе — это арифметика, requires_competence = false."""


class ItemReview(BaseModel):
    item_id: str
    competence_match: bool
    level_match: bool
    requires_competence: bool
    solvable_by: str | None = None
    issues: list[str] = []


class ReviewBatch(BaseModel):
    items: list[ItemReview]


class AlignmentReviewer(Agent):
    name = "alignment_reviewer"
    tool_names = ("get_competence_descriptor",)

    def __init__(self, ctx):
        super().__init__(ctx)
        self._descriptors: dict[str, dict] = {}

    async def review(self, cell: BlueprintCell, items: list[ItemWithOptions]) -> list[AlignmentVerdict]:
        if cell.cell_id not in self._descriptors:
            self._descriptors[cell.cell_id] = await self.use_tool(
                "get_competence_descriptor", competence=cell.competence, level=cell.target_level
            )
        by_id = {i.item_id: i for i in items}
        payload = {
            "descriptor": self._descriptors[cell.cell_id],
            "items": [
                {
                    **i.blind_view(),
                    "key": i.key_id,
                    "key_rationale": i.draft.key_rationale,
                    "misconceptions": {o.id: o.misconception for o in i.options if o.id != i.key_id},
                }
                for i in items
            ],
        }

        def validate(batch: ReviewBatch) -> None:
            if sorted(r.item_id for r in batch.items) != sorted(by_id):
                raise ValueError(f"нужны оценки ровно для заданий {sorted(by_id)}")

        batch = await self.ask_llm("review_alignment", system=SYSTEM, payload=payload, schema=ReviewBatch, validate=validate)
        reviews = {r.item_id: r for r in batch.items}
        return [self._judge(cell, reviews[item_id]) for item_id in by_id]

    @staticmethod
    def _judge(cell: BlueprintCell, r: ItemReview) -> AlignmentVerdict:
        issues = []
        if not r.competence_match:
            issues.append(f"Задание не проверяет компетенцию {cell.competence}.")
        if not r.level_match:
            issues.append(f"Сложность не соответствует уровню {cell.target_level}.")
        if not r.requires_competence:
            issues.append(
                f"Задание решается без компетенции {cell.competence}"
                + (f": достаточно умения «{r.solvable_by}»." if r.solvable_by else ".")
            )
        if issues:
            issues += [f"Методист: {x}" for x in r.issues]
        return AlignmentVerdict(
            item_id=r.item_id,
            decision="revise" if issues else "accept",
            competence_match=r.competence_match,
            level_match=r.level_match,
            requires_competence=r.requires_competence,
            solvable_by=r.solvable_by,
            issues=issues,
        )
