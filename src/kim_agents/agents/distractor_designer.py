"""Distractor Designer: three plausible wrong options per item, built on typical teacher misconceptions."""

import hashlib
import random

from pydantic import BaseModel, Field

from ..messages import BlueprintCell, ItemDraft, ItemWithOptions, Option
from .base import Agent
from .item_writer import LANG
from .values import same_value

SYSTEM = """Ты — Distractor Designer в многоагентной системе, которая собирает персональный тест (КИМ) для педагога.
Для каждого задания придумай 3 неверных, но правдоподобных варианта ответа (distractors).
Опирайся на типичные ошибки педагогов из списка misconceptions; укажи, на какой ошибке построен вариант.
Каждый дистрактор должен быть явно хуже правильного ответа с точки зрения компетенции, но не абсурдным,
и похожим на правильный по длине и стилю, чтобы ответ не угадывался по форме.
Для practical: дистрактор — решение, к которому приводит типичная ошибка ИНТЕРПРЕТАЦИИ данных
(не тот показатель, один источник вместо двух, итог вместо прироста), а не арифметический промах;
value — значение, которое получится при этой ошибке; оно должно отличаться от correct_value.
Напиши обратную связь (1–2 предложения) для правильного ответа (correct_feedback) и для каждого дистрактора.
Язык: {language}."""


class Distractor(BaseModel):
    text: str
    value: str | None = None
    feedback: str
    misconception: str


class ItemDistractors(BaseModel):
    item_id: str
    correct_feedback: str
    distractors: list[Distractor] = Field(min_length=3, max_length=3)


class DistractorBatch(BaseModel):
    items: list[ItemDistractors]


class DistractorDesigner(Agent):
    name = "distractor_designer"
    tool_names = ("get_misconceptions",)

    def __init__(self, ctx, language: str = "ru"):
        super().__init__(ctx)
        self.language = LANG.get(language, language)
        self._misconceptions: dict[str, list[str]] = {}

    async def design(self, cell: BlueprintCell, drafts: list[ItemDraft]) -> list[ItemWithOptions]:
        if cell.cell_id not in self._misconceptions:
            self._misconceptions[cell.cell_id] = await self.use_tool("get_misconceptions", area=cell.area, item_type=cell.item_type)
        by_id = {d.item_id: d for d in drafts}
        payload = {
            "misconceptions": self._misconceptions[cell.cell_id],
            "items": [
                d.model_dump(include={"item_id", "item_type", "stem", "artifact", "question", "correct_answer", "correct_value"})
                for d in drafts
            ],
        }

        def validate(batch: DistractorBatch) -> None:
            if sorted(i.item_id for i in batch.items) != sorted(by_id):
                raise ValueError(f"нужны дистракторы ровно для заданий {sorted(by_id)}")
            for i in batch.items:
                draft = by_id[i.item_id]
                texts = [draft.correct_answer.strip().lower()] + [d.text.strip().lower() for d in i.distractors]
                if len(set(texts)) != 4:
                    raise ValueError(f"{i.item_id}: варианты повторяются или совпадают с правильным")
                if draft.item_type == "practical":
                    for d in i.distractors:
                        if d.value is None or same_value(d.value, draft.correct_value):
                            raise ValueError(f"{i.item_id}: value дистрактора пусто или равно правильному ответу")

        batch = await self.ask_llm(
            "design_distractors", system=SYSTEM.format(language=self.language), payload=payload,
            schema=DistractorBatch, validate=validate,
        )
        return [self._assemble(by_id[i.item_id], i) for i in batch.items]

    @staticmethod
    def _assemble(draft: ItemDraft, d: ItemDistractors) -> ItemWithOptions:
        options = [(draft.correct_answer, draft.correct_value, d.correct_feedback, None, True)]
        options += [(x.text, x.value, x.feedback, x.misconception, False) for x in d.distractors]
        # Shuffle with a seed from the item id: the key lands in a different place per item, reproducibly.
        random.Random(int(hashlib.sha256(draft.item_id.encode()).hexdigest(), 16)).shuffle(options)
        ids = ["A", "B", "C", "D"]
        built = [Option(id=ids[n], text=t, value=v, feedback=f, misconception=m) for n, (t, v, f, m, _) in enumerate(options)]
        key = ids[next(n for n, o in enumerate(options) if o[4])]
        return ItemWithOptions(draft=draft, options=built, key_id=key)
