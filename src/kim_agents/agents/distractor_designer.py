"""Distractor Designer: three plausible wrong options per item, built on typical teacher misconceptions."""

import hashlib
import random

from pydantic import BaseModel, Field

from ..messages import BlueprintCell, ItemDraft, ItemWithOptions, Option
from .base import Agent
from .item_writer import LANG
from .values import same_value

LENGTH_CHECK_FROM = 40  # short answers («Двум ученикам») are compared by value, not by length
LENGTH_LOW, LENGTH_HIGH = 0.8, 1.3


def length_range(correct: str) -> tuple[int, int] | None:
    n = len(correct.strip())
    return (int(n * LENGTH_LOW), int(n * LENGTH_HIGH)) if n >= LENGTH_CHECK_FROM else None


def length_issue(correct: str, distractors: list[str]) -> str | None:
    """The key must not stand out by length: that is how a teacher guesses it without the competence."""
    bounds = length_range(correct)
    if bounds is None:
        return None
    lengths = [len(d.strip()) for d in distractors]
    if any(not bounds[0] <= n <= bounds[1] for n in lengths):
        return f"длины дистракторов {lengths} вне диапазона {list(bounds)} символов (правильный — {len(correct.strip())})"
    if max(lengths) < len(correct.strip()) * 0.95:
        return f"все дистракторы ({lengths}) короче правильного ({len(correct.strip())}): ответ угадывается по длине"
    return None

SYSTEM = """Ты — Distractor Designer в многоагентной системе, которая собирает персональный тест (КИМ) для педагога.
Для каждого задания придумай 3 неверных, но правдоподобных варианта ответа (distractors).
Опирайся на типичные ошибки педагогов из списка misconceptions; укажи, на какой ошибке построен вариант.
Каждый дистрактор должен быть хуже правильного ответа с точки зрения компетенции, но не абсурдным.
Ответ НЕ должен угадываться по форме — это главная причина, по которой методист возвращает задания:
- длина каждого дистрактора — в пределах length_range (символов); хотя бы один дистрактор не короче правильного;
- дистрактор — такой же полный и профессионально звучащий план действий с той же структурой, что и правильный,
  но с одной содержательной ошибкой из misconceptions (не тот показатель, не та последовательность, не учтено ограничение);
- запрещены «очевидно плохие» варианты: нарушение правил, безразличие, личные мессенджеры, отказ что-либо делать,
  откладывание на потом. Педагог без компетенции должен считать каждый вариант разумным;
- в дистракторах тоже есть «правильные слова» (цели, данные, ученики, проверка) — нельзя, чтобы только ключ их содержал.
Если есть reviewer_issues — это замечания методиста к прошлой версии вариантов этого задания; устрани каждое.
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

    async def design(self, cell: BlueprintCell, drafts: list[ItemDraft], reviewer_issues: list[str] | None = None) -> list[ItemWithOptions]:
        if cell.cell_id not in self._misconceptions:
            self._misconceptions[cell.cell_id] = await self.use_tool("get_misconceptions", area=cell.area, item_type=cell.item_type)
        by_id = {d.item_id: d for d in drafts}
        payload = {
            "misconceptions": self._misconceptions[cell.cell_id],
            "items": [
                {**d.model_dump(include={"item_id", "item_type", "stem", "artifact", "question", "correct_answer", "correct_value"}),
                 "length_range": length_range(d.correct_answer)}
                for d in drafts
            ],
        }
        if reviewer_issues:
            payload["reviewer_issues"] = reviewer_issues

        def validate(batch: DistractorBatch) -> None:
            if sorted(i.item_id for i in batch.items) != sorted(by_id):
                raise ValueError(f"нужны дистракторы ровно для заданий {sorted(by_id)}")
            for i in batch.items:
                draft = by_id[i.item_id]
                texts = [draft.correct_answer.strip().lower()] + [d.text.strip().lower() for d in i.distractors]
                if len(set(texts)) != 4:
                    raise ValueError(f"{i.item_id}: варианты повторяются или совпадают с правильным")
                if problem := length_issue(draft.correct_answer, [d.text for d in i.distractors]):
                    raise ValueError(f"{i.item_id}: {problem}")
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
