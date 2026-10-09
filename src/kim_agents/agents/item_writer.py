"""Item Writer: writes the case, the question and the key for one blueprint cell. Never writes distractors."""

from pydantic import BaseModel

from ..messages import Artifact, BlueprintCell, ItemDraft, ItemWithOptions
from .base import Agent

SYSTEM = """Ты — Item Writer в многоагентной системе, которая собирает персональный тест (КИМ) для педагога.
Ты пишешь ОСНОВУ задания по одной компетенции DigCompEdu. Неверные варианты (дистракторы) НЕ пиши — это работа другого агента.

Тип "sjt" (ситуационное задание): реалистичный кейс из практики школы или колледжа (3–5 предложений),
вопрос «Как вы поступите?» и ОДНО лучшее действие (correct_answer), которое соответствует целевому уровню дескриптора.
artifact и correct_value — null.

Тип "practical": задание на ПЕДАГОГИЧЕСКОЕ РЕШЕНИЕ по данным, а не на арифметику.
Короткий кейс с учебной целью, таблица-артефакт (columns + 5–8 строк rows) и вопрос о решении:
кому оказать поддержку, какую тему повторить, сработал ли приём, какой вывод из данных допустим.
- Сложность — в интерпретации: выбрать подходящий показатель (доля, а не абсолютное число; прирост, а не итог;
  динамика, а не последняя работа; пропуски — не нули; совмещение нескольких источников данных).
  Кейс НЕ должен подсказывать, какой показатель считать: это и есть то, что проверяет задание.
- Арифметика тривиальна: один-два простых шага.
- Вопрос вида «Сколько учеников…?» с готовым порогом в кейсе запрещён — это проверка счёта, а не компетенции.
  Предпочтительный ответ — объект решения: ученик, тема, группа, вывод.
- Ответ всё равно однозначно определяется таблицей: correct_value — это значение (имя, название темы или число),
  correct_answer — тот же ответ, сформулированный как действие или вывод педагога.

key_rationale: почему ответ лучший, со ссылкой на формулировку компетенции и уровень.
Все задания должны быть о разных ситуациях. Язык заданий: {language}."""

REVISE_SYSTEM = """Ты — Item Writer. Проверяющий вернул твоё задание с замечаниями.
Перепиши задание так, чтобы устранить каждое замечание; при необходимости придумай новую ситуацию.
Компетенция, тип и уровень остаются прежними. Дистракторы не пиши. Язык: {language}."""

LANG = {"ru": "русский", "kk": "казахский", "en": "английский"}


class WrittenItem(BaseModel):
    stem: str
    artifact: Artifact | None = None
    question: str
    correct_answer: str
    correct_value: str | None = None
    key_rationale: str


class WriterBatch(BaseModel):
    items: list[WrittenItem]


def _check_item(item: WrittenItem, item_type: str) -> None:
    if item_type == "practical" and (item.artifact is None or not item.correct_value):
        raise ValueError("у practical-задания обязательны artifact и correct_value")


class ItemWriter(Agent):
    name = "item_writer"
    tool_names = ("get_competence_descriptor", "get_bank_examples")

    def __init__(self, ctx, language: str = "ru"):
        super().__init__(ctx)
        self.language = LANG.get(language, language)
        self._context: dict[str, dict] = {}  # cell_id → descriptor + examples, fetched once per cell

    async def _cell_context(self, cell: BlueprintCell) -> dict:
        if cell.cell_id not in self._context:
            descriptor = await self.use_tool("get_competence_descriptor", competence=cell.competence, level=cell.target_level)
            examples = await self.use_tool("get_bank_examples", item_type=cell.item_type, limit=1)
            self._context[cell.cell_id] = {"descriptor": descriptor, "style_examples": examples}
        return self._context[cell.cell_id]

    async def write(self, cell: BlueprintCell) -> list[ItemDraft]:
        context = await self._cell_context(cell)
        payload = {"item_type": cell.item_type, "count": cell.count, **context}

        def validate(batch: WriterBatch) -> None:
            if len(batch.items) != cell.count:
                raise ValueError(f"нужно ровно {cell.count} заданий, получено {len(batch.items)}")
            for item in batch.items:
                _check_item(item, cell.item_type)

        batch = await self.ask_llm(
            "write_items", system=SYSTEM.format(language=self.language), payload=payload, schema=WriterBatch, validate=validate
        )
        return [self._draft(cell, w, f"{cell.cell_id}-i{n}", attempt=1) for n, w in enumerate(batch.items, start=1)]

    async def revise(self, cell: BlueprintCell, item: ItemWithOptions, issues: list[str]) -> ItemDraft:
        context = await self._cell_context(cell)
        payload = {
            "item_type": cell.item_type,
            **context,
            "previous_item": item.draft.model_dump(include={"stem", "artifact", "question", "correct_answer", "correct_value"}),
            "reviewer_issues": issues,
        }
        written = await self.ask_llm(
            "revise_item",
            system=REVISE_SYSTEM.format(language=self.language),
            payload=payload,
            schema=WrittenItem,
            validate=lambda w: _check_item(w, cell.item_type),
        )
        base_id = item.item_id.split("-r")[0]
        attempt = item.draft.attempt + 1
        return self._draft(cell, written, f"{base_id}-r{attempt - 1}", attempt=attempt)

    @staticmethod
    def _draft(cell: BlueprintCell, w: WrittenItem, item_id: str, attempt: int) -> ItemDraft:
        return ItemDraft(
            item_id=item_id,
            cell_id=cell.cell_id,
            item_type=cell.item_type,
            competence=cell.competence,
            target_level=cell.target_level,
            stem=w.stem,
            artifact=w.artifact,
            question=w.question,
            correct_answer=w.correct_answer,
            correct_value=w.correct_value if cell.item_type == "practical" else None,
            key_rationale=w.key_rationale,
            attempt=attempt,
        )
