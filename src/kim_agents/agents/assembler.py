"""Assembler: turns verified items into the final test document — order, sections, an intro for the teacher.

The LLM only decides the layout. Item texts are never sent back through it, so it cannot alter a verified item,
and it never sees options or keys, so the intro cannot give answers away."""

import hashlib
import random
import re

from pydantic import BaseModel, Field

from ..messages import (
    AssembledTest,
    BlueprintCell,
    ItemWithOptions,
    TestBlueprint,
    TestSection,
)
from .base import Agent
from .item_writer import LANG

SYSTEM = """Ты — Assembler в многоагентной системе, которая собирает персональный тест (КИМ) для педагога.
Все задания уже проверены другими агентами; менять их нельзя. Ты собираешь из них итоговый документ.

1. Разбей задания на разделы (sections) и задай порядок заданий. Каждое задание — ровно в одном разделе.
   Удобный порядок: от более знакомого к более сложному; рядом задания одной компетенции;
   практические задания с таблицами не ставь все подряд в конец, если можно чередовать.
2. title раздела — короткое название по компетенции или области (до 80 символов).
   note — одно предложение, что проверяет раздел (можно пустую строку).
3. intro — 2–4 предложения, обращение к педагогу на «вы»: зачем этот тест, по каким областям он составлен
   (опирайся на rationale ячеек), сколько заданий и как отвечать (выбрать один вариант).
   Не раскрывай правильные ответы и не намекай на них; не оценивай педагога.
Язык: {language}."""

INTRO_MAX = 900
OPTION_IDS = ("A", "B", "C", "D")
# Feedback like «в отличие от варианта B» would break if the letters were moved.
LETTER_REFERENCE = re.compile(r"\b(?:[Вв]ариант\w*|[Oo]ption)\s+[ABCD]\b|\b[ABCD]\)")


class SectionPlan(BaseModel):
    title: str = Field(min_length=1, max_length=80)
    note: str = ""
    item_ids: list[str] = Field(min_length=1)


class AssemblyPlan(BaseModel):
    intro: str = Field(min_length=1, max_length=INTRO_MAX)
    sections: list[SectionPlan] = Field(min_length=1)


def _slot(item: ItemWithOptions) -> str:
    # Revised items keep their slot: c1-i1-r1 goes where c1-i1 was.
    return item.item_id.split("-r")[0]


def default_layout(blueprint: TestBlueprint, items: list[ItemWithOptions]) -> list[TestSection]:
    """One section per blueprint cell, in blueprint order. Used when the Assembler is unavailable."""
    sections = []
    for cell in blueprint.cells:
        cell_items = sorted((i for i in items if i.draft.cell_id == cell.cell_id), key=_slot)
        if cell_items:
            sections.append(TestSection(title=f"{cell.area_name} · {cell.competence}", items=cell_items))
    return sections


def _refers_to_letters(item: ItemWithOptions) -> bool:
    texts = [item.draft.key_rationale] + [t for o in item.options for t in (o.text, o.feedback)]
    return any(LETTER_REFERENCE.search(t) for t in texts)


def _move_key(item: ItemWithOptions, target: str) -> ItemWithOptions:
    if item.key_id == target or _refers_to_letters(item):
        return item
    key = item.key_id

    def relabel(o):
        if o.id == key:
            return o.model_copy(update={"id": target})
        if o.id == target:
            return o.model_copy(update={"id": key})
        return o

    options = sorted((relabel(o) for o in item.options), key=lambda o: o.id)
    return item.model_copy(update={"options": options, "key_id": target})


def balance_keys(items: list[ItemWithOptions], seed: str) -> list[ItemWithOptions]:
    """Spread key letters evenly over the whole test, so the answer cannot be guessed from a letter pattern.

    Each item was shuffled on its own, so across a short test the keys can bunch up on one letter.
    Swapping the key with another option is safe: verification does not depend on letters."""
    targets = [OPTION_IDS[n % len(OPTION_IDS)] for n in range(len(items))]
    random.Random(int(hashlib.sha256(seed.encode()).hexdigest(), 16)).shuffle(targets)
    return [_move_key(item, t) for item, t in zip(items, targets, strict=True)]


def _finish(sections: list[TestSection], intro: str, by_llm: bool, seed: str) -> AssembledTest:
    balanced = iter(balance_keys([i for s in sections for i in s.items], seed))
    sections = [s.model_copy(update={"items": [next(balanced) for _ in s.items]}) for s in sections]
    return AssembledTest(intro=intro, sections=sections, by_llm=by_llm)


def fallback_test(blueprint: TestBlueprint, items: list[ItemWithOptions]) -> AssembledTest:
    return _finish(default_layout(blueprint, items), "", by_llm=False, seed=_seed(blueprint, items))


def _seed(blueprint: TestBlueprint, items: list[ItemWithOptions]) -> str:
    return blueprint.teacher_id + "|" + "|".join(sorted(i.item_id for i in items))


def _cell_view(c: BlueprintCell) -> dict:
    return c.model_dump(include={"cell_id", "area", "area_name", "competence", "target_level", "item_type", "rationale"})


class Assembler(Agent):
    name = "assembler"
    tool_names = ()

    def __init__(self, ctx, language: str = "ru"):
        super().__init__(ctx)
        self.language = LANG.get(language, language)

    async def assemble(self, blueprint: TestBlueprint, items: list[ItemWithOptions]) -> AssembledTest:
        by_id = {i.item_id: i for i in items}
        payload = {
            "current_level": blueprint.current_level,
            "cells": [_cell_view(c) for c in blueprint.cells],
            "items": [
                {"item_id": i.item_id, "cell_id": i.draft.cell_id, "competence": i.draft.competence,
                 "item_type": i.draft.item_type, "stem": i.draft.stem[:300], "question": i.draft.question}
                for i in items
            ],
        }

        def validate(plan: AssemblyPlan) -> None:
            placed = [item_id for s in plan.sections for item_id in s.item_ids]
            if sorted(placed) != sorted(by_id):
                raise ValueError(f"каждое из заданий {sorted(by_id)} должно стоять ровно в одном разделе; получено {placed}")

        plan = await self.ask_llm(
            "assemble_test", system=SYSTEM.format(language=self.language), payload=payload,
            schema=AssemblyPlan, validate=validate,
        )
        sections = [
            TestSection(title=s.title.strip(), note=s.note.strip(), items=[by_id[i] for i in s.item_ids])
            for s in plan.sections
        ]
        return _finish(sections, plan.intro.strip(), by_llm=True, seed=_seed(blueprint, items))
