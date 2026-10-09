"""Profile Analyst: finds the teacher's weak DigCompEdu areas and turns them into a test blueprint."""

from pydantic import BaseModel

from ..messages import LEVELS, BlueprintCell, ItemType, Level, TeacherProfile, TestBlueprint
from .base import Agent, AgentError

WEAK_MEAN = 2.0  # area mean on the 0–4 scale below which the area is weak
CRITICAL_MEAN = 1.5  # ... and below which it is priority 1
MAX_ZONES = 3
MIN_ITEMS_PER_ZONE = 2

SYSTEM = """Ты — Profile Analyst в многоагентной системе, которая собирает персональный тест (КИМ) для педагога.
Слабые области DigCompEdu уже отобраны по правилам, число заданий на область уже распределено — не меняй их.
Для каждой области:
1. выбери ОДНУ компетенцию (код вопроса опросника), которую важнее всего проверить: обычно ту, где самооценка ниже;
2. выбери тип задания из allowed_item_types: "sjt" — ситуационный кейс с выбором лучшего действия,
   "practical" — работа с таблицей данных и однозначно вычисляемый ответ; practical уместен, когда компетенция про данные или ресурсы;
3. в rationale одним-двумя предложениями по-русски объясни выбор, ссылаясь на баллы."""


class CellChoice(BaseModel):
    area: int
    competence: str
    item_type: ItemType
    rationale: str


class AnalystChoice(BaseModel):
    cells: list[CellChoice]


def level_for_total(total: float, levels: dict) -> Level:
    for code in LEVELS:
        if total <= levels[code]["max"]:
            return code
    return "C2"


def next_level(level: Level) -> Level:
    return LEVELS[min(LEVELS.index(level) + 1, len(LEVELS) - 1)]


def area_means(profile: TeacherProfile, areas: list[dict]) -> dict[int, float]:
    return {
        a["id"]: sum(profile.digcompedu_answers[int(c["code"][1:]) - 1] for c in a["competences"]) / len(a["competences"])
        for a in areas
    }


def select_zones(means: dict[int, float], items_total: int) -> list[tuple[int, int, int]]:
    """Returns [(area, priority, count)], weakest first. With no weak area, the two lowest are checked."""
    ranked = sorted(means, key=lambda a: (means[a], a))
    weak = [a for a in ranked if means[a] < WEAK_MEAN]
    zones = [(a, 1 if means[a] < CRITICAL_MEAN else 2) for a in weak] or [(a, 2) for a in ranked[:2]]
    zones = zones[: max(1, min(MAX_ZONES, items_total // MIN_ITEMS_PER_ZONE))]
    counts = {a: MIN_ITEMS_PER_ZONE for a, _ in zones}
    order = [a for a, p in zones if p == 1] + [a for a, p in zones if p == 2]
    for i in range(items_total - MIN_ITEMS_PER_ZONE * len(zones)):
        counts[order[i % len(order)]] += 1
    return [(a, p, counts[a]) for a, p in zones]


class ProfileAnalyst(Agent):
    name = "profile_analyst"
    tool_names = ("load_profile", "get_framework_overview")

    async def run(self, profile_path: str) -> tuple[TeacherProfile, TestBlueprint]:
        profile: TeacherProfile = await self.use_tool("load_profile", path=profile_path)
        levels = self.ctx.toolbox.framework["levels"]
        all_areas = self.ctx.toolbox.framework["areas"]
        means = area_means(profile, all_areas)
        current = level_for_total(sum(profile.digcompedu_answers), levels)
        zones = select_zones(means, profile.items_total)

        overview = await self.use_tool("get_framework_overview", areas=[a for a, _, _ in zones])
        by_area = {o["area"]: o for o in overview}
        payload = {
            "teacher_level": current,
            "zones": [
                {
                    "area": a,
                    "area_name": by_area[a]["area_name"],
                    "area_mean_0_4": round(means[a], 2),
                    "priority": p,
                    "items": n,
                    "allowed_item_types": by_area[a]["allowed_item_types"],
                    "competences": [
                        {**c, "self_score_0_4": profile.digcompedu_answers[int(c["code"][1:]) - 1]}
                        for c in by_area[a]["competences"]
                    ],
                }
                for a, p, n in zones
            ],
        }

        def validate(choice: AnalystChoice) -> None:
            got = sorted(c.area for c in choice.cells)
            if got != sorted(by_area):
                raise ValueError(f"нужны ровно области {sorted(by_area)}, получено {got}")
            for c in choice.cells:
                codes = [x["code"] for x in by_area[c.area]["competences"]]
                if c.competence not in codes:
                    raise ValueError(f"компетенция {c.competence} не из области {c.area}; допустимы {codes}")
                if c.item_type not in by_area[c.area]["allowed_item_types"]:
                    raise ValueError(f"тип {c.item_type} не допустим для области {c.area}")

        choice = await self.ask_llm("choose_cells", system=SYSTEM, payload=payload, schema=AnalystChoice, validate=validate)
        picked = {c.area: c for c in choice.cells}
        cells = []
        for n, (area, priority, count) in enumerate(zones, start=1):
            c = picked[area]
            cells.append(
                BlueprintCell(
                    cell_id=f"c{n}",
                    area=area,
                    area_name=by_area[area]["area_name"],
                    competence=c.competence,
                    target_level=next_level(level_for_total(means[area] * 22, levels)),
                    item_type=c.item_type,
                    count=count,
                    priority=priority,
                    rationale=c.rationale,
                )
            )
        try:
            blueprint = TestBlueprint(
                teacher_id=profile.teacher_id,
                language=profile.language,
                current_level=current,
                items_total=profile.items_total,
                cells=cells,
            )
        except ValueError as e:
            raise AgentError(self.name, f"спецификация теста не собралась: {e}") from e
        return profile, blueprint
