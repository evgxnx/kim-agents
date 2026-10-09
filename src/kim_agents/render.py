"""Human-readable test form in Markdown."""

from .messages import GenerationResult, ItemWithOptions

TYPE_NAMES = {"sjt": "ситуационное", "practical": "практическое"}


def _table(columns: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return lines


def _item(n: int, item: ItemWithOptions) -> list[str]:
    d = item.draft
    out = [f"### Задание {n} · {TYPE_NAMES[d.item_type]} · {d.competence} · уровень {d.target_level}", "", d.stem, ""]
    if d.artifact:
        out += _table(d.artifact.columns, d.artifact.rows) + [""]
    out += [f"**{d.question}**", ""]
    out += [f"- **{o.id})** {o.text}" for o in item.options]
    out += ["", "<details><summary>Ключ и обратная связь</summary>", "", f"**Ключ: {item.key_id}.** {d.key_rationale}", ""]
    out += [f"- {o.id}: {o.feedback}" for o in item.options]
    out += ["", "</details>", ""]
    return out


def render_markdown(result: GenerationResult) -> str:
    bp = result.blueprint
    out = [f"# Персональный тест · педагог {bp.teacher_id}", "", f"Текущий уровень DigCompEdu: **{bp.current_level}**. {result.message}", ""]
    out += ["## Спецификация", "", "| Ячейка | Область | Компетенция | Уровень | Тип | Заданий | Обоснование |", "|---|---|---|---|---|---|---|"]
    out += [
        f"| {c.cell_id} | {c.area}. {c.area_name} | {c.competence} | {c.target_level} | {TYPE_NAMES[c.item_type]} | {c.count} | {c.rationale} |"
        for c in bp.cells
    ]
    out += ["", "## Задания", ""]
    n = 0
    for cell in result.cells:
        # Revised items keep their slot: c1-i1-r1 goes where c1-i1 was.
        for item in sorted(cell.accepted, key=lambda i: i.item_id.split("-r")[0]):
            n += 1
            out += _item(n, item)
    gaps = [r for c in result.cells for r in c.unfilled_reasons]
    if gaps:
        out += ["## Незакрытые места", ""] + [f"- {g}" for g in gaps] + [""]
    return "\n".join(out)
