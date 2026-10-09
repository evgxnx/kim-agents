"""Share of LLM and tool calls per agent, from a run's calls.jsonl. Backs the «≤ 40% per agent» requirement."""

import json
from collections import Counter
from pathlib import Path

MAX_SHARE = 0.40


def load_shares(log_path: Path) -> dict[str, dict]:
    llm, tool = Counter(), Counter()
    for line in log_path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["kind"] == "llm":
            llm[rec["agent"]] += 1
        elif rec["kind"] == "tool":
            tool[rec["agent"]] += 1
    total = sum(llm.values()) + sum(tool.values())
    agents = sorted(set(llm) | set(tool))
    return {
        a: {"llm": llm[a], "tool": tool[a], "total": llm[a] + tool[a], "share": (llm[a] + tool[a]) / total if total else 0.0}
        for a in agents
    }


def format_report(shares: dict[str, dict]) -> str:
    total = sum(s["total"] for s in shares.values())
    lines = [
        f"{'Агент':<22}{'LLM':>6}{'Инстр.':>8}{'Всего':>7}{'Доля':>8}",
        "-" * 51,
    ]
    for agent, s in sorted(shares.items(), key=lambda kv: -kv[1]["total"]):
        flag = "  ← больше 40%" if s["share"] > MAX_SHARE else ""
        lines.append(f"{agent:<22}{s['llm']:>6}{s['tool']:>8}{s['total']:>7}{s['share']:>7.0%}{flag}")
    lines += ["-" * 51, f"{'Итого':<22}{'':>6}{'':>8}{total:>7}"]
    worst = max((s["share"] for s in shares.values()), default=0.0)
    verdict = "выполнено" if worst <= MAX_SHARE else "НЕ выполнено"
    lines.append(f"Требование «не более 40% вызовов на агента»: {verdict} (максимум {worst:.0%}).")
    return "\n".join(lines)
