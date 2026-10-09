"""CLI: `kim run` generates a test for one profile, `kim report` shows the load per agent for a run."""

import argparse
import asyncio
import sys
from dataclasses import replace
from pathlib import Path

from .config import load_settings
from .load_report import format_report, load_shares
from .messages import Envelope


def _describe(env: Envelope) -> str:
    p = env.payload
    if env.type == "TestBlueprint":
        return f"{len(p['cells'])} ячейки, {p['items_total']} заданий"
    if env.type == "BlueprintCell":
        return f"{p['cell_id']}: {p['competence']} · {p['item_type']} · {p['target_level']} · ×{p['count']}"
    if env.type in ("ItemDraftBatch", "ItemBatch"):
        return f"{p['cell_id']}: {len(p['items'])} шт."
    if env.type == "AlignmentBatch":
        ok = sum(v["decision"] == "accept" for v in p["verdicts"])
        return f"{p['cell_id']}: соответствует {ok} из {len(p['verdicts'])}"
    if env.type == "VerdictBatch":
        ok = sum(v["decision"] == "accept" for v in p["verdicts"])
        return f"{p['cell_id']}: принято {ok} из {len(p['verdicts'])}"
    if env.type == "RevisionRequest":
        return f"{p['item']['draft']['item_id']}: {p['issues'][0][:70]}"
    return ""


def _print_message(env: Envelope) -> None:
    retry = f" (попытка {env.attempt})" if env.attempt > 1 else ""
    print(f"  {env.sender:>20} → {env.recipient:<20} {env.type}{retry}  {_describe(env)}", flush=True)


def _run(args) -> int:
    from .runner import generate

    settings = load_settings()
    if args.runs_dir:
        settings = replace(settings, runs_dir=Path(args.runs_dir))
    if args.offline:
        from .offline_llm import OfflineLLM

        llm = OfflineLLM()
        print("Режим offline: ответы LLM заскриптованы, сеть не используется.")
    else:
        if not settings.openai_api_key:
            print("Нет OPENAI_API_KEY: задайте его в .env (см. .env.example) или запустите с --offline.", file=sys.stderr)
            return 2
        from .llm import OpenAIClient

        llm = OpenAIClient(settings.openai_api_key)
        print("Модели: " + ", ".join(f"{a}={m}" for a, m in settings.models.items()))

    result = asyncio.run(
        generate(args.profile, settings=settings, llm=llm, inject_wrong_key=args.inject_wrong_key, on_message=_print_message)
    )
    run_dir = settings.runs_dir / result.run_id
    print(f"\n[{result.status}] {result.message}")
    files = ("test.md, result.json" if result.blueprint else "result.json") + ", calls.jsonl"
    print(f"Прогон {result.run_id}: {run_dir}/ ({files})")
    log = run_dir / "calls.jsonl"
    if log.exists() and result.status != "failed":  # a failed run has no typical load to report
        print("\n" + format_report(load_shares(log)))
    return 0 if result.status != "failed" else 1


def _report(args) -> int:
    print(format_report(load_shares(Path(args.log))))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kim", description="Многоагентная генерация персонализированных КИМ")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="собрать тест для профиля педагога")
    run.add_argument("--profile", required=True, help="JSON-профиль педагога (см. fixtures/profiles)")
    run.add_argument("--offline", action="store_true", help="без сети: заскриптованные ответы вместо LLM")
    run.add_argument("--inject-wrong-key", action="store_true", help="подложить неверный ключ в первое задание (демо проверки)")
    run.add_argument("--runs-dir", help="куда писать прогоны (по умолчанию runs/)")
    run.set_defaults(func=_run)

    report = sub.add_parser("report", help="распределение вызовов по агентам")
    report.add_argument("log", help="путь к calls.jsonl прогона")
    report.set_defaults(func=_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
