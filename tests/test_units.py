import os

import pytest
from conftest import ScriptedLLM

from kim_agents.agents.base import Agent, RunContext
from kim_agents.agents.profile_analyst import next_level, select_zones
from kim_agents.agents.quality_verifier import BlindSolution, QualityVerifier
from kim_agents.agents.values import same_value
from kim_agents.calllog import CallLog
from kim_agents.load_report import format_report
from kim_agents.messages import ItemDraft, ItemWithOptions, Option
from kim_agents.state import StateStore
from kim_agents.tools import Toolbox
from kim_agents.tools.sandbox import run_python


def make_ctx(settings, tmp_path) -> RunContext:
    return RunContext(
        run_id="t", settings=settings, llm=ScriptedLLM(), log=CallLog(tmp_path / "calls.jsonl", "t"),
        state=StateStore(tmp_path / "kim.sqlite"), toolbox=Toolbox(),
    )


def practical_item(key_value="2") -> ItemWithOptions:
    draft = ItemDraft(
        item_id="c1-i1", cell_id="c1", item_type="practical", competence="Q14", target_level="B1",
        stem="s", question="q", correct_answer="Двум", correct_value=key_value, key_rationale="r",
    )
    options = [Option(id=i, text=t, value=v, feedback="f") for i, t, v in
               [("A", "Одному", "1"), ("B", "Двум", key_value), ("C", "Трём", "3"), ("D", "Четырём", "4")]]
    return ItemWithOptions(draft=draft, options=options, key_id="B")


def test_zones_priority_one_gets_the_extra_items():
    zones = select_zones({1: 1.5, 2: 2.7, 3: 2.5, 4: 1.33, 5: 2.7, 6: 2.4}, items_total=7)
    assert zones == [(4, 1, 4), (1, 2, 3)]


def test_next_level_is_capped():
    assert next_level("A2") == "B1" and next_level("C2") == "C2"


def test_agent_with_more_than_five_tools_is_refused(make_settings, tmp_path):
    class Greedy(Agent):
        name = "greedy"
        tool_names = ("load_profile", "get_framework_overview", "get_competence_descriptor",
                      "get_bank_examples", "get_misconceptions", "similarity_search")

    with pytest.raises(ValueError, match="больше 5"):
        Greedy(make_ctx(make_settings(), tmp_path))


async def test_agent_cannot_call_a_tool_it_did_not_declare(make_settings, tmp_path):
    from kim_agents.agents.base import AgentError
    from kim_agents.agents.item_writer import ItemWriter

    writer = ItemWriter(make_ctx(make_settings(), tmp_path))
    with pytest.raises(AgentError, match="не разрешён"):
        await writer.use_tool("run_python", jobs=[])


def test_blind_view_hides_the_key():
    view = practical_item().blind_view()
    flat = str(view)
    assert "key_id" not in view and "feedback" not in flat and "correct" not in flat and "value" not in flat


def test_recomputed_answer_that_disagrees_with_key_rejects_item(make_settings, tmp_path):
    verifier = QualityVerifier(make_ctx(make_settings(), tmp_path))
    item = practical_item()
    sol = BlindSolution(item_id="c1-i1", answer="B", confidence=0.9)
    verdict = verifier._judge(item, sol, {"score": 0.1, "match_id": None}, {"ok": True, "value": "3", "error": None})
    assert verdict.decision == "revise" and verdict.recomputed_match is False
    assert any("Пересчёт по таблице дал 3" in i for i in verdict.issues)


def test_duplicate_of_bank_item_is_rejected(make_settings, tmp_path):
    ctx = make_ctx(make_settings(), tmp_path)
    bank = ctx.toolbox.bank[0]
    sim = ctx.toolbox.similarity_search({"x": f"{bank['stem']} {bank['question']}"})
    verdict = QualityVerifier(ctx)._judge(
        practical_item(), BlindSolution(item_id="c1-i1", answer="B", confidence=0.9), sim["x"], None
    )
    assert sim["x"]["match_id"] == bank["item_id"] and verdict.decision == "revise"


def test_two_defensible_answers_reject_item(make_settings, tmp_path):
    verifier = QualityVerifier(make_ctx(make_settings(), tmp_path))
    sol = BlindSolution(item_id="c1-i1", answer="B", confidence=0.8, defensible_alternatives=["C"])
    verdict = verifier._judge(practical_item(), sol, {"score": 0.1, "match_id": None}, None)
    assert verdict.decision == "revise" and verdict.defensible_alternatives == ["C"]


def test_sandbox_computes_and_has_no_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    artifact = {"columns": ["a", "b"], "rows": [["x", 1], ["y", 3]]}
    assert run_python("answer = sum(r[1] for r in artifact['rows'])", artifact, 5)["value"] == "4"
    leaked = run_python("answer = str(sys.modules.get('os') and sys.modules['os'].environ.get('OPENAI_API_KEY'))", artifact, 5)
    assert "sk-secret" not in str(leaked["value"]) and os.environ["OPENAI_API_KEY"] == "sk-secret"
    for code in ("import os", "from os import system", "import importlib", "from subprocess import run", "__builtins__"):
        assert run_python(f"{code}\nanswer = 1", artifact, 5)["ok"] is False, code
    assert "таймаут" in run_python("while True: pass", artifact, 1)["error"]


def test_values_compare_numbers_and_names():
    assert same_value("7,5", "7.5") and same_value("2", "2.0") and same_value("Виктория", " виктория")
    assert not same_value("2", "3") and not same_value(None, "1")


def test_load_report_flags_an_overloaded_agent():
    text = format_report({"a": {"llm": 5, "tool": 0, "total": 5, "share": 0.5}, "b": {"llm": 5, "tool": 0, "total": 5, "share": 0.5}})
    assert "больше 40%" in text and "НЕ выполнено" in text
