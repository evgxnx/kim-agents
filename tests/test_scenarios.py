"""Test scenarios from the specification, run end to end on the scripted LLM.

Each test name carries the scenario number from the README table.
"""

import sqlite3

from conftest import ScriptedLLM, log_records, profile

from kim_agents.llm import FatalLLMError, TransientLLMError
from kim_agents.load_report import MAX_SHARE, load_shares
from kim_agents.runner import generate


async def run(settings, name, llm=None, **kw):
    return await generate(profile(name), settings=settings, llm=llm or ScriptedLLM(), **kw)


async def test_s01_weak_assessment_gets_practical_items_at_next_level(make_settings):
    settings = make_settings()
    result = await run(settings, "t-017-weak-assessment.json")

    assert result.status == "ok"
    bp = result.blueprint
    assert bp.current_level == "B1"
    assert [(c.area, c.priority) for c in bp.cells] == [(4, 1), (1, 2)]
    assessment = bp.cells[0]
    assert assessment.item_type == "practical" and assessment.target_level == "B1"
    assert sum(len(c.accepted) for c in result.cells) == bp.items_total == 6
    for cell in result.cells:
        for item in cell.accepted:
            assert len(item.options) == 4 and item.key_option().text == item.draft.correct_answer


async def test_s02_high_profile_checks_two_lowest_areas_at_advanced_level(make_settings):
    settings = make_settings()
    result = await run(settings, "t-042-high.json")

    assert result.status == "ok"
    assert len(result.blueprint.cells) == 2
    assert all(c.priority == 2 and c.target_level in ("C1", "C2") for c in result.blueprint.cells)


async def test_s03_many_weak_areas_capped_at_three_with_priority_first(make_settings):
    settings = make_settings()
    result = await run(settings, "t-003-many-weak.json")

    cells = result.blueprint.cells
    assert len(cells) == 3 and sum(c.count for c in cells) == 8
    assert cells[0].priority == 1 and cells[0].count >= cells[-1].count


async def test_s04_load_is_balanced_and_messages_are_stored(make_settings):
    settings = make_settings()
    result = await run(settings, "t-017-weak-assessment.json")

    shares = load_shares(settings.runs_dir / result.run_id / "calls.jsonl")
    assert len(shares) == 5
    assert max(s["share"] for s in shares.values()) <= MAX_SHARE
    db = sqlite3.connect(settings.runs_dir / "kim.sqlite")
    types = {t for (t,) in db.execute("SELECT type FROM messages WHERE run_id = ?", (result.run_id,))}
    assert {"TestBlueprint", "BlueprintCell", "ItemDraftBatch", "ItemBatch", "AlignmentBatch", "VerdictBatch"} <= types
    assert db.execute("SELECT status FROM runs WHERE run_id = ?", (result.run_id,)).fetchone() == ("ok",)


async def test_s05_wrong_key_is_caught_by_blind_solve_and_item_rewritten(make_settings):
    settings = make_settings()
    result = await run(settings, "t-017-weak-assessment.json", inject_wrong_key=True)

    records = log_records(settings, result.run_id)
    injected = next(r for r in records if r["name"] == "fault_injection")
    assert any(r["name"] == "revise_item" for r in records)
    assert result.status == "ok"
    accepted_ids = [i.item_id for c in result.cells for i in c.accepted]
    assert injected["item_id"] not in accepted_ids
    assert f"{injected['item_id']}-r1" in accepted_ids


async def test_s06_endless_rejection_stops_after_revision_limit(make_settings):
    def always_wrong(payload, response):
        for sol, item in zip(response["items"], payload["items"]):
            sol["answer"] = next(o["id"] for o in item["options"] if o["id"] != sol["answer"])
        return response

    settings = make_settings(max_revisions=2)
    llm = ScriptedLLM(transform={"blind_solve": always_wrong})
    result = await run(settings, "t-042-high.json", llm=llm)

    assert result.status == "failed"
    assert all(c.status == "unfilled" and c.unfilled_reasons for c in result.cells)
    # 4 items, each rewritten at most twice
    assert llm.calls["revise_item"] <= 4 * 2


async def test_s07_invalid_profile_fails_before_any_llm_call(make_settings):
    settings = make_settings()
    llm = ScriptedLLM()
    result = await run(settings, "t-bad-invalid.json", llm=llm)

    assert result.status == "failed"
    assert "нужно 22 ответа" in result.message
    assert sum(llm.calls.values()) == 0
    assert not any(r["kind"] == "llm" for r in log_records(settings, result.run_id))


async def test_s08_missing_profile_file_gives_clear_message(make_settings):
    settings = make_settings()
    result = await run(settings, "no-such-teacher.json")

    assert result.status == "failed" and "не найден" in result.message


async def test_s09_transient_llm_errors_are_retried(make_settings):
    settings = make_settings()
    llm = ScriptedLLM(faults={"write_items": [TransientLLMError("timeout"), TransientLLMError("429")]})
    result = await run(settings, "t-017-weak-assessment.json", llm=llm)

    assert result.status == "ok"
    retries = [r for r in log_records(settings, result.run_id) if r["kind"] == "llm" and r["status"] == "retry"]
    assert len(retries) == 2


async def test_s10_llm_down_gives_partial_result_and_message(make_settings):
    settings = make_settings(llm_retries=3)
    # The assessment cell's writer always gets a 503; the other cell goes through.
    def down_for_assessment(payload):
        return TransientLLMError("503") if payload["descriptor"]["area"] == 4 else None

    llm = ScriptedLLM(faults={"write_items": down_for_assessment})
    result = await run(settings, "t-017-weak-assessment.json", llm=llm)

    assert result.status == "partial"
    assert "LLM недоступна после 3 повторов" in result.message
    assert sorted(c.status for c in result.cells) == ["failed", "filled"]


async def test_s11_fatal_llm_error_is_not_retried(make_settings):
    settings = make_settings()
    llm = ScriptedLLM(faults={"choose_cells": [FatalLLMError("invalid api key")]})
    result = await run(settings, "t-017-weak-assessment.json", llm=llm)

    assert result.status == "failed" and "запрос к LLM отклонён" in result.message
    assert llm.calls["choose_cells"] == 1


async def test_s12_invalid_llm_json_is_repaired_once(make_settings):
    settings = make_settings()
    llm = ScriptedLLM(faults={"choose_cells": ['{"cells": "не список"}']})
    result = await run(settings, "t-017-weak-assessment.json", llm=llm)

    assert result.status == "ok"
    statuses = [r["status"] for r in log_records(settings, result.run_id) if r["name"] == "choose_cells"]
    assert statuses == ["invalid", "ok"]


async def test_s13_analyst_choice_outside_the_area_is_rejected(make_settings):
    def wrong_area(payload, response):
        response["cells"][0]["competence"] = "Q1"  # area 1 competence for the area 4 zone
        return response

    settings = make_settings()
    llm = ScriptedLLM(transform={"choose_cells": wrong_area})
    result = await run(settings, "t-017-weak-assessment.json", llm=llm)

    assert result.status == "failed"
    assert "не прошёл проверку дважды" in result.message and "не из области" in result.message


async def test_s14_step_limit_stops_the_run(make_settings):
    settings = make_settings(max_steps=8)
    result = await run(settings, "t-017-weak-assessment.json")

    assert result.status in ("failed", "partial")
    assert "лимит шагов" in result.message
    assert any(r["name"] == "step_limit" for r in log_records(settings, result.run_id))


async def test_s15_run_timeout_stops_the_run(make_settings):
    settings = make_settings(run_timeout_s=0.3)
    result = await run(settings, "t-017-weak-assessment.json", llm=ScriptedLLM(delay=0.2))

    assert result.status in ("failed", "partial")
    assert "превышено время" in result.message


async def test_s16_arithmetic_only_item_is_sent_back_by_alignment_reviewer(make_settings):
    def first_is_arithmetic(payload, response):
        for review in response["items"]:
            if review["item_id"] == "c1-i1":
                review["requires_competence"] = False
                review["solvable_by"] = "подсчёт строк таблицы"
        return response

    settings = make_settings()
    llm = ScriptedLLM(transform={"review_alignment": first_is_arithmetic})
    result = await run(settings, "t-017-weak-assessment.json", llm=llm)

    assert result.status == "ok"
    accepted_ids = [i.item_id for c in result.cells for i in c.accepted]
    assert "c1-i1" not in accepted_ids and "c1-i1-r1" in accepted_ids
    db = sqlite3.connect(settings.runs_dir / "kim.sqlite")
    (payload,) = db.execute(
        "SELECT payload FROM messages WHERE run_id = ? AND type = 'RevisionRequest' AND sender = 'alignment_reviewer'",
        (result.run_id,),
    ).fetchone()
    assert "решается без компетенции" in payload
    # the item rejected on alignment never reached the verifier
    assert llm.calls["blind_solve"] == 3

