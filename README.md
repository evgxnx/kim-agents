# KIM Agents

A team of AI agents that builds a **personal test for a teacher**, aimed at the teacher's weak areas in digital competence ([DigCompEdu](https://joint-research-centre.ec.europa.eu/digcompedu_en)).

- **Input:** the teacher's self-assessment (22 statements, each rated 0–4).
- **Output:** a set of test items. Each item has a correct answer with an explanation, feedback for every option, and a link to a competence and a level.

Self-assessment shows how teachers see themselves. The test gives a second, independent measure based on real cases and data. The project is part of a master's thesis on the EmotionalAI platform.

## Quick start

You need Python 3.12+ and [uv](https://docs.astral.sh/uv/) (`brew install uv` on macOS).

```bash
uv sync
cp .env.example .env    # add your OPENAI_API_KEY

# build a test for a teacher profile
uv run kim run --profile fixtures/profiles/t-017-weak-assessment.json

# demo: put a wrong answer key into the first item and watch the verifier catch it
uv run kim run --profile fixtures/profiles/t-017-weak-assessment.json --inject-wrong-key

# no network, no API key: LLM answers are scripted
uv run kim run --profile fixtures/profiles/t-017-weak-assessment.json --offline

# how many calls each agent made in a run
uv run kim report runs/<run_id>/calls.jsonl

# tests
uv run pytest
```

Each run writes to `runs/<run_id>/`:

- `test.md` — the test in readable form
- `result.json` — the full result
- `calls.jsonl` — a log of every LLM and tool call

All runs are also stored in `runs/kim.sqlite`. Example logs are in `examples/`.

## How it works

```
Teacher profile → Profile Analyst → test plan (cells)
   ↓ each cell runs in parallel
Item Writer → Distractor Designer → Alignment Reviewer → Quality Verifier → accepted
       ↑__________________ revision request ___________________|
```

A **cell** is one line of the test plan: one competence, one level, one item type, and how many items to write.

| Agent | What it does | Tools |
|---|---|---|
| **Profile Analyst** | Finds the weakest areas and plans the test | `load_profile`, `get_framework_overview` |
| **Item Writer** | Writes the case, the question and the correct answer | `get_competence_descriptor`, `get_bank_examples` |
| **Distractor Designer** | Adds wrong options based on typical teacher mistakes, plus feedback for each option | `get_misconceptions` |
| **Alignment Reviewer** | Checks that the item tests the right competence at the right level, and can't be solved by common sense alone | `get_competence_descriptor` |
| **Quality Verifier** | Solves the item **without seeing the answer key**, recomputes numbers with Python, looks for duplicates | `similarity_search`, `run_python` |

The **orchestrator** is plain code, not an LLM. It passes messages between agents, runs cells in parallel and enforces limits.

Why several agents instead of one: the agent that writes an item is bad at spotting its own mistakes. A separate verifier, on a different model and without the key, catches them.

### Item types

- **`sjt`** (situational): a school situation with 4 possible actions, one of them best.
- **`practical`**: a small data table (grades, LMS activity) and a question like "who needs help?". The verifier checks the answer by running code.

## Safety limits

- At most 80 steps and 10 minutes per run, 90 seconds per LLM call.
- Up to 3 retries for network errors; one retry if the LLM returns invalid JSON.
- At most 2 revisions per item; the loop stops if a rewrite comes back unchanged.
- At most 5 tools per agent, checked in code.
- If one cell fails, the test is built from the other cells.
- Python code runs in a separate process with a timeout and without the API key.

## Project layout

```
src/kim_agents/
  agents/      the five agents
  tools/       agent tools and the Python sandbox
  data/        DigCompEdu levels, item bank, typical mistakes
  messages.py  message types (Pydantic)
  runner.py    orchestrator
fixtures/      sample teacher profiles
tests/         16 scenarios + unit tests
```

## Limitations

- The Assembler (final test document) is not built yet. The test is written as Markdown.
- Offline mode uses a fixed set of scripted answers. It shows that the pipeline works, not how good real items are.
- For situational items, the "best" answer is an expert judgement. The blind check reduces this risk but does not remove it.
- Duplicate search compares text, not meaning.
- Russian only, DigCompEdu only.
