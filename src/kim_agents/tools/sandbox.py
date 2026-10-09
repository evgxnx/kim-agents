"""Runs verifier-written Python in a separate interpreter: timeout, empty environment, no project secrets.

This is a guard rail, not a security boundary: the code comes from our own verifier model.
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PRELUDE = "import json, math, statistics, sys\nartifact = json.loads(sys.stdin.read())\nanswer = None\n"
EPILOGUE = "\nprint(json.dumps({'answer': answer}, ensure_ascii=False, default=str))\n"
_MODULES = r"(os|sys|subprocess|socket|shutil|pathlib|importlib|ctypes|requests|urllib|http)"
FORBIDDEN = re.compile(
    rf"\b(import\s+{_MODULES}\b|from\s+{_MODULES}\b|__import__|__builtins__|__subclasses__|"
    r"open\s*\(|eval\s*\(|exec\s*\(|compile\s*\()"
)


def run_python(code: str, artifact: dict, timeout_s: float) -> dict:
    """Execute `code`, which reads `artifact` and sets `answer`. Returns {ok, value, error}."""
    if FORBIDDEN.search(code):
        return {"ok": False, "value": None, "error": "код использует запрещённые модули или функции"}
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "solve.py"
        script.write_text(PRELUDE + code + EPILOGUE, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                input=json.dumps(artifact, ensure_ascii=False),
                capture_output=True,
                check=False,
                text=True,
                timeout=timeout_s,
                cwd=tmp,
                env={"PATH": "/usr/bin:/bin"},  # deliberately no OPENAI_API_KEY
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "value": None, "error": f"превышен таймаут {timeout_s} с"}
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-1:] or ["неизвестная ошибка"]
        return {"ok": False, "value": None, "error": tail[0][:300]}
    try:
        value = json.loads(proc.stdout.strip().splitlines()[-1])["answer"]
    except (IndexError, KeyError, json.JSONDecodeError):
        return {"ok": False, "value": None, "error": "код не вывел ответ"}
    if value is None:
        return {"ok": False, "value": None, "error": "переменная answer не задана"}
    return {"ok": True, "value": str(value), "error": None}
