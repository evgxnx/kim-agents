"""The tools agents may call. Each agent declares which of them it uses (at most five)."""

import json
import re
from difflib import SequenceMatcher
from importlib import resources
from pathlib import Path

from pydantic import ValidationError

from ..messages import Level, TeacherProfile
from .sandbox import run_python as _run_python


class ProfileError(Exception):
    """The teacher profile is missing or invalid; nothing can be generated from it."""


def _data(name: str):
    return json.loads(resources.files("kim_agents.data").joinpath(name).read_text(encoding="utf-8"))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


class Toolbox:
    def __init__(self, *, sandbox_timeout_s: float = 5.0):
        self.framework = _data("digcompedu_ru.json")
        self.misconceptions = _data("misconceptions_ru.json")
        self.bank = _data("item_bank.json")
        self.sandbox_timeout_s = sandbox_timeout_s
        self._areas = {a["id"]: a for a in self.framework["areas"]}

    def get(self, name: str):
        fn = getattr(self, name, None)
        if name.startswith("_") or not callable(fn):
            raise KeyError(f"нет инструмента {name}")
        return fn

    # --- profile & framework -------------------------------------------------

    def load_profile(self, path: str) -> TeacherProfile:
        file = Path(path)
        if not file.is_file():
            raise ProfileError(f"файл профиля не найден: {path}")
        try:
            return TeacherProfile.model_validate_json(file.read_text(encoding="utf-8"))
        except ValidationError as e:
            reasons = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'профиль'}: {err['msg']}" for err in e.errors())
            raise ProfileError(f"профиль {file.name} невалиден — {reasons}") from e

    def get_framework_overview(self, areas: list[int]) -> list[dict]:
        return [
            {
                "area": a,
                "area_name": self._areas[a]["name"],
                "allowed_item_types": self._areas[a]["item_types"],
                "competences": self._areas[a]["competences"],
            }
            for a in areas
        ]

    def get_competence_descriptor(self, competence: str, level: Level) -> dict:
        for area in self.framework["areas"]:
            for comp in area["competences"]:
                if comp["code"] == competence:
                    lvl = self.framework["levels"][level]
                    return {
                        "competence": competence,
                        "statement": comp["name"],
                        "area": area["id"],
                        "area_name": area["name"],
                        "target_level": level,
                        "level_name": lvl["name"],
                        "level_descriptor": lvl["description"],
                    }
        raise KeyError(f"неизвестная компетенция {competence}")

    # --- item bank ------------------------------------------------------------

    def get_bank_examples(self, item_type: str, limit: int = 1) -> list[dict]:
        return [i for i in self.bank if i["item_type"] == item_type][:limit]

    def get_misconceptions(self, area: int, item_type: str) -> list[str]:
        by_type = self.misconceptions.get(item_type, {})
        return by_type.get(str(area)) or self.misconceptions["sjt"].get(str(area), [])

    def similarity_search(self, texts: dict[str, str], extra_corpus: dict[str, str] | None = None) -> dict[str, dict]:
        """Highest similarity of each text to the bank and to `extra_corpus` (items accepted earlier in the run)."""
        corpus = {i["item_id"]: f"{i['stem']} {i['question']}" for i in self.bank} | (extra_corpus or {})
        corpus = {k: _normalize(v) for k, v in corpus.items()}
        result = {}
        for item_id, text in texts.items():
            norm = _normalize(text)
            best_id, best = None, 0.0
            for other_id, other in corpus.items():
                if other_id == item_id:
                    continue
                score = SequenceMatcher(None, norm, other).ratio()
                if score > best:
                    best_id, best = other_id, score
            result[item_id] = {"score": round(best, 3), "match_id": best_id}
        return result

    # --- code execution -------------------------------------------------------

    def run_python(self, jobs: list[dict]) -> dict[str, dict]:
        """jobs: [{id, code, artifact}] → {id: {ok, value, error}}; one process per job."""
        return {job["id"]: _run_python(job["code"], job["artifact"], self.sandbox_timeout_s) for job in jobs}
