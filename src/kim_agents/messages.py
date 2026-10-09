"""Typed messages the agents exchange. Every hop is wrapped in an Envelope and validated on receipt."""

from datetime import UTC, datetime
from typing import Literal, Self
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

ItemType = Literal["sjt", "practical"]
Level = Literal["A1", "A2", "B1", "B2", "C1", "C2"]
OptionId = Literal["A", "B", "C", "D"]
LEVELS: tuple[Level, ...] = ("A1", "A2", "B1", "B2", "C1", "C2")


class TeacherProfile(BaseModel):
    teacher_id: str = Field(min_length=1)
    language: Literal["ru", "kk", "en"] = "ru"
    digcompedu_answers: list[int] = Field(description="Q1..Q22, scale 0–4")
    items_total: int = Field(default=6, ge=2, le=12)

    @field_validator("digcompedu_answers")
    @classmethod
    def _check_answers(cls, answers: list[int]) -> list[int]:
        if len(answers) != 22:
            raise ValueError(f"нужно 22 ответа DigCompEdu, получено {len(answers)}")
        bad = [f"Q{i + 1}={a}" for i, a in enumerate(answers) if not 0 <= a <= 4]
        if bad:
            raise ValueError(f"ответы вне шкалы 0–4: {', '.join(bad)}")
        return answers


class BlueprintCell(BaseModel):
    cell_id: str
    area: int
    area_name: str
    competence: str
    target_level: Level
    item_type: ItemType
    count: int = Field(ge=1)
    priority: Literal[1, 2]
    rationale: str


class TestBlueprint(BaseModel):
    __test__ = False  # not a pytest test class

    teacher_id: str
    language: str
    current_level: Level
    items_total: int
    cells: list[BlueprintCell]

    @model_validator(mode="after")
    def _counts_add_up(self) -> Self:
        if sum(c.count for c in self.cells) != self.items_total:
            raise ValueError("сумма заданий по ячейкам не равна items_total")
        return self


class Artifact(BaseModel):
    kind: Literal["table"] = "table"
    columns: list[str] = Field(min_length=2)
    rows: list[list[str | int | float]] = Field(min_length=3)


class ItemDraft(BaseModel):
    """What the Item Writer produces: the case, the question and the key — no distractors."""

    item_id: str
    cell_id: str
    item_type: ItemType
    competence: str
    target_level: Level
    stem: str
    artifact: Artifact | None = None
    question: str
    correct_answer: str
    correct_value: str | None = Field(default=None, description="practical: the machine-checkable answer")
    key_rationale: str
    attempt: int = 1


class ItemDraftBatch(BaseModel):
    cell_id: str
    items: list[ItemDraft]


class Option(BaseModel):
    id: OptionId
    text: str
    value: str | None = None
    feedback: str
    misconception: str | None = None


class ItemWithOptions(BaseModel):
    draft: ItemDraft
    options: list[Option] = Field(min_length=4, max_length=4)
    key_id: OptionId

    @property
    def item_id(self) -> str:
        return self.draft.item_id

    def key_option(self) -> Option:
        return next(o for o in self.options if o.id == self.key_id)

    def blind_view(self) -> dict:
        """What the verifier may see: no key, no rationale, no feedback (feedback gives the key away)."""
        return {
            "item_id": self.draft.item_id,
            "item_type": self.draft.item_type,
            "stem": self.draft.stem,
            "artifact": self.draft.artifact.model_dump() if self.draft.artifact else None,
            "question": self.draft.question,
            "options": [{"id": o.id, "text": o.text} for o in self.options],
        }


class ItemBatch(BaseModel):
    cell_id: str
    items: list[ItemWithOptions]


class AlignmentVerdict(BaseModel):
    item_id: str
    decision: Literal["accept", "revise"]
    competence_match: bool
    level_match: bool
    requires_competence: bool = Field(description="false: the item is solvable by general skills (arithmetic, common sense) alone")
    solvable_by: str | None = None
    issues: list[str] = []


class AlignmentBatch(BaseModel):
    cell_id: str
    verdicts: list[AlignmentVerdict]


class QualityVerdict(BaseModel):
    item_id: str
    decision: Literal["accept", "revise"]
    blind_answer: OptionId
    blind_confidence: float
    key_match: bool
    defensible_alternatives: list[OptionId] = []
    recomputed_value: str | None = None
    recomputed_match: bool | None = None
    max_similarity: float = 0.0
    similar_to: str | None = None
    issues: list[str] = []


class VerdictBatch(BaseModel):
    cell_id: str
    verdicts: list[QualityVerdict]


class RevisionRequest(BaseModel):
    cell_id: str
    item: ItemWithOptions
    issues: list[str]


class CellResult(BaseModel):
    cell_id: str
    status: Literal["filled", "partial", "unfilled", "failed"]
    accepted: list[ItemWithOptions] = []
    unfilled_reasons: list[str] = []


class GenerationResult(BaseModel):
    run_id: str
    status: Literal["ok", "partial", "failed"]
    message: str
    blueprint: TestBlueprint | None = None
    cells: list[CellResult] = []


class Envelope(BaseModel):
    msg_id: str = Field(default_factory=lambda: uuid4().hex)
    run_id: str
    sender: str
    recipient: str
    type: str
    attempt: int = 1
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    payload: dict

    @classmethod
    def wrap(cls, run_id: str, sender: str, recipient: str, body: BaseModel, attempt: int = 1) -> "Envelope":
        return cls(
            run_id=run_id,
            sender=sender,
            recipient=recipient,
            type=type(body).__name__,
            attempt=attempt,
            payload=body.model_dump(mode="json"),
        )

    def open[T: BaseModel](self, model: type[T]) -> T:
        if self.type != model.__name__:
            raise TypeError(f"{self.recipient} ждёт {model.__name__}, получено {self.type}")
        return model.model_validate(self.payload)
