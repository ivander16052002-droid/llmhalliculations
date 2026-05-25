from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT


@dataclass(frozen=True)
class BenchmarkQuestion:
    id: int
    discipline: str
    question: str
    source_ref: str


def load_benchmark(path: str | Path) -> list[BenchmarkQuestion]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    df = pd.read_csv(path, encoding="utf-8")

    required_columns = {"id", "discipline", "question", "source_ref"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Benchmark file is missing columns: {missing}")

    questions: list[BenchmarkQuestion] = []
    for row in df.to_dict(orient="records"):
        questions.append(
            BenchmarkQuestion(
                id=int(row["id"]),
                discipline=str(row["discipline"]),
                question=str(row["question"]),
                source_ref=str(row["source_ref"]),
            )
        )

    return questions


def get_question_by_id(
    questions: list[BenchmarkQuestion],
    question_id: int,
) -> BenchmarkQuestion:
    for question in questions:
        if question.id == question_id:
            return question

    raise ValueError(f"Question with id={question_id} not found")