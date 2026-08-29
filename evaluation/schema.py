"""Day 4: the test question schema and loader, same shape as the course."""

import json
from pathlib import Path

from pydantic import BaseModel, Field

TEST_FILE = Path(__file__).parent / "tests.jsonl"


class TestQuestion(BaseModel):
    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(description="Strings that must appear in the retrieved chunks")
    reference_answer: str = Field(description="The answer a perfect system would give")
    category: str = Field(description="direct_fact, occasion, constraint or enumeration")


def load_tests() -> list[TestQuestion]:
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        return [TestQuestion(**json.loads(line)) for line in f if line.strip()]
