"""
Shared fixtures. Nothing here touches the network or a vector store: these tests
exist so the numbers in the README can be trusted, and a test that costs money
is a test nobody runs.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# rag.py builds its ChatOpenAI client at import time, so importing anything that
# reaches it needs a key present even though these tests never make a call.
# A real .env still wins: rag.py calls load_dotenv(override=True).
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")


@dataclass
class FakeDoc:
    """Stands in for a langchain Document. Only page_content is ever scored."""

    page_content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class FakeTest:
    """Stands in for a TestQuestion. Only keywords matter to the ceiling."""

    keywords: list[str]
    question: str = "q"
    reference_answer: str = "a"
    category: str = "direct_fact"


@pytest.fixture
def docs():
    """Four chunks; the word 'kurta' appears at rank 2 and again at rank 4."""
    return [
        FakeDoc("a plain cotton shirt"),
        FakeDoc("an indigo kurta for men"),
        FakeDoc("wide leg linen trousers"),
        FakeDoc("a second kurta, embroidered"),
    ]
