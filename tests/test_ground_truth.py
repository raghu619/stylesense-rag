"""
The README claims the ground truth is parsed from the corpus, "not hand written,
so a typo cannot silently score zero forever". This is the test that makes that
claim enforceable instead of aspirational: a keyword that appears nowhere in the
corpus is unreachable, and would quietly drag coverage down on every run.
"""

from pathlib import Path

import pytest

from evaluation.schema import load_tests

CORPUS = Path(__file__).parent.parent / "knowledge-base"
CATEGORIES = {"direct_fact", "occasion", "constraint", "enumeration"}


@pytest.fixture(scope="module")
def corpus_text():
    files = sorted(CORPUS.rglob("*.md"))
    assert files, f"no corpus found under {CORPUS}"
    return "\n".join(f.read_text(encoding="utf-8").lower() for f in files)


@pytest.fixture(scope="module")
def tests():
    return load_tests()


def test_test_set_loads(tests):
    assert len(tests) == 16


def test_every_question_has_keywords(tests):
    for t in tests:
        assert t.keywords, f"no keywords for: {t.question}"


def test_every_category_is_known(tests):
    for t in tests:
        assert t.category in CATEGORIES, f"unknown category {t.category!r}"


def test_all_four_categories_are_represented(tests):
    assert {t.category for t in tests} == CATEGORIES


def test_every_keyword_appears_somewhere_in_the_corpus(tests, corpus_text):
    """The core guard: an unreachable keyword scores zero forever, silently."""
    unreachable = [
        (t.question, kw) for t in tests for kw in t.keywords
        if kw.lower() not in corpus_text
    ]
    assert not unreachable, f"keywords not present in the corpus: {unreachable}"


def test_no_duplicate_keywords_within_a_question(tests):
    """A repeated keyword would double count toward that question's coverage."""
    for t in tests:
        lowered = [kw.lower() for kw in t.keywords]
        assert len(lowered) == len(set(lowered)), f"duplicate keyword in: {t.question}"


def test_every_question_has_a_reference_answer(tests):
    for t in tests:
        assert t.reference_answer.strip(), f"no reference answer for: {t.question}"
