"""
Result 2 in the README rests entirely on the ceiling: coverage rises with k for
the trivial reason that a question expecting 13 keywords cannot score above 8/13
at k=8. If ceiling() is wrong, "share of what was possible" is wrong, and the
conclusion that raising k was a mirage collapses.
"""

from experiment import ceiling
from tests.conftest import FakeTest


def test_k_smaller_than_keywords_caps_the_score():
    """13 keywords, 8 slots: at best 8 of them can be found."""
    assert ceiling([FakeTest(["kw"] * 13)], k=8) == 8 / 13 * 100


def test_k_larger_than_keywords_is_one_hundred_percent():
    """3 keywords and 8 slots is 100%, not 8/3. The min() is the whole point."""
    assert ceiling([FakeTest(["kw"] * 3)], k=8) == 100.0


def test_k_equal_to_keywords_is_one_hundred_percent():
    assert ceiling([FakeTest(["kw"] * 8)], k=8) == 100.0


def test_ceiling_averages_across_questions():
    """One capped question and one uncapped: (8/13 + 1) / 2."""
    tests = [FakeTest(["kw"] * 13), FakeTest(["kw"] * 2)]
    assert ceiling(tests, k=8) == (8 / 13 + 1.0) / 2 * 100


def test_ceiling_never_decreases_as_k_grows():
    tests = [FakeTest(["kw"] * 13), FakeTest(["kw"] * 5), FakeTest(["kw"] * 2)]
    scores = [ceiling(tests, k) for k in (1, 2, 4, 8, 16, 32)]
    assert scores == sorted(scores)


def test_ceiling_is_never_above_one_hundred():
    tests = [FakeTest(["kw"] * 13), FakeTest(["kw"] * 1)]
    assert ceiling(tests, k=1000) == 100.0


def test_real_test_set_ceiling_matches_the_readme():
    """
    The README reports a 90.8% ceiling at k=8 and 100% at k=16 for the real
    16-question set. If the test set changes, these numbers must be updated
    in the README too, and this test is what will force that.
    """
    from evaluation.schema import load_tests

    tests = load_tests()
    assert round(ceiling(tests, k=8), 1) == 90.8
    assert ceiling(tests, k=16) == 100.0
