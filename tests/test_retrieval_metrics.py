"""
The retrieval metrics decide every number in the README, so they get tested
against hand-computed values rather than against themselves.
"""

import math

from evaluation.eval_retrieval import ndcg, reciprocal_rank
from tests.conftest import FakeDoc


class TestReciprocalRank:
    def test_hit_at_rank_one_scores_one(self, docs):
        assert reciprocal_rank("shirt", docs) == 1.0

    def test_hit_at_rank_two_scores_one_half(self, docs):
        assert reciprocal_rank("kurta", docs) == 0.5

    def test_hit_at_rank_three_scores_one_third(self, docs):
        assert reciprocal_rank("linen", docs) == 1 / 3

    def test_miss_scores_zero(self, docs):
        assert reciprocal_rank("saree", docs) == 0.0

    def test_matching_is_case_insensitive(self, docs):
        assert reciprocal_rank("KURTA", docs) == 0.5

    def test_uses_first_occurrence_not_last(self, docs):
        """'kurta' is at rank 2 and rank 4. Reciprocal rank means the first."""
        assert reciprocal_rank("kurta", docs) == 0.5

    def test_empty_retrieval_scores_zero(self):
        assert reciprocal_rank("kurta", []) == 0.0


class TestNDCG:
    def test_single_hit_at_rank_one_is_perfect(self, docs):
        """One hit, already at the top, so actual == ideal."""
        assert ndcg("shirt", docs, k=4) == 1.0

    def test_hit_at_rank_two_is_discounted(self, docs):
        """dcg = 1/log2(3); ideal puts that hit at rank 1, so idcg = 1."""
        assert ndcg("linen", docs, k=4) == 1 / math.log2(4)

    def test_miss_scores_zero_without_dividing_by_zero(self, docs):
        """No relevant chunk means idcg == 0. Must return 0.0, not raise."""
        assert ndcg("saree", docs, k=4) == 0.0

    def test_all_chunks_relevant_is_perfect(self):
        every = [FakeDoc("kurta one"), FakeDoc("kurta two"), FakeDoc("kurta three")]
        assert ndcg("kurta", every, k=3) == 1.0

    def test_k_truncates_before_scoring(self, docs):
        """'linen' sits at rank 3, so at k=2 it must not be counted at all."""
        assert ndcg("linen", docs, k=2) == 0.0

    def test_ndcg_never_exceeds_one(self, docs):
        for keyword in ("shirt", "kurta", "linen", "saree"):
            assert 0.0 <= ndcg(keyword, docs, k=4) <= 1.0
