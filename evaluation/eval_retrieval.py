"""
Day 4, part one: score retrieval without an LLM.

Three metrics, all computed from whether a keyword appears in a retrieved chunk.

  MRR       for each keyword, 1 / rank of the first chunk containing it, then averaged.
            Rewards getting one right answer to the top.
  nDCG      discounted gain, so a hit at rank 1 is worth more than a hit at rank 8.
  coverage  what percentage of the expected keywords appeared anywhere in the top k.
            Rewards completeness, which is what enumeration questions actually need.

No API calls beyond embedding each question, no judge, so it is fast, cheap and
identical on every run. That is what makes it usable as an ablation.

Run:  python evaluation/eval_retrieval.py
      python evaluation/eval_retrieval.py vector_db/c500_o100 12
"""

import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.test import TestQuestion, load_tests  # noqa: E402
from rag import DEFAULT_DB, DEFAULT_K, fetch_context  # noqa: E402


def reciprocal_rank(keyword: str, docs) -> float:
    needle = keyword.lower()
    for rank, doc in enumerate(docs, start=1):
        if needle in doc.page_content.lower():
            return 1.0 / rank
    return 0.0


def ndcg(keyword: str, docs, k: int) -> float:
    needle = keyword.lower()
    relevances = [1 if needle in d.page_content.lower() else 0 for d in docs[:k]]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))
    ideal = sorted(relevances, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def score(test: TestQuestion, db: str, k: int, search_type: str = "similarity") -> dict:
    docs = fetch_context(test.question, db, k, search_type)
    ranks = [reciprocal_rank(kw, docs) for kw in test.keywords]
    gains = [ndcg(kw, docs, k) for kw in test.keywords]
    found = sum(1 for r in ranks if r > 0)
    return {
        "mrr": sum(ranks) / len(ranks),
        "ndcg": sum(gains) / len(gains),
        "coverage": found / len(test.keywords) * 100,
        "found": found,
        "total": len(test.keywords),
        "retrieved": len(docs),
    }


def main():
    db = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB
    k = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_K

    tests = load_tests()
    print(f"{len(tests)} questions, db={db}, k={k}\n")

    rows = []
    by_category = defaultdict(list)
    for test in tests:
        result = score(test, db, k)
        rows.append((test, result))
        by_category[test.category].append(result)
        print(f"  {result['coverage']:>5.1f}%  {result['found']:>2}/{result['total']:<2} "
              f"mrr {result['mrr']:.3f}   {test.question[:52]}")

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    print(f"\n{'category':<14}{'n':>3}{'MRR':>9}{'nDCG':>9}{'coverage':>11}")
    print("-" * 46)
    for category in sorted(by_category):
        results = by_category[category]
        print(f"{category:<14}{len(results):>3}"
              f"{mean([r['mrr'] for r in results]):>9.3f}"
              f"{mean([r['ndcg'] for r in results]):>9.3f}"
              f"{mean([r['coverage'] for r in results]):>10.1f}%")

    everything = [r for _, r in rows]
    print("-" * 46)
    print(f"{'OVERALL':<14}{len(everything):>3}"
          f"{mean([r['mrr'] for r in everything]):>9.3f}"
          f"{mean([r['ndcg'] for r in everything]):>9.3f}"
          f"{mean([r['coverage'] for r in everything]):>10.1f}%")


if __name__ == "__main__":
    main()
