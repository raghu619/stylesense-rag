"""
Day 5, part zero: the recall ceiling.

eval_retrieval.py asks "did the right chunks rank well".
This asks the question that comes before it: "were they returned at all".

Reranking does not retrieve, it reorders. The pipeline is

    question -> embed -> Chroma returns top N -> reranker reorders -> keep top k

so a reranker can only ever score as well as what came back in those N. If the
query "something under 1500 rupees" returns 3 of the 13 qualifying products in
its top 20, a perfect reranker still scores 3/13. The other 10 were never in
the room. Same limit applies to query rewriting, which changes the query but
hands the result to the same nearest neighbour search.

So recall@N is the hard ceiling on every rerank or rewrite configuration, and
it is worth knowing before spending anything on either.

Detection is substring match on the product name, the same rule eval_retrieval
uses. Less elegant than parsing the source path, and deliberate: the 58%
constraint coverage in the README was produced by substring matching, so
anything meant to be compared against it has to be measured the same way.

Run:  python evaluation/eval_recall.py
      python evaluation/eval_recall.py vector_db/c1000_o200
      python evaluation/eval_recall.py vector_db/c2000_o400 all
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.schema import TestQuestion, load_tests  # noqa: E402
from rag import DEFAULT_DB, fetch_context  # noqa: E402

# Deep enough to cover the whole index at any chunk size this repo uses.
# Chroma returns everything it has if you ask for more than it holds, which is
# what makes the last column a correctness check rather than another datapoint.
MAX_DEPTH = 1000

CHECKPOINTS = [4, 8, 16, 32, 64]


def first_rank(keyword: str, docs) -> int | None:
    """1-based rank of the first chunk containing this keyword, or None if absent."""
    needle = keyword.lower()
    for rank, doc in enumerate(docs, start=1):
        if needle in doc.page_content.lower():
            return rank
    return None


def recall_curve(test: TestQuestion, db: str) -> tuple[dict[int, int], dict[str, int | None], int]:
    """
    Retrieve once at full depth and slice, rather than one query per checkpoint.

    Chroma returns a ranked list, so the top 8 of a depth-1000 query is the same
    list as a top-8 query. Five separate calls would pay five times for the same
    ordering. The cheapest experiment is the one that does not redo work it has.

    Returns (recall at each checkpoint, first rank per keyword, chunks retrieved).
    """
    docs = fetch_context(test.question, db, k=MAX_DEPTH)
    ranks = {kw: first_rank(kw, docs) for kw in test.keywords}

    depth = len(docs)
    checkpoints = [n for n in CHECKPOINTS if n < depth] + [depth]
    recall = {
        n: sum(1 for r in ranks.values() if r is not None and r <= n) for n in checkpoints
    }
    return recall, ranks, depth


def report(db: str = DEFAULT_DB, category: str = "constraint") -> None:
    tests = load_tests()
    if category != "all":
        tests = [t for t in tests if t.category == category]

    print(f"\nRecall ceiling   db={db}   category={category}   depth=full index")
    print("=" * 92)

    curves = [(t, *recall_curve(t, db)) for t in tests]
    depth = max(c[3] for c in curves)
    columns = [n for n in CHECKPOINTS if n < depth] + [depth]

    header = f"{'question':<44}{'expect':>7}" + "".join(f"{'@' + str(n):>7}" for n in columns)
    print(header)
    print("-" * 92)

    for test, recall, _ranks, _d in curves:
        row = f"{test.question[:43]:<44}{len(test.keywords):>7}"
        row += "".join(f"{recall.get(n, 0):>7}" for n in columns)
        print(row)

    print("-" * 92)
    print(
        f"The @{depth} column is the whole index. Anything short of 'expect' there is an\n"
        f"unreachable keyword, meaning a broken test set rather than a weak retriever."
    )

    print("\n\nRank of the first chunk containing each expected product")
    print("=" * 92)
    print("A product at rank 30 is a ranking problem, a reranker pulling 32 can reach it.")
    print("A product marked MISS is a recall problem and no amount of reordering finds it.\n")

    for test, _recall, ranks, _d in curves:
        print(f"{test.question}")
        for keyword, rank in sorted(ranks.items(), key=lambda kv: (kv[1] is None, kv[1] or 0)):
            marker = f"rank {rank:>3}" if rank is not None else "MISS    "
            print(f"    {marker}   {keyword}")
        print()


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB
    category = sys.argv[2] if len(sys.argv) > 2 else "constraint"
    report(db, category)
