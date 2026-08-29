"""
Day 4 applied: change one thing at a time, measure, write it down.

Sweeps three axes against the same 16 questions:

  chunk size   how the documents were split at ingestion time
  k            how many chunks retrieval returns
  search type  plain similarity, or MMR which trades some similarity for diversity

Uses the retrieval metrics only, because they are free and deterministic. Run the
judge once at the end, on the winner.

Run:  python experiment.py
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluation.eval_retrieval import score  # noqa: E402
from evaluation.schema import load_tests  # noqa: E402

STORES = ["vector_db/c500_o100", "vector_db/c1000_o200", "vector_db/c2000_o400"]
KS = [4, 8, 16]
SEARCH_TYPES = ["similarity", "mmr"]


def ceiling(tests, k: int) -> float:
    """The best coverage arithmetically possible at this k. Know your own limits."""
    return sum(min(k, len(t.keywords)) / len(t.keywords) for t in tests) / len(tests) * 100


def run(tests, db: str, k: int, search_type: str) -> dict:
    results = [score(t, db, k, search_type) for t in tests]
    by_category = defaultdict(list)
    for test, result in zip(tests, results):
        by_category[test.category].append(result)

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    row = {
        "db": db.split("/")[-1],
        "k": k,
        "search": search_type,
        "mrr": mean([r["mrr"] for r in results]),
        "ndcg": mean([r["ndcg"] for r in results]),
        "coverage": mean([r["coverage"] for r in results]),
    }
    for category, group in by_category.items():
        row[category] = mean([r["coverage"] for r in group])
    return row


def main():
    tests = load_tests()
    missing = [db for db in STORES if not Path(db).exists()]
    if missing:
        print("These vector stores do not exist yet. Build them first:")
        for db in missing:
            size, overlap = db.split("/")[-1][1:].split("_o")
            print(f"  python ingest.py {size} {overlap}")
        raise SystemExit(1)

    rows = []
    total = len(STORES) * len(KS) * len(SEARCH_TYPES)
    for db in STORES:
        for k in KS:
            for search_type in SEARCH_TYPES:
                rows.append(run(tests, db, k, search_type))
                print(f"  {len(rows)}/{total} done", end="\r", flush=True)
    print(" " * 30, end="\r")

    header = (f"{'chunks':<12}{'k':>3}{'search':>12}{'MRR':>8}{'nDCG':>8}"
              f"{'coverage':>10}{'ceiling':>9}{'fact':>7}{'occ':>7}{'con':>7}{'enum':>7}")
    lines = [header, "-" * len(header)]
    for row in sorted(rows, key=lambda r: -r["coverage"]):
        lines.append(
            f"{row['db']:<12}{row['k']:>3}{row['search']:>12}"
            f"{row['mrr']:>8.3f}{row['ndcg']:>8.3f}{row['coverage']:>9.1f}%"
            f"{ceiling(tests, row['k']):>8.1f}%"
            f"{row.get('direct_fact', 0):>6.0f}%{row.get('occasion', 0):>6.0f}%"
            f"{row.get('constraint', 0):>6.0f}%{row.get('enumeration', 0):>6.0f}%"
        )
    table = "\n".join(lines)
    print(table)

    best = max(rows, key=lambda r: r["coverage"])
    baseline = next(r for r in rows
                    if r["db"] == "c1000_o200" and r["k"] == 8 and r["search"] == "similarity")
    print(f"\nbaseline  {baseline['db']} k=8 similarity   coverage {baseline['coverage']:.1f}%")
    print(f"best      {best['db']} k={best['k']} {best['search']}   "
          f"coverage {best['coverage']:.1f}%")
    print(f"delta     {best['coverage'] - baseline['coverage']:+.1f} points")

    Path("eval_results.md").write_text(
        f"# StyleSense retrieval ablation\n\n"
        f"{len(tests)} questions, 16 configurations, `text-embedding-3-small`.\n"
        f"Coverage is the percentage of expected keywords appearing in the retrieved "
        f"chunks. Ceiling is the best coverage arithmetically possible at that k.\n\n"
        f"```\n{table}\n```\n", encoding="utf-8")
    print("\nwrote eval_results.md")


if __name__ == "__main__":
    main()
