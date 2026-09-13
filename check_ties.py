"""
How close are the results to each other, and which ranks are unstable?

Rebuilding vector_db/c2000_o400 with added price metadata left the retrieved
SET identical but reshuffled two adjacent pairs. Set-based metrics (precision,
recall) survived that. Rank-based metrics (MRR, nDCG) would not have.

The suspected cause is near-ties in an approximate index. Chroma uses HNSW, a
graph-based approximate nearest neighbour structure, so when two vectors sit at
nearly the same distance from the query, which one the traversal reaches first
can differ between builds. This script tests that by printing the actual
distances and the gap between consecutive ranks.

Read it like this:

  a LARGE gap between rank n and n+1  ->  that boundary is stable, the ordering
                                          is a real preference
  a TINY gap                          ->  a coin flip that will land differently
                                          on the next rebuild

This is the noise floor. Any reranking gain in step 4 that is smaller than the
gaps flagged here is indistinguishable from rebuild noise, and reporting it as
an improvement would be reporting the coin flip.

Run:  python check_ties.py
      python check_ties.py vector_db/c2000_o400 8
"""

import sys

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from catalogue import load_catalogue
from evaluation.schema import load_tests
from rag import DEFAULT_DB, DEFAULT_K, EMBEDDING_MODEL

load_dotenv(override=True)

# Gaps below this share of the total spread across k results are called unstable.
TIE_FRACTION = 0.05


def main(db: str, k: int, category: str) -> None:
    store = Chroma(persist_directory=db, embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL))
    catalogue = load_catalogue()
    tests = [t for t in load_tests() if category == "all" or t.category == category]

    print(f"\nDistance gaps between consecutive ranks   db={db}   k={k}")
    print("=" * 88)

    for test in tests:
        hits = store.similarity_search_with_score(test.question, k=k)
        if not hits:
            continue
        distances = [score for _doc, score in hits]
        spread = max(distances) - min(distances) or 1e-12

        print(f"\n{test.question}")
        print(f"   {'rank':>4}  {'distance':>9}  {'gap':>8}  {'gap %':>7}  item")
        previous = None
        for rank, (doc, score) in enumerate(hits, start=1):
            entry = catalogue.get(doc.metadata.get("source", ""))
            name = entry["title"] if entry else doc.metadata.get("source", "?").split("/")[-1]
            if previous is None:
                gap_text, pct_text, flag = "", "", ""
            else:
                gap = score - previous
                share = gap / spread
                gap_text = f"{gap:.5f}"
                pct_text = f"{share * 100:5.1f}%"
                flag = "  <-- near tie, unstable across rebuilds" if share < TIE_FRACTION else ""
            print(f"   {rank:>4}  {score:>9.5f}  {gap_text:>8}  {pct_text:>7}  {name}{flag}")
            previous = score

    print("\n" + "=" * 88)
    print(f"'near tie' = the gap to the previous rank is under {TIE_FRACTION:.0%} of the spread")
    print("across all k results. Those boundaries are coin flips, not preferences.")


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB
    k = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_K
    category = sys.argv[3] if len(sys.argv) > 3 else "constraint"
    main(db, k, category)
