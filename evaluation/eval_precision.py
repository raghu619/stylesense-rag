"""
Day 5, part two: precision, lift over random, and what the wasted slots contain.

eval_recall.py asked "did the right products come back at all".
This asks the other half: "of the k slots we showed the model, how many were
worth their tokens, and what was in the ones that were not".

Precision is the metric reranking moves. A reranker cannot retrieve anything
new, it reorders a fixed candidate set, so scoring a reranker without precision
means scoring it on a metric it does not control.

Three outputs, in increasing order of bluntness:

  precision@k   share of slots holding an expected answer
  lift          precision divided by the base rate, which is what you would
                score by drawing k products at random. Lift 1.0 means the
                retriever contributed nothing.
  norm          (precision - chance) / (best possible - chance). 0.0 is a
                blindfolded draw, 1.0 is a perfect retriever. Lift alone is
                not enough: when every expected answer fits inside k, lift
                collapses to catalogue_size/k and the question cancels out,
                so two unrelated queries both print 6.50 at k=8. That is a
                ceiling, not a score. Same correction Cohen's kappa applies:
                subtract what luck gives you, divide by what perfection would.
                A ratio is only interpretable between its floor and ceiling.
  the listing   what a shopper asking for "under Rs 1500" was actually shown,
                with prices. A rate tells you 25%. The listing tells you rank 1
                was Rs 7999.

Two verdicts per chunk, and they are INDEPENDENT, not one category:

  is_answer   the chunk contains an expected product name (ground truth)
  violates    the chunk's product costs more than the stated ceiling

A men's kurta returned for "women's dresses under 3000" is not a price
violation at Rs 1999, but it is still a wrong answer. Forcing those into one
mutually exclusive verdict is what produced recall of 850% in the first draft
of this file: "priced under 3000" was silently substituted for "is an expected
answer", and there are 17 products under 3000 but only 2 correct ones.

Prices are read from the SOURCE FILE, never from the chunk text. At small chunk
sizes a chunk can contain a product name and not its Price line, because the
splitter cut between them. Metadata survives chunking; text does not.

Run:  python evaluation/eval_precision.py
      python evaluation/eval_precision.py vector_db/c2000_o400 8 constraint --filter
      python evaluation/eval_precision.py vector_db/c2000_o400 8 constraint --rewrite
      python evaluation/eval_precision.py vector_db/c2000_o400 8 constraint --rewrite --expand
      python evaluation/eval_precision.py vector_db/c2000_o400 32
      python evaluation/eval_precision.py vector_db/c2000_o400 8 all
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from catalogue import load_catalogue, parse_price_limit  # noqa: E402
from evaluation.schema import TestQuestion, load_tests  # noqa: E402
from rag import DEFAULT_DB, DEFAULT_K, fetch_context  # noqa: E402
from rerank import last_call  # noqa: E402
from rewrite import last_call as rewrite_call  # noqa: E402

def judge(doc, catalogue: dict, limit: int | None, keywords: list[str]) -> dict:
    """Two independent verdicts for one chunk. Neither implies the other."""
    entry = catalogue.get(doc.metadata.get("source", ""))
    body = doc.page_content.lower()
    return {
        "entry": entry,
        "is_answer": any(kw.lower() in body for kw in keywords),
        "violates": entry is not None and limit is not None and entry["price"] > limit,
    }


def score(test: TestQuestion, catalogue: dict, db: str, k: int, price_filter: bool = False,
          rerank_from: int | None = None, rewrite: bool = False,
          dual: bool = False, rewrite_style: str = "focus") -> dict:
    last_call.clear()
    rewrite_call.clear()
    docs = fetch_context(test.question, db, k=k, price_filter=price_filter,
                         rerank_from=rerank_from, rewrite=rewrite, dual=dual,
                         rewrite_style=rewrite_style)
    # Both stages bill the same query, so the meters add rather than replace.
    meter = {
        key: last_call.get(key, 0) + rewrite_call.get(key, 0)
        for key in ("seconds", "input_tokens", "output_tokens")
    } if (last_call or rewrite_call) else {}
    if meter:
        meter |= {"rejected_ids": last_call.get("rejected_ids", 0),
                  "omitted_ids": last_call.get("omitted_ids", 0),
                  "parse_failed": last_call.get("parse_failed", False),
                  "rewritten": rewrite_call.get("rewritten", "")}
    limit = parse_price_limit(test.question)
    verdicts = [judge(d, catalogue, limit, test.keywords) for d in docs]
    retrieved = len(docs) or 1

    # Recall counts distinct expected keywords, exactly as eval_retrieval does,
    # so it is bounded by 100% by construction and stays comparable to the
    # coverage numbers already published in the README.
    found = sum(
        1 for kw in test.keywords if any(kw.lower() in d.page_content.lower() for d in docs)
    )

    hits = sum(1 for v in verdicts if v["is_answer"])
    base_rate = len(test.keywords) / len(catalogue) * 100
    precision = hits / retrieved * 100

    # Best precision reachable at this k: you cannot fill more slots than there
    # are correct answers, and you cannot use more slots than k.
    best = min(k, len(test.keywords)) / retrieved * 100
    span = best - base_rate

    return {
        "precision": precision,
        "recall": found / len(test.keywords) * 100,
        "base_rate": base_rate,
        "best": best,
        "norm": (precision - base_rate) / span if span > 0 else 0.0,
        "lift": precision / base_rate if base_rate else 0.0,
        "violates": sum(1 for v in verdicts if v["violates"]),
        "non_product": sum(1 for v in verdicts if v["entry"] is None),
        "limit": limit,
        "docs": docs,
        "verdicts": verdicts,
        "meter": meter,
    }


def report(db: str = DEFAULT_DB, k: int = DEFAULT_K, category: str = "constraint",
           price_filter: bool = False, rerank_from: int | None = None,
           rewrite: bool = False, dual: bool = False,
           rewrite_style: str = "focus") -> None:
    catalogue = load_catalogue()
    tests = [t for t in load_tests() if category == "all" or t.category == category]
    results = [(t, score(t, catalogue, db, k, price_filter, rerank_from, rewrite, dual,
                         rewrite_style)) for t in tests]

    mode = "filter" if price_filter else "no filter"
    mode += f" + rerank {rerank_from}->{k}" if rerank_from else ""
    mode += ((" + dual rewrite" if dual else " + rewrite") + f" [{rewrite_style}]") if rewrite else ""
    print(f"\nPrecision, lift and violations   db={db}   k={k}   [{mode}]")
    print("=" * 100)
    print(
        f"{'question':<38}{'prec@k':>8}{'random':>8}{'best':>7}{'norm':>7}{'lift':>7}{'recall@k':>10}{'violate':>9}"
    )
    print("-" * 100)
    for test, s in results:
        print(
            f"{test.question[:37]:<38}{s['precision']:>7.1f}%{s['base_rate']:>7.1f}%"
            f"{s['best']:>6.1f}%{s['norm']:>7.2f}{s['lift']:>7.2f}"
            f"{s['recall']:>9.1f}%{s['violates']:>9}"
        )
    print("-" * 100)
    print("random: precision from drawing k products out of the catalogue blindly (the floor)")
    print("best:   precision a perfect retriever would reach at this k (the ceiling)")
    print("norm:   where you sit between them. 0.00 = chance, 1.00 = perfect")
    print("lift:   precision / random. Capped at catalogue/k, so read norm instead")

    meters = [s["meter"] for _t, s in results if s["meter"]]
    if meters:
        print(f"\nReranker cost   {len(meters)} calls")
        print(f"  seconds/query   {sum(m['seconds'] for m in meters)/len(meters):.2f}")
        print(f"  input tokens    {sum(m['input_tokens'] for m in meters)/len(meters):,.0f} avg")
        print(f"  output tokens   {sum(m['output_tokens'] for m in meters)/len(meters):,.0f} avg")
        bad = sum(m.get("rejected_ids", 0) + m.get("omitted_ids", 0) for m in meters)
        fails = sum(1 for m in meters if m.get("parse_failed"))
        print(f"  malformed ids   {bad}   parse failures {fails}")
        rewrites = [(t.question, s["meter"].get("rewritten")) for t, s in results
                    if s["meter"].get("rewritten")]
        if rewrites:
            print("\nHow the questions were rewritten")
            for original, new in rewrites:
                print(f"  {original}\n    -> {new}")

    print("\n\nWhat the shopper was actually shown")
    print("=" * 100)
    for test, s in results:
        limit = f"limit Rs {s['limit']}" if s["limit"] else "no price ceiling"
        print(f"\n{test.question}   [{limit}]")
        for rank, (doc, v) in enumerate(zip(s["docs"], s["verdicts"]), start=1):
            entry = v["entry"]
            label = (
                f"Rs {entry['price']:<6} {entry['title']}"
                if entry
                else f"{'':<9} {Path(doc.metadata.get('source', '?')).stem}"
            )
            flags = []
            if v["violates"]:
                flags.append("VIOLATES")
            if not v["is_answer"]:
                flags.append("not an answer")
            if entry is None:
                flags = ["guide/policy"]
            suffix = f"   <-- {', '.join(flags)}" if flags else ""
            print(f"   {rank:>3}.  {label}{suffix}")


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB
    k = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_K
    category = sys.argv[3] if len(sys.argv) > 3 else "constraint"
    price_filter = "--filter" in sys.argv
    rerank_from = None
    if "--rerank" in sys.argv:
        rerank_from = int(sys.argv[sys.argv.index("--rerank") + 1])
    dual = "--dual" in sys.argv
    rewrite = dual or "--rewrite" in sys.argv
    style = "expand" if "--expand" in sys.argv else "focus"
    report(db, k, category, price_filter, rerank_from, rewrite, dual, style)
