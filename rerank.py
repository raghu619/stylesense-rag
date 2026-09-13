"""
Stage two: an LLM reranker.

Chroma runs a BI-ENCODER. The query and the document are encoded separately,
and the document's vector was computed at ingestion, before the query existed.
A single frozen vector has to serve every future question, which is why it can
represent "this is a pair of trousers" and cannot represent "7999 is more than
1500": the comparison needs both numbers in the same computation and they never
are.

An LLM reranker is a CROSS-ENCODER. Query and document go through the model
together, so it can attend from the token 1500 to the token 7999 and compare
them. The price of that is that nothing can be precomputed, which is why this
only ever runs over a candidate list rather than the whole catalogue.

It cannot retrieve. It reorders what stage one handed it, so recall after
reranking can never exceed recall of the candidate set.

Listwise, not pointwise: all candidates go in one call and the model returns an
ordering. One request instead of N, and the model can compare candidates against
each other rather than scoring each in isolation.

Candidate text is NOT truncated. The obvious token saving would cut the Details
block, which is exactly where the Price line lives, and removing the field the
reranker exists to read would be a self-defeating optimisation.
"""

import time

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv(override=True)

RERANK_MODEL = "gpt-4.1-mini"

SYSTEM_PROMPT = """You re-rank search results for an Indian fashion store.

You are given a shopper's question and a numbered list of candidate products.
Order EVERY candidate id from most relevant to least relevant for that question.

Judge relevance against every part of the question, not just the topic:
- a stated price ceiling is a hard requirement, a product above it is irrelevant
- a stated gender is a hard requirement
- a stated garment category is a hard requirement

Return every id exactly once. Do not invent ids. Do not omit ids."""

# Metering for the last call. A module-level global is acceptable here because
# the evaluation is single threaded and reads it immediately after the call.
# It would be wrong in a server, where two requests would overwrite each other.
last_call: dict = {}

_llm: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=RERANK_MODEL, temperature=0)
    return _llm


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="Every candidate id, from most relevant to least relevant, each exactly once"
    )


def _repair(order: list[int], n: int) -> tuple[list[int], int, int]:
    """
    Make the model's answer safe to index with.

    A model asked for a permutation of 1..n can return an id out of range, the
    same id twice, or fewer ids than it was given. Indexing straight into the
    candidate list with an unchecked reply raises IndexError on the first, and
    silently drops documents on the third, which is worse because the run
    completes and the numbers look plausible.

    Invalid and duplicate ids are dropped. Ids the model forgot are appended in
    their original order, so the output is always a full permutation.
    Returns (clean order, count rejected, count the model omitted).
    """
    seen: set[int] = set()
    clean: list[int] = []
    rejected = 0
    for value in order:
        if not isinstance(value, int) or value < 1 or value > n or value in seen:
            rejected += 1
            continue
        seen.add(value)
        clean.append(value)
    missing = [i for i in range(1, n + 1) if i not in seen]
    return clean + missing, rejected, len(missing)


def rerank(question: str, docs: list[Document], top_k: int) -> list[Document]:
    """Reorder docs by relevance to question, return the best top_k. Meters into last_call."""
    if len(docs) <= 1:
        last_call.clear()
        return docs[:top_k]

    candidates = "\n\n".join(
        f"# CANDIDATE {i}\n{d.page_content}" for i, d in enumerate(docs, start=1)
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\n{candidates}"},
    ]

    started = time.perf_counter()
    result = get_llm().with_structured_output(RankOrder, include_raw=True).invoke(messages)
    elapsed = time.perf_counter() - started

    parsed = result.get("parsed")
    order = parsed.order if parsed else list(range(1, len(docs) + 1))
    order, rejected, omitted = _repair(order, len(docs))

    usage = getattr(result.get("raw"), "usage_metadata", None) or {}
    last_call.update(
        {
            "seconds": elapsed,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "candidates": len(docs),
            "rejected_ids": rejected,
            "omitted_ids": omitted,
            "parse_failed": parsed is None,
        }
    )
    return [docs[i - 1] for i in order][:top_k]
