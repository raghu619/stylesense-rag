"""
Query rewriting: change the question before searching it.

The problem it exists for is vocabulary mismatch. A shopper writes "something
for a hot day" and the catalogue says "breathable linen, summer weight". Same
meaning, no shared words, and a query embedding sitting between two vocabularies
matches neither well. Rewriting moves the query into the corpus's own language.

It acts at the same stage as the price filter: it changes WHICH candidates come
back. It cannot reorder them afterwards, and it cannot enforce anything.

For this catalogue there is a trap worth naming. Cheap products and expensive
products are described in different words: 'casual', 'cotton', 'relaxed',
'machine wash' cluster on items under Rs 1500, while 'tailored', 'festive',
'dry clean only', 'winter' cluster above Rs 3000. So a rewriter can improve the
score on a budget question by riding that correlation, without understanding
the budget at all. That is worse than failing, because the number moves while
the guarantee does not exist. Watch the violation count, not the coverage.
"""

import time

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

REWRITE_MODEL = "gpt-4.1-mini"

# Two rewrite prompts, because the first run showed the PROMPT mattered more
# than the technique, and a confound you can switch is no longer a confound.
#
#   focus   the course's instruction: one VERY short specific question.
#           Narrows the query.
#   expand  translate the question into the corpus's own vocabulary.
#           Widens it, which is how 'fabric', 'fit' and 'care' got in and pulled
#           three of eight slots to the guide documents instead of products.
#
# 'focus' is the default because it is the one the course actually ships, and a
# comparison against someone else's design has to run their design.

PROMPTS = {
    "focus": """You are in a conversation with a shopper, answering questions about an
Indian fashion store's catalogue.
You are about to look up information in a Knowledge Base to answer their question.

This is the shopper's current question:
{question}

Respond only with a single, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content.
Focus on the question details.
IMPORTANT: Respond ONLY with the knowledgebase query, nothing else.""",

    "expand": """You rewrite a shopper's question into a search query for an
Indian fashion store's product catalogue.

Product pages describe fabric, fit, occasion and care, in the store's own words.
Rewrite the question into the vocabulary those pages would use.

Keep every constraint the shopper stated, including any price, gender or garment
type. Reply with the search query only, no explanation, one line.

The shopper's question:
{question}""",
}

last_call: dict = {}
_llm: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=REWRITE_MODEL, temperature=0)
    return _llm


def rewrite(question: str, style: str = "focus") -> str:
    """The rewritten search query. Falls back to the original if anything fails."""
    started = time.perf_counter()
    response = get_llm().invoke(
        [{"role": "system", "content": PROMPTS[style].format(question=question)}]
    )
    text = (response.content or "").strip().strip('"')
    usage = getattr(response, "usage_metadata", None) or {}
    last_call.update({
        "seconds": time.perf_counter() - started,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "original": question,
        "rewritten": text,
        "style": style,
    })
    return text or question


def merge(primary: list[Document], secondary: list[Document]) -> list[Document]:
    """
    Interleave two result lists, dropping duplicates, primary first at each step.

    The obvious implementation appends the whole second list after the first,
    which is what the course material does. That makes rank correlate with
    WHICH query found a chunk rather than how relevant it is, so anything
    reading the merged order downstream inherits that bias. Interleaving keeps
    the two queries on equal footing.
    """
    merged: list[Document] = []
    seen: set[str] = set()
    for a, b in zip(primary, secondary):
        for doc in (a, b):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged.append(doc)
    for doc in primary[len(secondary):] + secondary[len(primary):]:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            merged.append(doc)
    return merged
