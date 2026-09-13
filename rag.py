"""
The RAG pipeline, in one place.

app.py and the evaluation both import from here. That is deliberate: if the
evaluation builds its own retriever, you are measuring a system you do not ship.
"""

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, convert_to_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from catalogue import parse_price_limit
from rerank import rerank as llm_rerank
from rewrite import merge as merge_results
from rewrite import rewrite as llm_rewrite

load_dotenv(override=True)

MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
# Chosen by measurement, not by copying the tutorial default.
# 1000/200 scored 64.0% coverage, 2000/400 scored 69.6% at the same k.
# See eval_results.md for all 18 configurations.
DEFAULT_DB = "vector_db/c2000_o400"
DEFAULT_K = 8

SYSTEM_PROMPT = """You are the StyleSense assistant, for an Indian online fashion store.
You help shoppers decide what to buy and answer questions about our products and policies.

Use only the context below. If the answer is not in it, say you do not have that information.
When you recommend products, name them and give the price.
Be concise and practical. No hype.

Context:
{context}
"""

_llm: ChatOpenAI | None = None
_retrievers: dict = {}
_stores: dict = {}


def get_llm() -> ChatOpenAI:
    """
    Built on first use, not at import.

    The retrieval evaluation imports this module for fetch_context and never
    generates an answer, so importing must not require an API key. Constructing
    the client at import time made the free, deterministic half of the
    evaluation depend on credentials it never uses.
    """
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=MODEL, temperature=0)
    return _llm


def get_retriever(db: str = DEFAULT_DB, k: int = DEFAULT_K, search_type: str = "similarity"):
    """
    Cached, because building one per question is slow and pointless.

    search_type "similarity" is plain nearest neighbour: it will happily return
    eight near duplicate chunks from the same document.

    search_type "mmr" is maximal marginal relevance: it pulls fetch_k candidates,
    then picks k of them balancing similarity to the query against dissimilarity
    to what it has already chosen. lambda_mult 1.0 is pure relevance, 0.0 is pure
    diversity.
    """
    key = (db, k, search_type)
    if key not in _retrievers:
        store = Chroma(persist_directory=db,
                       embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL))
        kwargs = {"k": k}
        if search_type == "mmr":
            kwargs |= {"fetch_k": max(40, k * 4), "lambda_mult": 0.5}
        # k belongs in search_kwargs. Passing it to invoke() is silently ignored
        # by some langchain versions, which is how people tune a number that
        # never reaches the retriever.
        _retrievers[key] = store.as_retriever(search_type=search_type, search_kwargs=kwargs)
    return _retrievers[key]


def get_store(db: str = DEFAULT_DB) -> Chroma:
    """The raw store, cached. Needed for filtered search, which the retriever
    interface cannot express cleanly because the filter changes per question."""
    if db not in _stores:
        _stores[db] = Chroma(persist_directory=db,
                             embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL))
    return _stores[db]


def _search(query: str, db: str, k: int, search_type: str, limit: int | None):
    """One retrieval. The price predicate is applied here or not at all."""
    if limit is None:
        return get_retriever(db, k, search_type).invoke(query)
    return get_store(db).similarity_search(query, k=k, filter={"price": {"$lte": limit}})


def fetch_context(question: str, db: str = DEFAULT_DB, k: int = DEFAULT_K,
                  search_type: str = "similarity",
                  price_filter: bool = False,
                  rerank_from: int | None = None,
                  rewrite: bool = False,
                  dual: bool = False,
                  rewrite_style: str = "focus") -> list[Document]:
    """
    price_filter=False, rerank_from=None, rewrite=False is the shipped pipeline,
    untouched, so the baseline numbers in the README stay reproducible.

    price_filter=True parses a price ceiling out of the question and hands it to
    Chroma as a metadata predicate. This is a PRE-filter: Chroma restricts the
    candidate set before the nearest neighbour search, so all k slots come back
    holding legal products. Filtering afterwards in Python would delete the
    violators and leave you with fewer than k, raising precision while leaving
    recall exactly where it was.

    A filter can return nothing. Similarity search never can, it always hands
    back its k nearest however poor they are. That difference is the whole
    reason a filter can be honest about having no answer.

    rerank_from=N retrieves N candidates instead of k, then lets a cross-encoder
    cut them to k. Stage one is tuned for recall, stage two for precision. The
    reranker cannot retrieve, so recall is still capped by what N returned.

    rewrite=True searches a rewritten query instead of the shopper's words, and
    rewrite_style picks which instruction the rewriter is given: "focus" narrows
    the query the way the course does, "expand" translates it into the corpus's
    vocabulary. The first run of this showed the prompt mattering more than the
    technique, so it is a parameter rather than a decision buried in a string.

    dual=True searches both the original and the rewrite and interleaves the
    results, which is what the course material does.

    The price ceiling is always parsed from the ORIGINAL question, never the
    rewritten one. A rewriter asked to produce a search query can drop the
    number, and a filter silently losing its predicate would look like the
    filter failing rather than the rewriter interfering.
    """
    limit = parse_price_limit(question) if price_filter else None

    if rerank_from:
        candidates = fetch_context(question, db, rerank_from, search_type,
                                   price_filter, None, rewrite, dual, rewrite_style)
        return llm_rerank(question, candidates, k)

    if not rewrite:
        return _search(question, db, k, search_type, limit)

    rewritten = llm_rewrite(question, rewrite_style)
    if not dual:
        return _search(rewritten, db, k, search_type, limit)

    return merge_results(
        _search(question, db, k, search_type, limit),
        _search(rewritten, db, k, search_type, limit),
    )[:k]


def answer_question(question: str, history: list[dict] | None = None,
                    db: str = DEFAULT_DB, k: int = DEFAULT_K,
                    search_type: str = "similarity",
                    price_filter: bool = False,
                    rerank_from: int | None = None,
                    rewrite: bool = False,
                    dual: bool = False,
                    rewrite_style: str = "focus") -> tuple[str, list[Document]]:
    docs = fetch_context(question, db, k, search_type, price_filter, rerank_from,
                         rewrite, dual, rewrite_style)
    context = "\n\n".join(
        f"Extract from {d.metadata['source']}:\n{d.page_content}" for d in docs
    )
    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]
    messages.extend(convert_to_messages(history or []))
    messages.append(HumanMessage(content=question))
    return get_llm().invoke(messages).content, docs
