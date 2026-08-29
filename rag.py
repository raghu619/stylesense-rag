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

_llm = ChatOpenAI(model=MODEL, temperature=0)
_retrievers: dict = {}


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


def fetch_context(question: str, db: str = DEFAULT_DB, k: int = DEFAULT_K,
                  search_type: str = "similarity") -> list[Document]:
    return get_retriever(db, k, search_type).invoke(question)


def answer_question(question: str, history: list[dict] | None = None,
                    db: str = DEFAULT_DB, k: int = DEFAULT_K,
                    search_type: str = "similarity") -> tuple[str, list[Document]]:
    docs = fetch_context(question, db, k, search_type)
    context = "\n\n".join(
        f"Extract from {d.metadata['source']}:\n{d.page_content}" for d in docs
    )
    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]
    messages.extend(convert_to_messages(history or []))
    messages.append(HumanMessage(content=question))
    return _llm.invoke(messages).content, docs
