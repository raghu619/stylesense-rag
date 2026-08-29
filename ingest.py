"""
Day 2: documents -> chunks -> vectors -> Chroma.

Run with defaults:      python ingest.py
Or try other settings:  python ingest.py 500 100
"""

import sys
import glob
import os
from collections import Counter

import tiktoken
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv(override=True)

EMBEDDING_MODEL = "text-embedding-3-small"

CHUNK_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
CHUNK_OVERLAP = int(sys.argv[2]) if len(sys.argv) > 2 else 200
DB_NAME = f"vector_db/c{CHUNK_SIZE}_o{CHUNK_OVERLAP}"


def load_documents():
    """Each folder name becomes doc_type metadata, exactly like Insurellm."""
    documents = []
    for folder in glob.glob("knowledge-base/*"):
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder, glob="**/*.md", loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        for doc in loader.load():
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents


documents = load_documents()
text = "\n\n".join(d.page_content for d in documents)
tokens = len(tiktoken.get_encoding("cl100k_base").encode(text))

print(f"Documents      {len(documents)}")
print(f"Characters     {len(text):,}")
print(f"Tokens         {tokens:,}")
print(f"Chunk size     {CHUNK_SIZE}, overlap {CHUNK_OVERLAP}")

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(documents)

print(f"\nChunks         {len(chunks)}")
by_type = Counter(c.metadata["doc_type"] for c in chunks)
docs_by_type = Counter(d.metadata["doc_type"] for d in documents)
for doc_type in sorted(by_type):
    print(f"  {doc_type:<10} {docs_by_type[doc_type]:>3} docs -> {by_type[doc_type]:>3} chunks")

sizes = [len(c.page_content) for c in chunks]
print(f"\nChunk length   avg {sum(sizes)//len(sizes)}, min {min(sizes)}, max {max(sizes)}")

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
if os.path.exists(DB_NAME):
    Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_NAME)
collection = store._collection
dimensions = len(collection.get(limit=1, include=["embeddings"])["embeddings"][0])
print(f"\nStored         {collection.count():,} vectors of {dimensions:,} dimensions")
print(f"Location       {DB_NAME}")

print(f"\nFirst chunk:\n{'-' * 60}\n{chunks[0].page_content[:400]}\n{'-' * 60}")
