"""
Day 2 part C: reduce 1536 dimensions to 2 and 3, and look at the vector store.

Run:  python visualize.py            (uses vector_db/c1000_o200)
      python visualize.py 500 100    (any store you have built)
"""

import sys

import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from sklearn.manifold import TSNE

load_dotenv(override=True)

size = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
overlap = int(sys.argv[2]) if len(sys.argv) > 2 else 200
DB_NAME = f"vector_db/c{size}_o{overlap}"

store = Chroma(persist_directory=DB_NAME,
               embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))
result = store._collection.get(include=["embeddings", "documents", "metadatas"])

vectors = np.array(result["embeddings"])
documents = result["documents"]
doc_types = [m["doc_type"] for m in result["metadatas"]]
palette = {"products": "#C31E64", "guides": "#0C6F68", "policies": "#B8860B"}
colors = [palette[t] for t in doc_types]

print(f"{len(vectors)} vectors of {vectors.shape[1]} dimensions")

labels = [f"{t}<br>{d[:110].replace(chr(10), ' ')}..." for t, d in zip(doc_types, documents)]

for dims in (2, 3):
    reduced = TSNE(n_components=dims, random_state=42, perplexity=15).fit_transform(vectors)
    marker = dict(size=6, color=colors, opacity=0.85)
    if dims == 2:
        trace = go.Scatter(x=reduced[:, 0], y=reduced[:, 1], mode="markers",
                           marker=marker, text=labels, hoverinfo="text")
    else:
        trace = go.Scatter3d(x=reduced[:, 0], y=reduced[:, 1], z=reduced[:, 2],
                             mode="markers", marker=marker, text=labels, hoverinfo="text")

    fig = go.Figure(data=[trace])
    fig.update_layout(title=f"StyleSense knowledge base, {dims}D t-SNE ({size}/{overlap})",
                      width=900, height=700, margin=dict(r=20, b=20, l=20, t=50))
    out = f"tsne_{dims}d.html"
    fig.write_html(out)
    print(f"wrote {out}")
    fig.show()
