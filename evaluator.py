"""
Day 4: the evaluation dashboard.

Two evaluations, side by side, over whichever configuration you select.

  Retrieval   MRR, nDCG, coverage. No LLM, instant, deterministic, free.
              Run this constantly.
  Answers     accuracy, completeness, relevance, scored by a judge model.
              Slow, costs money, drifts between runs. Run this to confirm.

The point of putting them in one window is that you can watch the relationship:
when retrieval coverage moves, answer completeness moves with it.

Run:  python evaluator.py
"""

from collections import defaultdict
from pathlib import Path

import gradio as gr
import pandas as pd

from evaluation.eval_answers import judge
from evaluation.eval_retrieval import score
from evaluation.test import load_tests
from rag import DEFAULT_DB, DEFAULT_K, answer_question

TESTS = load_tests()
STORES = sorted(str(p) for p in Path("vector_db").iterdir() if p.is_dir())

GREEN, AMBER, RED = "#0C6F68", "#B8860B", "#C31E64"


def colour(value: float, good: float, ok: float) -> str:
    return GREEN if value >= good else AMBER if value >= ok else RED


def card(label: str, value: str, tone: str, note: str = "") -> str:
    return f"""
    <div style="margin:8px 0;padding:14px 16px;background:#faf8fb;border-radius:8px;
                border-left:5px solid {tone};">
      <div style="font-size:13px;color:#6B6577">{label}</div>
      <div style="font-size:26px;font-weight:700;color:{tone}">{value}</div>
      <div style="font-size:12px;color:#6B6577">{note}</div>
    </div>"""


def ceiling(k: int) -> float:
    """Best coverage arithmetically possible at this k, given the test set."""
    return sum(min(k, len(t.keywords)) / len(t.keywords) for t in TESTS) / len(TESTS) * 100


def mean(values):
    return sum(values) / len(values) if values else 0.0


def run_retrieval(db, k, search_type, progress=gr.Progress()):
    k = int(k)
    results, rows = [], []
    by_category = defaultdict(list)

    for index, test in enumerate(TESTS):
        result = score(test, db, k, search_type)
        results.append(result)
        by_category[test.category].append(result)
        rows.append({
            "category": test.category,
            "question": test.question,
            "found": f"{result['found']}/{result['total']}",
            "coverage %": round(result["coverage"], 1),
            "MRR": round(result["mrr"], 3),
        })
        progress((index + 1) / len(TESTS), desc=f"question {index + 1}")

    cov = mean([r["coverage"] for r in results])
    top = ceiling(k)
    html = (
        card("Coverage", f"{cov:.1f}%", colour(cov, 75, 60),
             f"ceiling at k={k} is {top:.1f}%, so you captured {cov / top * 100:.0f}% of what is possible")
        + card("MRR", f"{mean([r['mrr'] for r in results]):.3f}",
               colour(mean([r["mrr"] for r in results]), 0.5, 0.3),
               "rewards getting one right chunk to the top")
        + card("nDCG", f"{mean([r['ndcg'] for r in results]):.3f}",
               colour(mean([r["ndcg"] for r in results]), 0.5, 0.3),
               "rewards rank position")
    )

    chart = pd.DataFrame([
        {"category": c, "coverage": round(mean([r["coverage"] for r in g]), 1)}
        for c, g in sorted(by_category.items())
    ])
    return html, chart, pd.DataFrame(rows)


def run_answers(db, k, search_type, progress=gr.Progress()):
    k = int(k)
    results, rows = [], []
    by_category = defaultdict(list)

    for index, test in enumerate(TESTS):
        generated, _ = answer_question(test.question, db=db, k=k, search_type=search_type)
        verdict = judge(test, generated)
        results.append(verdict)
        by_category[test.category].append(verdict)
        rows.append({
            "category": test.category,
            "question": test.question,
            "acc": verdict.accuracy,
            "comp": verdict.completeness,
            "rel": verdict.relevance,
            "feedback": verdict.feedback,
        })
        progress((index + 1) / len(TESTS), desc=f"answering and judging {index + 1}")

    accuracy = mean([r.accuracy for r in results])
    complete = mean([r.completeness for r in results])
    relevance = mean([r.relevance for r in results])
    html = (
        card("Accuracy", f"{accuracy:.2f}/5", colour(accuracy, 4.5, 4.0), "is it true")
        + card("Completeness", f"{complete:.2f}/5", colour(complete, 4.5, 4.0),
               "did it say everything it should have")
        + card("Relevance", f"{relevance:.2f}/5", colour(relevance, 4.5, 4.0),
               "did it answer the question without padding")
    )

    chart = pd.DataFrame([
        {"category": c, "completeness": round(mean([r.completeness for r in g]), 2)}
        for c, g in sorted(by_category.items())
    ])
    return html, chart, pd.DataFrame(rows)


with gr.Blocks(title="StyleSense evaluation", theme=gr.themes.Soft()) as ui:
    gr.Markdown("# StyleSense evaluation dashboard")
    gr.Markdown(
        f"{len(TESTS)} questions across four categories. Pick a configuration and measure it."
    )

    with gr.Row():
        db = gr.Dropdown(STORES, value=DEFAULT_DB, label="vector store (chunk size / overlap)")
        k = gr.Slider(2, 24, value=DEFAULT_K, step=1, label="k")
        search_type = gr.Radio(["similarity", "mmr"], value="similarity", label="search type")

    gr.Markdown("## Retrieval\nNo LLM. Instant, free, identical every run.")
    retrieval_button = gr.Button("Run retrieval evaluation", variant="primary")
    with gr.Row():
        retrieval_metrics = gr.HTML()
        retrieval_chart = gr.BarPlot(x="category", y="coverage", y_lim=[0, 100],
                                     title="Coverage by category", height=320)
    retrieval_table = gr.Dataframe(wrap=True)

    gr.Markdown("## Answers\nA judge model reads each answer against the reference. "
                "Slower, costs money, drifts a little between runs.")
    answer_button = gr.Button("Run answer evaluation", variant="primary")
    with gr.Row():
        answer_metrics = gr.HTML()
        answer_chart = gr.BarPlot(x="category", y="completeness", y_lim=[1, 5],
                                  title="Completeness by category", height=320)
    answer_table = gr.Dataframe(wrap=True)

    retrieval_button.click(run_retrieval, [db, k, search_type],
                           [retrieval_metrics, retrieval_chart, retrieval_table])
    answer_button.click(run_answers, [db, k, search_type],
                        [answer_metrics, answer_chart, answer_table])

if __name__ == "__main__":
    ui.launch(inbrowser=True)
