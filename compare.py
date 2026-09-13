"""
Day 5: the comparison, built to be understood by someone who has not read the code.

The first version of this file opened as an empty dashboard with four buttons,
a slider labelled "reranker candidate depth", and a column called "normalised".
Anyone landing on it had to already know the story to read it. This version
tells the story instead:

  1. show the bug, as a shopper would see it, before asking for any input
  2. show the cheap fix beside it
  3. only then offer the expensive fix, behind a button, because it costs money
  4. put the knobs at the bottom for people who want to try to break it

The free configurations render on load. Nothing is behind a click unless
clicking it spends something.

Run:  python compare.py
"""

from pathlib import Path

import gradio as gr
import pandas as pd

from catalogue import load_catalogue
import pricemap
from evaluation.eval_precision import score
from evaluation.schema import load_tests
from rag import DEFAULT_DB, DEFAULT_K

CATALOGUE = load_catalogue()
TESTS = [t for t in load_tests() if t.category == "constraint"]
STORES = sorted(str(p) for p in Path("vector_db").iterdir() if p.is_dir())
OPENING = "Show me something under 1500 rupees."

GREEN, RED, INK, SOFT, LINE = "#0C6F68", "#C31E64", "#2B2733", "#6B6577", "#e6e2ea"

# name, what it does, family, price_filter, rerank, rewrite, dual
#
# The three "clever" rows are the techniques the course teaches. The two "rule"
# rows are a database predicate. Grouping them is the point of the chart: the
# question is not which configuration wins, it is whether any amount of model
# in the loop beats one WHERE clause.
CONFIGS = [
    ("Plain search", "what the shop does today",
     "rule", False, False, False, False),
    ("Reword the question", "an AI rewrites the search before running it",
     "clever", False, False, True, False),
    ("Search it twice", "run the original and the reworded, merge the results",
     "clever", False, False, True, True),
    ("Reword + twice + AI picks", "the full pipeline the course recommends",
     "clever", False, True, True, True),
    ("Let an AI pick", "show a model 32 products, it chooses 8",
     "clever", False, True, False, False),
    ("A price rule", "drop anything over budget before searching",
     "rule", True, False, False, False),
    ("Price rule + AI picks", "the rule decides who is eligible, the model the order",
     "rule", True, True, False, False),
]
FREE = {"Plain search", "A price rule"}


def mean(values):
    return sum(values) / len(values) if values else 0.0


def panel(test, name, blurb, cfg, db, k, depth):
    """One configuration's results for one question, as a shopper would see them."""
    _fam, price_filter, reranks, rewrite, dual = cfg
    result = score(test, CATALOGUE, db, k, price_filter,
                   depth if reranks else None, rewrite, dual)
    rows = ""
    for rank, (doc, verdict) in enumerate(zip(result["docs"], result["verdicts"]), start=1):
        entry = verdict["entry"]
        label = entry["title"] if entry else Path(doc.metadata.get("source", "?")).stem
        price = f"Rs {entry['price']:,}" if entry else "—"
        over = verdict["violates"]
        rows += (
            f"<div style='display:grid;grid-template-columns:16px 74px 1fr;gap:8px;"
            f"padding:5px 12px;font-size:12.5px;align-items:baseline;"
            f"background:{'#fdeef3' if over else 'transparent'}'>"
            f"<span style='color:{SOFT};font-size:11px'>{rank}</span>"
            f"<span style='font-family:monospace;font-weight:600;"
            f"color:{RED if over else INK}'>{price}</span>"
            f"<span style='color:{INK}'>{label}</span></div>"
        )

    bad = result["violates"]
    tone = RED if bad else GREEN
    verdict_line = (
        f"{bad} of these cost more than the shopper asked for" if bad
        else f"every one of these is inside the budget"
    )
    return (
        f"<div style='flex:1;min-width:250px;border:1px solid {LINE};border-radius:10px;"
        f"overflow:hidden;background:#fff'>"
        f"<div style='padding:12px 14px;border-bottom:3px solid {tone}'>"
        f"<div style='font-weight:700;color:{INK};font-size:15px'>{name}</div>"
        f"<div style='font-size:12px;color:{SOFT}'>{blurb}</div></div>"
        f"<div style='padding:6px 0'>{rows}</div>"
        f"<div style='padding:9px 14px;border-top:1px solid {LINE};font-size:12.5px;"
        f"color:{tone};font-weight:600'>{verdict_line}</div></div>"
    )


def show_free(question, db, k):
    """Plain search against the price rule. No model calls, so this runs on load."""
    test = next(t for t in TESTS if t.question == question)
    panels = [
        panel(test, name, blurb, (fam, pf, rr, rw, du), db, int(k), None)
        for name, blurb, fam, pf, rr, rw, du in CONFIGS if name in FREE
    ]
    return f"<div style='display:flex;gap:14px;flex-wrap:wrap'>{''.join(panels)}</div>"


def show_paid(question, db, k, depth):
    """Every configuration that spends a model call."""
    test = next(t for t in TESTS if t.question == question)
    panels = [
        panel(test, name, blurb, (fam, pf, rr, rw, du), db, int(k), int(depth))
        for name, blurb, fam, pf, rr, rw, du in CONFIGS if name not in FREE
    ]
    return f"<div style='display:flex;gap:14px;flex-wrap:wrap'>{''.join(panels)}</div>"


def price_axis(question, db, k):
    """Feed the figure real numbers: what qualifies, and what plain search returned."""
    test = next(t for t in TESTS if t.question == question)
    limit = pricemap.parse_price_limit(question)
    if limit is None:
        return pricemap.render(question, [], [])
    result = score(test, CATALOGUE, db, int(k), False, None)
    returned = [v['entry']['price'] for v in result['verdicts'] if v['entry']]
    qualify = [e['price'] for e in CATALOGUE.values() if e['price'] <= limit]
    return pricemap.render(question, qualify, returned)


def run_all(db, k, depth, progress=gr.Progress()):
    """Every configuration over every budget question. This is the scorecard."""
    k, depth = int(k), int(depth)
    runs, rows = {}, []

    for index, (name, blurb, fam, pf, reranks, rewrite, dual) in enumerate(CONFIGS):
        progress(index / len(CONFIGS), desc=name)
        results = [score(t, CATALOGUE, db, k, pf, depth if reranks else None, rewrite, dual)
                   for t in TESTS]
        meters = [r["meter"] for r in results if r["meter"]]
        runs[name] = {
            "found": mean([r["recall"] for r in results]),
            "bad": sum(r["violates"] for r in results),
            "seconds": mean([m["seconds"] for m in meters]) if meters else 0.0,
            "family": fam,
        }
        rows.append({
            "approach": name,
            "how it works": blurb,
            "right products found": f"{runs[name]['found']:.0f}%",
            "shown but too expensive": runs[name]["bad"],
            "wait per question": f"{runs[name]['seconds']:.1f}s" if meters else "none",
        })

    rows.sort(key=lambda r: float(r["right products found"].rstrip("%")))
    a = runs["Plain search"]["found"]
    b = runs["A price rule"]["found"]
    d = runs["Price rule + AI picks"]["found"]
    best_clever = max(runs[n]["found"] for n in runs if runs[n]["family"] == "clever")
    best_clever_name = max((n for n in runs if runs[n]["family"] == "clever"),
                           key=lambda n: runs[n]["found"])
    wait = runs["Price rule + AI picks"]["seconds"]
    headline = f"""
    <div style="border:1px solid {LINE};border-radius:10px;overflow:hidden;background:#fff">
      <div style="padding:16px 18px;border-bottom:1px solid {LINE}">
        <div style="font-size:13px;color:{SOFT}">Best of the three AI techniques</div>
        <div style="font-size:30px;font-weight:700;color:{INK}">{best_clever:.0f}%</div>
        <div style="font-size:13px;color:{SOFT}">
          &ldquo;{best_clever_name}&rdquo;, up from {a:.0f}%. Every question now waits on a model.</div>
      </div>
      <div style="padding:16px 18px;border-bottom:1px solid {LINE}">
        <div style="font-size:13px;color:{SOFT}">One database rule</div>
        <div style="font-size:30px;font-weight:700;color:{GREEN}">{b:.0f}%</div>
        <div style="font-size:13px;color:{SOFT}">
          {b - best_clever:+.0f} points better than the best AI technique, instant, and free.</div>
      </div>
      <div style="padding:16px 18px">
        <div style="font-size:13px;color:{SOFT}">Adding the AI on top of the rule</div>
        <div style="font-size:30px;font-weight:700;color:{RED if d - b < 1 else GREEN}">
          {d - b:+.0f} points</div>
        <div style="font-size:13px;color:{SOFT}">
          and every shopper waits {wait:.1f}s while a model re-reads results
          that were already correct.</div>
      </div>
    </div>"""

    chart = pd.DataFrame([
        {"approach": name,
         "found %": round(runs[name]["found"], 1),
         "kind": "AI in the loop" if fam == "clever" else "plain search / database rule"}
        for name, _b, fam, _pf, _r, _rw, _du in CONFIGS
    ]).sort_values("found %")
    return headline, chart, pd.DataFrame(rows)


RECORDING_CHECK_CSS = """
/* A pre-flight check for screen recording, in pure CSS so it needs no JS.
   The dashboard stacks its result panels once the viewport drops below about
   850px, which is also roughly where a screen recording stops being landscape.
   So the same breakpoint that reflows the layout can tell you whether the
   window is ready to record. */
#rec-check { border-radius: 8px; padding: 10px 14px; font-size: 13px; font-weight: 600; }
#rec-check::after { display: block; }
@media (min-width: 850px) {
  #rec-check { background: #fdeef3; color: #C31E64; border: 1px solid #C31E64; }
  #rec-check::after { content: "Window too wide to record. Drag the right edge left until this turns green, then drag your recording selection from the top of the screen to the bottom."; }
}
@media (max-width: 849px) {
  #rec-check { background: #e8f3f2; color: #0C6F68; border: 1px solid #0C6F68; }
  #rec-check::after { content: "Ready to record. The panels below are stacked, so a tall selection will capture them at readable size."; }
}
"""


with gr.Blocks(title="StyleSense: why it showed a Rs 7999 answer",
               theme=gr.themes.Soft(), css=RECORDING_CHECK_CSS) as ui:

    gr.HTML("<div></div>", elem_id="rec-check")

    gr.Markdown(
        "# A shopper asked for something under Rs 1,500\n"
        "### and the shop opened with a pair of Rs 7,999 trousers."
    )
    gr.Markdown(
        "Search finds things by **meaning**, not by maths. It turns words into points on a map "
        "and returns whatever sits nearby. *Kurta* has a place on that map. *Under 1500* does "
        "not, so the closest thing to it is Rs 1,499, and the genuinely cheap items sink to the "
        "bottom.\n\n"
        "Below is the same question run two ways, live, against a 52-product catalogue. "
        "**Red rows cost more than the shopper asked for.**"
    )

    question = gr.Dropdown(
        [t.question for t in TESTS], value=OPENING, label="Try a different question"
    )

    gr.Markdown("### The bug, and the boring fix")
    free_panels = gr.HTML()

    gr.Markdown(
        "The fix on the right is not clever. Every price is saved as a number when the "
        "catalogue is loaded, the budget is read out of the question, and anything over it is "
        "removed **before** searching. The search is still just as bad at understanding prices. "
        "It has simply stopped being asked."
    )

    gr.Markdown(
        "---\n"
        "### Why plain search gets it this wrong\n"
        "Search turns words into points on a map and returns whatever sits nearby. It puts "
        "*1500* somewhere, and the nearest things to it are Rs 1,499 and Rs 1,299. But "
        "**under 1500 is a range, not a point**, and a point cannot stand in for a range. "
        "Both rows below are drawn on the same price axis."
    )
    axis = gr.HTML()

    gr.Markdown(
        "---\n"
        "### So does an AI do it better?\n"
        "There are three popular answers. **Reword the question** before searching. "
        "**Search it twice**, once as asked and once reworded, and merge. Or **let a model read** "
        "the top 32 products and pick 8 itself. Each one puts a language model in the path of "
        "every question a shopper asks. Press the button to run all of them."
    )
    paid_button = gr.Button("Run the AI techniques (about a minute, costs a few paise)",
                            variant="primary")
    paid_panels = gr.HTML()

    gr.Markdown(
        "---\n"
        "### Scorecard\n"
        "Every approach, across every budget question in the test set. "
        "*Right products found* is the share of qualifying products that reached the shopper."
    )
    score_button = gr.Button("Score every approach (about a minute)", variant="primary")
    with gr.Row():
        headline = gr.HTML()
        chart = gr.BarPlot(x="found %", y="approach", color="kind", x_lim=[0, 100],
                           title="Right products found", height=380)
    table = gr.Dataframe(wrap=True)

    with gr.Accordion("Settings, if you want to try to break it", open=False):
        gr.Markdown(
            "Change these and re-run. The finding should hold: the price rule wins, and the "
            "AI reader adds nothing on top of it."
        )
        with gr.Row():
            db = gr.Dropdown(STORES, value=DEFAULT_DB,
                             label="how the catalogue was split up")
            k = gr.Slider(2, 24, value=DEFAULT_K, step=1,
                          label="how many results to show")
            depth = gr.Slider(8, 64, value=32, step=8,
                              label="how many products the AI reader gets to see")

    ui.load(show_free, [question, db, k], free_panels)
    ui.load(price_axis, [question, db, k], axis)
    for control in (question, db, k):
        control.change(show_free, [question, db, k], free_panels)
        control.change(price_axis, [question, db, k], axis)
    paid_button.click(show_paid, [question, db, k, depth], paid_panels)
    score_button.click(run_all, [db, k, depth], [headline, chart, table])

if __name__ == "__main__":
    ui.launch(inbrowser=True)
