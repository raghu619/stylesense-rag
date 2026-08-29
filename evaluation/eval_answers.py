"""
Day 4, part two: LLM as a judge.

Retrieval metrics cannot tell you whether the final answer was any good. A system
can retrieve all four kurtas and still name only two. So a second model reads the
generated answer next to the reference answer and scores three things:

  accuracy      is it factually right
  completeness  did it cover everything the reference covers
  relevance     did it answer the question asked, without padding

The judge is a model, so it is slower, costs money, and is not perfectly stable
between runs. That is the trade against the retrieval metrics, which are instant,
free and deterministic. Use retrieval metrics to iterate, use the judge to confirm.

Run:  python evaluation/eval_answers.py
"""

import sys
from collections import defaultdict
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.test import TestQuestion, load_tests  # noqa: E402
from rag import DEFAULT_DB, DEFAULT_K, answer_question  # noqa: E402

JUDGE_MODEL = "gpt-4.1-mini"
client = OpenAI()


class AnswerEval(BaseModel):
    feedback: str = Field(description="One or two sentences on what was right or missing")
    accuracy: int = Field(description="1 to 5. Any factual error scores 1.")
    completeness: int = Field(
        description="1 to 5. Only score 5 if everything in the reference answer is covered."
    )
    relevance: int = Field(
        description="1 to 5. Only score 5 if it answers the question and adds nothing extra."
    )


def judge(test: TestQuestion, generated: str) -> AnswerEval:
    prompt = f"""Question:
{test.question}

Generated answer:
{generated}

Reference answer:
{test.reference_answer}

Score the generated answer against the reference on accuracy, completeness and
relevance, from 1 (very poor) to 5 (ideal). Be strict. If the generated answer
lists fewer items than the reference, completeness must be low."""

    return client.chat.completions.parse(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content":
             "You are a strict evaluator of question answering systems. Only perfect "
             "answers score 5."},
            {"role": "user", "content": prompt},
        ],
        response_format=AnswerEval,
    ).choices[0].message.parsed


def main():
    db = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB
    k = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_K

    tests = load_tests()
    print(f"{len(tests)} questions, db={db}, k={k}\n")

    by_category = defaultdict(list)
    worst = []

    for index, test in enumerate(tests, start=1):
        generated, _docs = answer_question(test.question, db=db, k=k)
        result = judge(test, generated)
        by_category[test.category].append(result)
        worst.append((result.accuracy + result.completeness + result.relevance, test, result))
        print(f"  {index:>2}/{len(tests)}  acc {result.accuracy}  comp {result.completeness}  "
              f"rel {result.relevance}   {test.question[:46]}")

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    print(f"\n{'category':<14}{'n':>3}{'accuracy':>10}{'complete':>10}{'relevance':>11}")
    print("-" * 48)
    for category in sorted(by_category):
        results = by_category[category]
        print(f"{category:<14}{len(results):>3}"
              f"{mean([r.accuracy for r in results]):>10.2f}"
              f"{mean([r.completeness for r in results]):>10.2f}"
              f"{mean([r.relevance for r in results]):>11.2f}")

    everything = [r for results in by_category.values() for r in results]
    print("-" * 48)
    print(f"{'OVERALL':<14}{len(everything):>3}"
          f"{mean([r.accuracy for r in everything]):>10.2f}"
          f"{mean([r.completeness for r in everything]):>10.2f}"
          f"{mean([r.relevance for r in everything]):>11.2f}")

    print("\nThree worst answers, with the judge's reasoning:")
    for total, test, result in sorted(worst, key=lambda row: row[0])[:3]:
        print(f"\n  [{total}/15] {test.question}")
        print(f"          {result.feedback}")


if __name__ == "__main__":
    main()
