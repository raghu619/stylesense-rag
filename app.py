"""Day 3: the Gradio front end. All the logic lives in rag.py."""

import gradio as gr

from rag import answer_question


def chat(question: str, history: list[dict]) -> str:
    answer, docs = answer_question(question, history)
    print(f"\nQ: {question}")
    for d in docs:
        print(f"   retrieved: {d.metadata['doc_type']:<9} {d.metadata['source'].split('/')[-1]}")
    return answer


if __name__ == "__main__":
    gr.ChatInterface(
        chat,
        title="StyleSense",
        examples=[
            "What should I wear to a beach wedding?",
            "Show me something under 1500 rupees",
            "I need office clothes for the monsoon",
            "What is your returns policy?",
            "Which kurtas do you have for men?",
        ],
    ).launch(inbrowser=True)
