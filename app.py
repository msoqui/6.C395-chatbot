"""
Gradio web interface for the MIT Course Catalog Chatbot.

Run locally:
    python app.py

Then open http://localhost:7860 in your browser.
"""

import gradio as gr
from src.chat import Chatbot


def create_chatbot():
    chatbot = Chatbot()

    def chat(message, history):
        """
        Called by Gradio on every user message.

        Normalizes Gradio's history format (which differs between Gradio 3 and 5)
        into a flat list of (user_msg, assistant_msg) tuples for the Chatbot class.
        """
        pairs = []
        if history:
            if isinstance(history[0], dict):
                # Gradio 5: flat list of {"role": ..., "content": ...} dicts
                for i in range(0, len(history) - 1, 2):
                    u = history[i].get("content", "") or ""
                    a = history[i + 1].get("content", "") if i + 1 < len(history) else ""
                    if not isinstance(u, str): u = str(u)
                    if not isinstance(a, str): a = str(a)
                    pairs.append((u, a))
            else:
                # Gradio 3/4: list of [user, assistant] pairs
                for h in history:
                    u = h[0] if isinstance(h[0], str) else str(h[0] or "")
                    a = h[1] if isinstance(h[1], str) else str(h[1] or "")
                    pairs.append((u, a))

        return chatbot.get_response(message, pairs)

    demo = gr.ChatInterface(
        chat,
        title="MIT Course Advisor",
        description=(
            "Ask me about MIT courses — I can help you find classes that fit your "
            "major, requirements (CI-H, HASS, REST), interests, and schedule. "
            "Powered by the full MIT catalog. "
            "Free tier note: if you see a 503 error, wait a few seconds and try again."
        ),
        examples=[
            "I'm a Course 6 junior and need a CI-H. I'm interested in ethics or policy.",
            "What are some good HASS-S courses with no prerequisites?",
            "I need a REST elective that overlaps with biology or chemistry.",
            "Can you recommend machine learning electives for an undergrad with calculus?",
            "What's the difference between 6.1010 and 6.1000?",
        ],
    )
    return demo


if __name__ == "__main__":
    demo = create_chatbot()
    demo.launch()
