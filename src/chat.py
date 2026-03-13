import re
from huggingface_hub import InferenceClient
from config import BASE_MODEL, MY_MODEL, HF_TOKEN
from src.retriever import Retriever

SYSTEM_TEMPLATE = """\
You are a helpful MIT course advisor. Use only the course data below to answer questions. Do not invent course details.

COURSE DATA:
{course_context}

If a course isn't in the data above, say so and direct the student to student.mit.edu/catalog.
"""


class Chatbot:
    def __init__(self):
        model_id = MY_MODEL if MY_MODEL else BASE_MODEL
        self.client = InferenceClient(model=model_id, token=HF_TOKEN)
        self.retriever = Retriever()

    def format_prompt(self, user_input: str, history: list) -> list:
        # Include recent history in the retrieval query so course numbers
        # mentioned in prior turns stay in context on follow-up messages.
        recent_history = " ".join(u for u, _ in history[-3:])
        retrieval_query = f"{recent_history} {user_input}".strip()

        # Use fewer results when asking about specific courses (less noise),
        # more results for broad recommendation queries.
        mentioned = re.findall(r'\b\d+\.\d+[A-Za-z]?\b', retrieval_query)
        k = max(6, len(mentioned) * 2) if mentioned else 12

        courses = self.retriever.retrieve(retrieval_query, k=k)
        course_context = "\n\n".join(courses) if courses else "No course data retrieved."

        messages = [
            {"role": "system", "content": SYSTEM_TEMPLATE.format(course_context=course_context)}
        ]
        for user_msg, assistant_msg in history:
            messages.append({"role": "user",      "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": user_input})
        return messages

    def get_response(self, user_input: str, history: list) -> str:
        messages = self.format_prompt(user_input, history)
        response = self.client.chat_completion(
            messages=messages,
            max_tokens=1024,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
