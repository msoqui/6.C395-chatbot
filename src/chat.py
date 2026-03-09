from huggingface_hub import InferenceClient
from config import BASE_MODEL, MY_MODEL, HF_TOKEN
from src.retriever import Retriever

SYSTEM_TEMPLATE = """\
You are an MIT course advisor helping students navigate the MIT course catalog.

Here is relevant course data retrieved for this conversation:

{course_context}

Guidelines:
- Always cite course numbers (e.g., 6.1010, 24.02) when recommending courses
- Help students filter by major, requirements (CI-H, HASS-H, HASS-S, HASS-A, REST), interests, and schedule
- If asked about a course not in the data above, say so and direct the student to student.mit.edu/catalog
- Never fabricate prerequisite details, units, or schedule information not in the data
- When uncertain, say so clearly rather than guessing
- Be conversational and helpful — like a knowledgeable upperclassman who knows the catalog well
"""


class Chatbot:
    def __init__(self):
        model_id = MY_MODEL if MY_MODEL else BASE_MODEL
        self.client = InferenceClient(model=model_id, token=HF_TOKEN)
        self.retriever = Retriever()

    def format_prompt(self, user_input: str, history: list) -> list:
        """
        Build a messages list for the LLM.

        Retrieves the top-5 most relevant courses for the user's message and
        injects them into the system prompt. Appends full conversation history
        for multi-turn memory.

        Args:
            user_input: the current user message
            history: list of (user_msg, assistant_msg) tuples from prior turns

        Returns:
            list of {"role": ..., "content": ...} dicts
        """
        courses = self.retriever.retrieve(user_input, k=5)
        course_context = "\n\n".join(courses) if courses else "No specific courses retrieved."

        messages = [
            {"role": "system", "content": SYSTEM_TEMPLATE.format(course_context=course_context)}
        ]

        for user_msg, assistant_msg in history:
            messages.append({"role": "user",      "content": user_msg})
            messages.append({"role": "assistant",  "content": assistant_msg})

        messages.append({"role": "user", "content": user_input})
        return messages

    def get_response(self, user_input: str, history: list) -> str:
        """
        Generate a response from the LLM given the current message and history.

        Args:
            user_input: the current user message
            history: list of (user_msg, assistant_msg) tuples from prior turns

        Returns:
            the assistant's response as a string
        """
        messages = self.format_prompt(user_input, history)
        response = self.client.chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.6,
        )
        return response.choices[0].message.content.strip()
