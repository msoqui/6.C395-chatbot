import ast
import re
from huggingface_hub import InferenceClient
from config import BASE_MODEL, MY_MODEL, HF_TOKEN
from src.retriever import Retriever

SYSTEM_TEMPLATE = """\

You are a friendly and knowledgeable MIT course advisor. Help students find courses that match their interests, requirements, and schedule.

STRICT RULES:
1. Only describe courses that appear in COURSE DATA below. Never draw on outside knowledge about MIT courses.
2. NEVER invent course details, prerequisites, schedules, times, instructors, ratings, or enrollment figures.
3. If you cannot find relevant courses, say "I don't have course data for that — please check student.mit.edu/catalog."
4. If the question is unrelated to MIT course advising, politely say you can only help with course selection.

STYLE:
- Be conversational and warm. Explain why each course might be a good fit for the student.
- Lead with the course name and a natural description. Weave in data points (rating, workload, schedule, instructor) naturally when they add color.
- When recommending courses generally, favor ones that are well-rated and popular.
- If a course has no rating or schedule data, just skip those details — don't mention their absence.
- If asked about semester availability, only mention courses that are actually offered that semester per the data.

COURSE DATA:
{course_context}

If a course isn't in the data above, direct the student to student.mit.edu/catalog.
"""


class Chatbot:
    def __init__(self):
        model_id = MY_MODEL if MY_MODEL else BASE_MODEL
        self.client = InferenceClient(model=model_id, token=HF_TOKEN)
        self.retriever = Retriever()

    def expand_query(self, user_input: str, history: list) -> str:
        """Use LLM to expand query with relevant terms before retrieval."""
        history_text = ""
        for user_msg, assistant_msg in history[-4:]:
            history_text += f"Student: {user_msg}\nAdvisor: {assistant_msg}\n"
    
        expansion_prompt = f"""
        Given this conversation context:
            {history_text}
            Student: {user_input}

            Expand the latest student message into a self-contained search query.  Here are your instructions:

            You are helping expand this query for an MIT course retrieval system. Here is exactly how retrieval works:
            1. Exact course numbers (e.g. "18.01", "6.1010") guarantee that specific course is returned — only include course numbers you are CERTAIN about
            2. Department names/numbers trigger dept filtering (e.g. "economics course 14", "math course 18")
            3. Exact attribute keywords trigger hard filters: CI-H, CI-M, HASS-H, HASS-A, HASS-S, HASS-E, HASS-AH, REST.
            4. "no prerequisites" triggers a no-prereq filter
            5. All other terms are used for semantic similarity matching against course titles and descriptions

            Rules for expansion:
            - If the query mentions specific course numbers, keep them exactly and focus expansion on related concepts and follow-on courses
            - If the query is about comparing courses (e.g. "difference between X and Y"), keep both course numbers and add descriptive terms about what each course covers — do NOT add unrelated course numbers
            - If the query is vague (e.g. "intro econ"), add the department name/number, likely course numbers, and relevant description terms
            - If the query already has attribute keywords (CI-H, HASS-S etc.), preserve them exactly. Only include these filters if the user asks for them explicitly.
            - If the current message changes topic (e.g. from CS to economics), expand based on the new topic only
            - NEVER invent course numbers you are not certain about
            - NEVER add course numbers from a different department than what was asked about

            Return ONLY the expanded query, no explanation.

            Examples:
            "intro econ" → "introductory economics microeconomics macroeconomics principles course 14 14.01 14.02"
            "difference between 6.1010 and 6.1000" → "6.1010 6.1000 course 6 programming computer science difference comparison"
            "HASS-S no prereqs" → "HASS-S social science no prerequisites introductory"
            "what comes after 18.01" → "18.01 calculus course 18 mathematics follow-on next level 18.02 18.03"
            """
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": expansion_prompt}],
            max_tokens=150,
            temperature=0.1,
        )
        expanded = response.choices[0].message.content.strip()
        return expanded

    DEBUG_BOOL = False
    def format_prompt(self, user_input: str, history: list, debug = DEBUG_BOOL) -> list:
        expanded_query = self.expand_query(user_input, history)
        if debug:
            print(f"DEBUG expanded: {expanded_query}")

        # Use fewer results when asking about specific courses (less noise),
        # more results for broad recommendation queries.
        mentioned = re.findall(r'\b\d+[A-Za-z]*\.[A-Za-z0-9]+(?:\[J\])?\b', expanded_query)
        k = max(20, len(mentioned) * 6) if mentioned else 25

        courses = self.retriever.retrieve(expanded_query, k=k)
        if not courses:
            print("DEBUG: expansion produced no results, falling back to raw query")
            courses = self.retriever.retrieve(user_input, k=k)
            courses = [c for c in courses]
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
        content = response.choices[0].message.content
        if isinstance(content, list):
            content = " ".join(block.get("text", "") for block in content if isinstance(block, dict))
        elif isinstance(content, str) and content.startswith("[{"):
            try:
                blocks = ast.literal_eval(content)
                if isinstance(blocks, list):
                    content = " ".join(b.get("text", "") for b in blocks if isinstance(b, dict))
            except Exception:
                # fallback: strip the wrapper manually
                content = re.sub(r"^\[\{.*?'text':\s*'", "", content, flags=re.DOTALL)
                content = re.sub(r"',\s*'type':.*?\}\]$", "", content, flags=re.DOTALL)
        return content.replace('\\n', '\n').strip()
