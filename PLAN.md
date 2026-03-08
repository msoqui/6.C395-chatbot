# MIT Course Catalog Chatbot — Implementation Plan

## Goal
Build a conversational chatbot that helps MIT students navigate the course catalog. Students describe their situation (major, year, requirements needed, interests, schedule constraints) and receive accurate, personalized course recommendations.

---

## Scope

**In scope:**
- Conversational multi-turn chat with memory of the current session
- Course recommendations filtered by: major/department, year, distribution requirements (CI-H, HASS, REST), and interest area
- Honest handling of uncertainty (bot directs to student.mit.edu/catalog when unsure)
- Deployment to HuggingFace Spaces (free tier, CPU)

**Out of scope:**
- Live schedule/availability data (real-time scraping)
- Fine-tuning the base model (see note below)
- User accounts or saved schedules across sessions

**Note on fine-tuning:** Fine-tuning teaches a model *behavior and style*, not facts. For a course catalog chatbot, the problem is factual accuracy — not how the model talks. Fine-tuning course Q&A pairs would bake facts into model weights, which still hallucinate on edge cases and go stale every semester. The effort is not worth it compared to giving the model accurate text to read at inference time.

---

## The Core Decision: Static Context vs. RAG

Both approaches are viable. The choice affects accuracy, coverage, and implementation complexity.

### Option A: Static Context (Prompt Engineering)

Manually curate a fixed set of courses and embed them directly in the system prompt on every request.

**How it works:**
```
[System prompt]
You are an MIT course advisor. Here is the course catalog:
{COURSE_CONTEXT}  ← ~50-80 courses, written once, always included
...
User: {message}
```

**Tradeoffs:**

| | |
|---|---|
| + Simple to implement | - Limited to ~50–80 hand-picked courses |
| + No new dependencies | - You decide what's "important" — easy to miss courses |
| + Predictable behavior | - Prompt gets long fast; approaches token limits |
| + Easy to debug (you can read the whole context) | - Manual update required each semester |
| + Works well if your user base is narrow (e.g., Course 6 students only) | - Model may hallucinate on courses not in context |

**Best for:** A quick working prototype, or if you scope the chatbot narrowly (e.g., only Course 6 / HASS requirements).

---

### Option B: RAG (Retrieval-Augmented Generation)

Scrape the full MIT catalog, embed all courses into a vector store (FAISS), and at query time retrieve only the most relevant courses to inject into the prompt.

**How it works:**
```
User query
    ↓
Embed query (sentence-transformers, runs on Space CPU)
    ↓
FAISS retrieves top-5 most relevant courses
    ↓
Inject only those 5 courses into prompt
    ↓
Send to Llama-3.1-8B via HF Inference API
```

**Important:** The HuggingFace Space CPU only runs the Gradio app — LLM inference goes to HF's Inference API. So the Space CPU just needs to handle FAISS lookups and a small embedding model (~80MB). This is feasible on free-tier CPU.

**Tradeoffs:**

| | |
|---|---|
| + Covers the entire MIT catalog (600+ courses) | - More files and dependencies to set up |
| + Higher factual accuracy — model reads actual catalog text | - Scraper needs to be written and maintained |
| + Shorter prompts (only relevant courses injected) | - Embedding quality determines retrieval quality |
| + Easy to update: re-scrape, re-embed, done | - Harder to debug ("why did it retrieve these 5?") |
| + Scales to any size dataset | - Cold start: index must load at Space startup |

**New dependencies:** `sentence-transformers`, `faiss-cpu`, `beautifulsoup4`, `requests`

**Best for:** Higher accuracy, full catalog coverage, and a stronger evaluation story in the memo.

---

## Recommendation

**Start with Option A to get something working, then upgrade to Option B.**

The core logic (`format_prompt`, `get_response`, `chat`) is identical between the two — the only difference is how `COURSE_CONTEXT` is produced. Build Option A first so you have a testable chatbot quickly, then swap in the retriever.

If time is short, Option A scoped to Course 6 + HASS courses is a complete, defensible submission.
If you want a stronger project, Option B is not that much more work once the scraper is done.

---

## Implementation Steps

### Shared (both options)

**Step 1: Implement `get_response()` in `src/chat.py`**
- Call `self.client.text_generation()` with a formatted prompt
- Parameters: `max_new_tokens=512`, `temperature=0.6`, `stop_sequences=["User:"]`
- Strip trailing whitespace and stop tokens from output

**Step 2: Implement `format_prompt()` in `src/chat.py`**
- Inject course context (static string or RAG results) at the top
- Append full conversation history for multi-turn memory
- Include instructions: recommend by course number, admit uncertainty, don't fabricate schedules

```
[SYSTEM]
You are an MIT course advisor. Here is relevant course catalog data:
{course_context}

- Always cite course numbers (e.g., 6.3900, 24.131)
- Filter by requirements (CI-H, HASS, REST), major, schedule, interests
- If asked about a course not in your data, say so and point to student.mit.edu/catalog
- Never fabricate prerequisite or schedule details

[CONVERSATION HISTORY]
User: ...
Assistant: ...

User: {user_input}
Assistant:
```

**Step 3: Implement `chat()` in `app.py`**
- Rebuild conversation history from Gradio's `history` list and pass to `get_response()`
- Update UI: title, description, example questions for MIT students

**Step 4: Test locally, then deploy**
- `python app.py`, test against `mit_chatbot_conversation_example.txt`
- Push to HuggingFace Space, add `HF_TOKEN` secret

---

### Option A only: Build static context (`src/context.py`)
- Manually curate ~50–80 courses from student.mit.edu/catalog
- Prioritize: Course 6 core/electives, Course 18, popular CI-H/HASS options
- Structure each entry: number, title, units, prereqs, attributes, description
- Output: a `COURSE_CONTEXT` string imported by `chat.py`

---

### Option B only: Build the RAG pipeline

**`src/scraper.py`** — run once offline, commit output to repo
- Scrape student.mit.edu/catalog by department
- Parse each course: number, title, units, prereqs, attributes (CI-H/HASS/REST), description
- Save as `data/courses.json`

**`src/retriever.py`** — loaded at app startup
- Load `data/courses.json`
- Embed all courses with `sentence-transformers/all-MiniLM-L6-v2`
- Build FAISS index (precomputed; save/load from `data/course_index.faiss`)
- Expose `retrieve(query, k=5) → list[str]` method

**`src/chat.py`** changes
- Import `Retriever`, instantiate in `__init__`
- In `format_prompt()`, call `self.retriever.retrieve(user_input)` and inject results

---

## File Summary

| File | Option A | Option B |
|---|---|---|
| `src/context.py` | Create — static course string | Not needed |
| `src/scraper.py` | Not needed | Create — scrapes catalog to JSON |
| `src/retriever.py` | Not needed | Create — FAISS index + retrieve() |
| `data/courses.json` | Not needed | Generated by scraper |
| `data/course_index.faiss` | Not needed | Generated by retriever |
| `src/chat.py` | Implement 2 methods | Implement 2 methods + retriever call |
| `app.py` | Implement chat() | Implement chat() |
| `requirements.txt` | No change | Add 4 dependencies |

---

## Evaluation (for the memo)

| Dimension | How to test |
|---|---|
| Accuracy | Ask about 10 specific courses — verify prereqs/attributes against the actual catalog |
| Constraint reasoning | Give a 3-constraint query (major + requirement + interest) — check all 3 are respected |
| Uncertainty handling | Ask about a course outside the context — does it admit it doesn't know? |
| Conversation continuity | Ask a follow-up referencing a prior answer — does it maintain context? |
| Hallucination rate | Count fabricated details (wrong prereqs, invented room numbers, etc.) |

Option B will score better on accuracy and hallucination rate by design — worth noting in the memo.
