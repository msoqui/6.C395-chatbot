"""
src/retriever.py — FAISS-based course retriever with attribute/department pre-filtering

Retrieval strategy per query:
1. Exact-match any course numbers mentioned (guaranteed inclusion)
2. Retrieve follow-on courses that require any mentioned course number
3. Broad semantic search over the full pre-built FAISS index
4. Post-filter by attributes (CI-H, HASS-*, REST), no-prereqs, special subject constraints
5. Dept prefix is a hard preference: preferred dept fills slots first, others as fallback
6. Rerank final results using a cross-encoder for relevance scoring
"""

import json
import os
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "courses.json")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "course_index.faiss")

ATTRIBUTE_PATTERNS = {
    "CI-H":          [r'\bci-h\b', r'\bci h\b', r'\bcommunication intensive\b', r'\bcomm intensive\b'],
    "CI-M":          [r'\bci-m\b', r'\bci m\b'],
    "HASS-H":        [r'\bhass-h\b', r'\bhass h\b', r'\bhumanities\b'],
    "HASS-A":        [r'\bhass-a\b', r'\bhass a\b', r'\barts\b'],
    "HASS-S":        [r'\bhass-s\b', r'\bhass s\b', r'\bsocial science\b'],
    "HASS-E":        [r'\bhass-e\b', r'\bhass e\b'],
    "HASS-AH":       [r'\bhass-ah\b', r'\bhass ah\b'],
    "REST":          [r'\brest\b', r'\bscience requirement\b'],
    "Institute-Lab": [r'\binstitute.?lab\b', r'\blab requirement\b'],
    "Calculus I (GIR)":  [r'\bcalculus\s+i\s+gir\b', r'\bcal1\b'],
    "Calculus II (GIR)": [r'\bcalculus\s+ii\s+gir\b', r'\bcal2\b'],
    "Physics I (GIR)":   [r'\bphysics\s+i\s+gir\b', r'\bphy1\b'],
    "Physics II (GIR)":  [r'\bphysics\s+ii\s+gir\b', r'\bphy2\b'],
    "Chemistry (GIR)":   [r'\bchemistry\s+gir\b', r'\bchem\s+gir\b'],
    "Biology (GIR)":     [r'\bbiology\s+gir\b', r'\bbio\s+gir\b'],
}
HASS_ATTRS = {"HASS-H", "HASS-A", "HASS-S", "HASS-E", "HASS-AH"}

DEPT_ALIASES = {
    "1":  ["civil engineering", "civil", "env engineering"],
    "2":  ["mechanical engineering", "mechanical", "mech e", "meche"],
    "3":  ["materials science", "materials", "dmse"],
    "4":  ["architecture", "arch"],
    "5":  ["chemistry", "chem"],
    "6":  ["computer science", "cs", "electrical engineering", "ee", "eecs", "course 6"],
    "7":  ["biology", "bio"],
    "8":  ["physics"],
    "9":  ["brain", "cognitive science", "cogsci", "neuroscience", "bcs"],
    "10": ["chemical engineering", "cheme"],
    "11": ["urban planning", "urban studies", "planning"],
    "12": ["earth science", "atmospheric science", "oceanography"],
    "14": ["economics", "econ"],
    "15": ["management", "business", "sloan"],
    "16": ["aerospace", "aero", "astro"],
    "17": ["political science", "poli sci", "polisci"],
    "18": ["mathematics", "math", "maths"],
    "20": ["biological engineering", "be"],
    "21": ["humanities"],
    "22": ["nuclear engineering", "nuclear"],
    "24": ["philosophy", "linguistics"],
    "gir": ["gir", "general institute requirement", "calculus gir", "physics gir", "chemistry gir", "biology gir"],
}

def _build_reverse_prereqs(courses: list) -> dict:
    """Map each course number to the list of courses that require it."""
    reverse = {}
    for c in courses:
        prereq_str = c.get("prereqs") or ""
        # Extract all course numbers mentioned in the prereq string
        for num in re.findall(r'\b\d+[A-Za-z]*\.[A-Za-z0-9]+(?:\[J\])?\b', prereq_str):
            reverse.setdefault(num.upper(), []).append(c["number"])
    return reverse


def _course_to_text(course, required_by=None):
    attrs = ", ".join(course.get("attributes", []))
    schedule = course.get('schedule', '')
    instructors = course.get('instructors', '')
    rating = f"{course['rating']:.1f}/7.0" if course.get('rating') else ''
    hours = f"{course['hours']:.1f} hrs/week" if course.get('hours') else ''
    enrollment = f"Avg enrollment: {course['enrollment']:.0f}" if course.get('enrollment') else ''

    text = (
        f"{course['number']} {course['title']}. "
        f"Units: {course.get('units', '')}. "
        f"Prereqs: {course.get('prereqs', 'None')}. "
        f"Attributes: {attrs}. "
        f"Schedule: {schedule}. "
        f"Instructors: {instructors}. "
        f"Rating: {rating}. Hours: {hours}. Enrollment: {enrollment}. "
        f"{course.get('description', '')}"
    )
    if required_by:
        text += f" Students who complete this course can go on to take: {', '.join(required_by)}."
    return text.strip()


class Retriever:
    def __init__(self, data_path: str = DATA_PATH, index_path: str = INDEX_PATH):
        print("Loading course data...")
        with open(data_path, encoding="utf-8") as f:
            self.courses = json.load(f)

        reverse_prereqs = _build_reverse_prereqs(self.courses)
        self.texts = [
            _course_to_text(c, required_by=reverse_prereqs.get(c["number"].upper()))
            for c in self.courses
        ]
        # Index by normalized number: uppercase, brackets/suffixes stripped.
        # This lets "18.C06" match "18.C06[J]", "6.1200" match "6.1200[J]", etc.
        self.course_by_number = {}
        for i, c in enumerate(self.courses):
            for key in self._number_variants(c["number"]):
                self.course_by_number.setdefault(key, i)
        # Index of courses that require each course number (for follow-on lookups)
        self.required_by = reverse_prereqs

        print(f"Loading embedding model ({MODEL_NAME})...")
        self.model = SentenceTransformer(MODEL_NAME)
        self.reranker = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')

        if os.path.exists(index_path):
            print("Loading cached FAISS index...")
            self.index = faiss.read_index(index_path)
        else:
            print(f"Building FAISS index over {len(self.texts)} courses (one-time)...")
            embeddings = np.array(self.model.encode(self.texts, show_progress_bar=True, batch_size=64), dtype="float32")
            faiss.normalize_L2(embeddings)
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings)
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            faiss.write_index(self.index, index_path)
            print(f"Index saved to {index_path}")

        print(f"Retriever ready ({len(self.courses)} courses indexed).")

    def _number_variants(self, number: str) -> list[str]:
        """Return lookup keys for a course number (handles [J], case, whitespace)."""
        base = number.upper().strip()
        variants = [base]
        stripped = re.sub(r'\[.*?\]', '', base).strip()
        if stripped != base:
            variants.append(stripped)
        return variants

    def _format_course(self, c):
        attrs = ", ".join(c.get("attributes", []))
        prereq = (c.get("prereqs") or "None").strip()
        prereq_is_none = prereq.lower() in ("none", "")
        level = "Entry-level (no prerequisites)" if prereq_is_none else "Requires prior coursework"
        rating = f"{c['rating']:.1f}/7.0" if c.get('rating') else 'N/A'
        hours = f"{c['hours']:.1f} hrs/week" if c.get('hours') else 'N/A'
        enrollment = f"{c['enrollment']:.0f} students" if c.get('enrollment') else 'N/A'
        
        return (
            f"Course {c['number']}: {c['title']}\n"
            f"  LEVEL: {level}\n"
            f"  Schedule: {c.get('schedule', 'See catalog')}\n"
            f"  Instructors: {c.get('instructors', 'Staff')}\n"
            f"  Rating: {rating} | Avg hours: {hours} | Avg enrollment: {enrollment}\n"
            f"  Prerequisites: {prereq}\n"
            f"  Units: {c.get('units', 'N/A')} | Attributes: {attrs}\n"
            f"  Description: {c.get('description', '')}"
        )

    def _extract_filters(self, query: str):
        """
        Returns (required_attrs, dept_prefix, no_prereqs).
        dept_prefix only set on explicit "course N" mentions (or aliases to a dept, e.g. 'math' : 18), not bare course numbers.
        """
        q = query.lower()

        required_attrs = set()
        if re.search(r'\bhass\b(?![-\s](?:h|a|s|e)\b)', q):
            required_attrs |= HASS_ATTRS
        for attr, patterns in ATTRIBUTE_PATTERNS.items():
            if any(re.search(p, q) for p in patterns):
                required_attrs.add(attr)

        dept_prefix = None
        dept_match = re.search(r'\bcourse\s+(\d+)\b', q)
        if dept_match:
            dept_prefix = dept_match.group(1) + "."
        else:
            for dept_num, aliases in DEPT_ALIASES.items():
                if any(alias in q for alias in aliases):
                    dept_prefix = dept_num + "."
                    break

        days_map = {
        "monday": "Mon", "tuesday": "Tue", "wednesday": "Wed",
        "thursday": "Thu", "friday": "Fri",
        "mon": "Mon", "tue": "Tue", "wed": "Wed", "thu": "Thu", "fri": "Fri",
        "mwf": "Mon", "tr": "Tue",  # common MIT schedule patterns
            }
        schedule_day = None
        for day_str, day_code in days_map.items():
            if day_str in q:
                schedule_day = day_code
                break

        no_prereqs = bool(re.search(r'\bno\s+pre\w*\b|\bwithout\s+pre\w*\b', q))

        fall_only = bool(re.search(r'\bfall\b', q))
        spring_only = bool(re.search(r'\bspring\b', q))

        return required_attrs, dept_prefix, no_prereqs, fall_only, spring_only, schedule_day

    def retrieve(self, query: str, k: int = 10) -> list[str]:
        """
        Return the top-k most relevant formatted course strings.

        Uses the pre-built FAISS index for all semantic search — no re-encoding
        of course texts at query time.
        """
        results = []
        seen = set()

        # Step 1: exact course number matches (guaranteed)
        mentioned = re.findall(r'\b(\d+[A-Za-z]*\.[A-Za-z0-9]+)(?:\[J\])?\b', query)
        for num in mentioned:
            idx = self.course_by_number.get(re.sub(r'\[.*?\]', '', num.upper()).strip())
            if idx is not None and idx not in seen:
                results.append(self._format_course(self.courses[idx]))
                seen.add(idx)

        # Step 1b: also retrieve courses that require any mentioned course number
        # (makes "what to take after X" and "what requires X" answerable)
        for num in mentioned:
            for follow_on in self.required_by.get(num.upper(), [])[:5]:
                idx = self.course_by_number.get(follow_on.upper())
                if idx is not None and idx not in seen:
                    results.append(self._format_course(self.courses[idx]))
                    seen.add(idx)

        remaining = k - len(results)
        if remaining <= 0:
            return results

        # Step 2: extract filters
        required_attrs, dept_prefix, no_prereqs, fall_only, spring_only, schedule_day = self._extract_filters(query)

        # Step 3: broad semantic search over the full pre-built index
        query_vec = np.array(self.model.encode([query]), dtype="float32")
        faiss.normalize_L2(query_vec)
        search_n = min(k * 30, len(self.courses))  # cast a wide net, then filter
        _, raw_indices = self.index.search(query_vec, search_n)

        # Step 4: post-filter and split into dept-preferred vs rest
        query_wants_special = "special subject" in query.lower()
        preferred, others = [], []
        for idx in raw_indices[0]:
            if idx in seen or idx >= len(self.courses):
                continue
            c = self.courses[idx]
            # Filter out vague placeholder courses unless explicitly asked for
            if not query_wants_special:
                if "Special Subject" in c.get("title", "") or "ad hoc basis" in c.get("description", ""):
                    continue
            attrs = set(c.get("attributes", []))
            prereq = (c.get("prereqs") or "None").strip()

            if required_attrs and not (required_attrs & attrs):
                continue
            if no_prereqs and prereq.lower() not in ("none", ""):
                continue
            if schedule_day and schedule_day not in c.get('schedule', ''):
                continue
            if fall_only and "fall" not in attrs:
                continue
            if spring_only and "spring" not in attrs:
                continue

            if dept_prefix and c["number"].startswith(dept_prefix):
                preferred.append(idx)
            else:
                others.append(idx)

        ordered = (preferred + others)

        for idx in ordered[:remaining]:
            if idx not in seen:
                results.append(self._format_course(self.courses[idx]))
                seen.add(idx)

        if len(results) > 1:
            scores = self.reranker.predict([(query, r) for r in results])
            results = [r for _, r in sorted(zip(scores, results), reverse=True)]
        return results


if __name__ == "__main__":
    r = Retriever()
    for label, q in [
        ("CI-H ethics", "CI-H courses about ethics"),
        ("Course 6 ML", "Course 6 machine learning"),
        ("HASS-S no prereqs", "HASS-S courses with no prerequisites"),
        ("specific lookup", "What is 6.1010?"),
    ]:
        print(f"\n--- {label} ---")
        for hit in r.retrieve(q, k=5):
            print(hit); print()
