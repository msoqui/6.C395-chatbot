"""
src/retriever.py — FAISS-based course retriever with attribute/department pre-filtering

Retrieval strategy per query:
1. Exact-match any course numbers mentioned (guaranteed inclusion)
2. Broad semantic search over the full pre-built FAISS index
3. Post-filter results by attributes (CI-H, HASS-*, REST), no-prereqs constraint
4. Dept prefix (e.g. "Course 6") is a soft preference: fills half the slots first
5. Merge exact matches + filtered semantic results, cap at k
"""

import json
import os
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

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
}
HASS_ATTRS = {"HASS-H", "HASS-A", "HASS-S", "HASS-E", "HASS-AH"}


def _build_reverse_prereqs(courses: list) -> dict:
    """Map each course number to the list of courses that require it."""
    reverse = {}
    for c in courses:
        prereq_str = c.get("prereqs") or ""
        # Extract all course numbers mentioned in the prereq string
        for num in re.findall(r'\b\d+[A-Za-z]*\.[A-Za-z0-9]+(?:\[J\])?\b', prereq_str):
            reverse.setdefault(num.upper(), []).append(c["number"])
    return reverse


def _course_to_text(course: dict, required_by: list = None) -> str:
    """Text blob used for embedding — richer = better retrieval."""
    attrs = ", ".join(course.get("attributes", []))
    text = (
        f"{course['number']} {course['title']}. "
        f"Units: {course.get('units', '')}. "
        f"Prereqs: {course.get('prereqs', 'None')}. "
        f"Attributes: {attrs}. "
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

    def _format_course(self, c: dict) -> str:
        attrs = ", ".join(c.get("attributes", []))
        prereq = (c.get("prereqs") or "None").strip()
        prereq_is_none = prereq.lower() in ("none", "")
        level = "Entry-level (no prerequisites)" if prereq_is_none else "Requires prior coursework — must complete prerequisites first"
        return (
            f"Course {c['number']}: {c['title']}\n"
            f"  LEVEL: {level}\n"
            f"  Prerequisites: {prereq}\n"
            f"  Units: {c.get('units', 'N/A')} | Attributes: {attrs}\n"
            f"  Description: {c.get('description', '')}"
        )

    def _extract_filters(self, query: str):
        """
        Returns (required_attrs, dept_prefix, no_prereqs).
        dept_prefix only set on explicit "course N" mentions, not bare course numbers.
        """
        q = query.lower()

        required_attrs = set()
        if re.search(r'\bhass\b', q):
            required_attrs |= HASS_ATTRS
        for attr, patterns in ATTRIBUTE_PATTERNS.items():
            if any(re.search(p, q) for p in patterns):
                required_attrs.add(attr)

        dept_prefix = None
        dept_match = re.search(r'\bcourse\s+(\d+)\b', q)
        if dept_match:
            dept_prefix = dept_match.group(1) + "."

        no_prereqs = bool(re.search(r'\bno\s+pre\w*\b|\bwithout\s+pre\w*\b', q))

        return required_attrs, dept_prefix, no_prereqs

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
        required_attrs, dept_prefix, no_prereqs = self._extract_filters(query)

        # Step 3: broad semantic search over the full pre-built index
        query_vec = np.array(self.model.encode([query]), dtype="float32")
        faiss.normalize_L2(query_vec)
        search_n = min(k * 15, len(self.courses))  # cast a wide net, then filter
        _, raw_indices = self.index.search(query_vec, search_n)

        # Step 4: post-filter and split into dept-preferred vs rest
        preferred, others = [], []
        for idx in raw_indices[0]:
            if idx in seen or idx >= len(self.courses):
                continue
            c = self.courses[idx]
            attrs = set(c.get("attributes", []))
            prereq = (c.get("prereqs") or "None").strip()

            if required_attrs and not (required_attrs & attrs):
                continue
            if no_prereqs and prereq.lower() not in ("none", ""):
                continue

            if dept_prefix and c["number"].startswith(dept_prefix):
                preferred.append(idx)
            else:
                others.append(idx)

        # Dept is a soft preference: fill up to half slots from preferred dept
        if dept_prefix and preferred:
            half = max(remaining // 2, 1)
            ordered = preferred[:half] + others[:remaining]
        else:
            ordered = (preferred + others)

        for idx in ordered[:remaining]:
            if idx not in seen:
                results.append(self._format_course(self.courses[idx]))
                seen.add(idx)

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
