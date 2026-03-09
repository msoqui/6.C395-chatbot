"""
src/retriever.py — FAISS-based course retriever

Loads data/courses.json, embeds all courses with a small sentence-transformer,
builds a FAISS index, and exposes retrieve(query, k) for use in chat.py.

The index is built once at startup and cached for the lifetime of the process.
On a HuggingFace CPU Space this takes ~10-20 seconds on first load.
"""

import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # ~80MB, fast on CPU
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "courses.json")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "course_index.faiss")


def _course_to_text(course: dict) -> str:
    """
    Convert a course dict to a single searchable text blob.
    This is what gets embedded — richer text = better retrieval.
    """
    attrs = ", ".join(course.get("attributes", []))
    return (
        f"{course['number']} {course['title']}. "
        f"Units: {course.get('units', '')}. "
        f"Prereqs: {course.get('prereqs', 'None')}. "
        f"Attributes: {attrs}. "
        f"{course.get('description', '')}"
    ).strip()


class Retriever:
    def __init__(self, data_path: str = DATA_PATH, index_path: str = INDEX_PATH):
        print("Loading course data...")
        with open(data_path, encoding="utf-8") as f:
            self.courses = json.load(f)

        self.texts = [_course_to_text(c) for c in self.courses]

        print(f"Loading embedding model ({MODEL_NAME})...")
        self.model = SentenceTransformer(MODEL_NAME)

        if os.path.exists(index_path):
            print("Loading cached FAISS index...")
            self.index = faiss.read_index(index_path)
        else:
            print(f"Building FAISS index over {len(self.texts)} courses (one-time cost)...")
            embeddings = self.model.encode(self.texts, show_progress_bar=True, batch_size=64)
            embeddings = np.array(embeddings, dtype="float32")
            faiss.normalize_L2(embeddings)  # cosine similarity via inner product

            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings)

            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            faiss.write_index(self.index, index_path)
            print(f"Index saved to {index_path}")

        print(f"Retriever ready ({len(self.courses)} courses indexed).")

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        """
        Return the top-k most relevant course text blobs for a given query.
        These are injected directly into the LLM prompt.
        """
        query_vec = np.array(self.model.encode([query]), dtype="float32")
        faiss.normalize_L2(query_vec)

        _, indices = self.index.search(query_vec, k)

        results = []
        for idx in indices[0]:
            if idx < len(self.courses):
                c = self.courses[idx]
                attrs = ", ".join(c.get("attributes", []))
                results.append(
                    f"Course {c['number']}: {c['title']}\n"
                    f"  Units: {c.get('units', 'N/A')} | "
                    f"Prereqs: {c.get('prereqs', 'None')} | "
                    f"Attributes: {attrs}\n"
                    f"  {c.get('description', '')}"
                )
        return results


if __name__ == "__main__":
    # Quick smoke test
    r = Retriever()
    print("\n--- Test query: 'ethics and moral philosophy' ---")
    for hit in r.retrieve("ethics and moral philosophy", k=3):
        print(hit)
        print()

    print("--- Test query: 'machine learning with math prereqs' ---")
    for hit in r.retrieve("machine learning with math prereqs", k=3):
        print(hit)
        print()

    print("--- Test query: 'HASS social science about culture' ---")
    for hit in r.retrieve("HASS social science about culture", k=3):
        print(hit)
        print()
