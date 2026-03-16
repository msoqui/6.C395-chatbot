from sentence_transformers import SentenceTransformer
_preload = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

import re
from src.retriever import Retriever
from src.chat import Chatbot

retriever = Retriever()
chatbot = Chatbot()

# ── Test cases ──────────────────────────────────────────────────────────────
# Format: (query, expected_course, required_attr, dept_prefix, semester)
RETRIEVAL_TESTS = [
    # Basic dept retrieval
    ("intro microeconomics",                "14.01",  None,     "14.", None),
    ("undergrad IO econ",                   "14.20",  None,     "14.", None),
    ("machine learning course 6",           "6.3900", None,     "6.",  None),
    ("intro to inference",                  "6.3800", None,     "6.",  None),
    ("linear algebra",                      "18.06",  None,     "18.", None),

    # Attribute filtering
    ("CI-H ethics course",                  None,     "CI-H",   None,  None),
    ("HASS-S no prerequisites",             None,     "HASS-S", None,  None),
    ("REST requirement physics",            None,     "REST",   None,  None),

    # Semester filtering
    ("fall classes CS",                     None,     "fall",   "6.",  "fall"),
    ("spring urban planning classes",       None,     "spring", "11.", "spring"),

    # Undergrad/grad filtering
    ("UG econ classes",                 None,     "undergrad", "14.", None),
    ("graduate biology",                None,     "graduate",  "7.",  None),
]

# ── Metric 1: Retrieval precision@10 ────────────────────────────────────────
def test_retrieval_precision(tests, k=10):
    hits = 0
    total = 0
    for query, expected, _, _, _ in tests:
        if not expected:
            continue
        results = retriever.retrieve(query, k=k)
        found = any(expected in r for r in results)
        hits += found
        total += 1
        print(f"  {'✓' if found else '✗'} '{query}' → {expected}")
    print(f"\nRetrieval precision@{k}: {hits}/{total} ({100*hits//total}%)\n")
    return hits / total

# ── Metric 2: Attribute filter compliance ───────────────────────────────────
def test_filter_compliance(tests, k=10):
    hits = 0
    total = 0
    for query, _, required_attr, _, _ in tests:
        if not required_attr:
            continue
        results = retriever.retrieve(query, k=k)
        if not results:
            print(f"  ✗ '{query}' → no results")
            total += 1
            continue
        compliant = sum(1 for r in results if required_attr in r)
        rate = compliant / len(results)
        hits += rate
        total += 1
        print(f"  {'✓' if rate == 1.0 else '~'} '{query}' → {compliant}/{len(results)} results have {required_attr} ({rate:.0%})")
    print(f"\nFilter compliance: {hits/total:.0%}\n")
    return hits / total

# ── Metric 3: Dept precision ─────────────────────────────────────────────────
def test_dept_precision(tests, k=10):
    total_rate = 0
    total = 0
    for query, _, _, dept_prefix, _ in tests:
        if not dept_prefix:
            continue
        results = retriever.retrieve(query, k=k)
        if not results:
            print(f"  ✗ '{query}' → no results")
            total += 1
            continue
        dept_hits = sum(1 for r in results if f"Course {dept_prefix}" in r)
        rate = dept_hits / len(results)
        total_rate += rate
        total += 1
        print(f"  {'✓' if rate >= 0.5 else '✗'} '{query}' → {dept_hits}/{len(results)} from dept {dept_prefix} ({rate:.0%})")
    print(f"\nDept precision: {total_rate/total:.0%}\n")
    return total_rate / total

# ── Metric 4: Response hallucination check ───────────────────────────────────
HALLUCINATION_TESTS = [
    # 14.01 is MW 1pm
    ("What time is 14.01 offered?", ["1", "1:00", "MonWed", "MW"]),
    
    # 6.1000 has no rating in Hydrant (newer course)
    ("What's the rating for 6.1000?", ["don't have", "no rating", "not available", "catalog"]),
    
    # 6.1020 is offered in spring
    ("Is 6.1020 offered in the spring?", ["yes", "spring", "offered"]),
    
    # 6.C395 instructor
    ("Who teaches 6.C395?", ["S. Mullainathan"]),

    # Scheduling
    ("What time is 6.1210 offered?",            ["TueThu", "11", "26-100"]),
    ("What's the schedule for 18.06?",          ["MonWedFri", "10", "26-100"]),
    ("When does 6.1800 meet?",                  ["MonWed", "1", "26-100"]),
    ("What time is 6.1020?",                    ["TueThu", "9.30", "9:30"]),

    # Ratings
    ("What's the rating for 6.1010?",           ["5.8", "5.9", "5.87"]),
    ("How is 14.20 rated?",                     ["6.6", "6.56"]),
    ("What's the rating for 18.02?",            ["5.3"]),

    # Instructors  
    ("Who teaches 6.1210?",                     ["Chapman"]),
    ("Who is the instructor for 18.06?",        ["Urschel"]),
    ("Who teaches 6.1800?",                     ["LaCurts"]),
    ("Who teaches 6.3900?",                     ["Shen"]),

    # Semester availability
    ("Is 14.20 offered in the fall?",           ["no", "spring", "not offered", "only"]),
    ("Is 6.1010 offered in the fall?",          ["yes", "fall", "offered"]),
    ("Is 6.1800 offered in the fall?",          ["no", "spring", "not offered", "only"]),

    # Undergrad/grad
    ("Is 6.1210 an undergrad or grad course?",  ["undergraduate", "undergrad"]),
]

def test_hallucination(tests):
    passed = 0
    for query, expected_signals in tests:
        response = chatbot.get_response(query, [])
        print(response)
        found = any(signal.lower() in response.lower() for signal in expected_signals)
        passed += found
        print(f"  {'✓' if found else '✗'} '{query}'")
        if not found:
            print(f"    Response: {response[:150]}")
    print(f"\nHallucination check: {passed}/{len(tests)} ({100*passed//len(tests)}%)\n")
    return passed / len(tests)

# ── Run all ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("1. RETRIEVAL PRECISION@10")
print("=" * 60)
p = test_retrieval_precision(RETRIEVAL_TESTS)

print("=" * 60)
print("2. ATTRIBUTE FILTER COMPLIANCE")
print("=" * 60)
fc = test_filter_compliance(RETRIEVAL_TESTS)

print("=" * 60)
print("3. DEPARTMENT PRECISION")
print("=" * 60)
dp = test_dept_precision(RETRIEVAL_TESTS)

print("=" * 60)
print("4. HALLUCINATION CHECK")
print("=" * 60)
h = test_hallucination(HALLUCINATION_TESTS)

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Retrieval precision@10:    {p:.0%}")
print(f"  Filter compliance:         {fc:.0%}")
print(f"  Department precision:      {dp:.0%}")
print(f"  Hallucination check:       {h:.0%}")