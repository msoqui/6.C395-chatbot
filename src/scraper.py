"""
src/scraper.py — MIT Course Catalog Scraper

Fetches all courses from student.mit.edu/catalog and saves to data/courses.json.
Run once offline before deploying:

    python src/scraper.py

The output file is committed to the repo so the HuggingFace Space never scrapes.
"""

import json
import os
import re
import time

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://student.mit.edu/catalog/"

DEPT_PREFIXES = [
    "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9",
    "m10", "m11", "m12", "m14", "m15", "m16", "m17", "m18",
    "m20", "m21", "m21A", "mCMS", "m21W", "m21G", "m21H",
    "m21L", "m21M", "m21T", "mWGS", "m22", "m24", "mCC",
    "mCG", "mCSB", "mCSE", "mEC", "mEM", "mES", "mHST",
    "mIDS", "mMAS", "mSCM", "mAS", "mMS", "mNS", "mSTS",
    "mSWE", "mSP",
]


def all_dept_pages():
    """Yield all department page slugs by auto-discovering a/b/c/... sub-pages."""
    for prefix in DEPT_PREFIXES:
        for letter in "abcdefghij":
            yield prefix + letter

# Maps icon filename → human-readable attribute label.
# Confirmed by inspecting m6a, m24a, m21a, m21Aa, m21Ma pages.
ICON_MAP = {
    "rest.gif":    "REST",
    "hassH.gif":   "HASS-H",
    "hassA.gif":   "HASS-A",
    "hassS.gif":   "HASS-S",
    "hassT.gif":   "HASS-E",
    "hassAH.gif":  "HASS-AH",
    "cih1.gif":    "CI-H",
    "cim.gif":     "CI-M",
    "Lab.gif":     "Institute-Lab",
    "nooffer.gif": "not-offered-2526",
    "nonext.gif":  "not-offered-2627",
    "repeat.gif":  "repeatable",
    "under.gif":   "undergrad",
    "grad.gif":    "graduate",
    "fall.gif":    "fall",
    "spring.gif":  "spring",
    "iap.gif":     "iap",
    "summer.gif":  "summer",
}

# Course numbers look like: 6.1000, 18.01, 24.00, 21A.100, WGS.101, CSE.C20
COURSE_NUMBER_RE = re.compile(r'^[A-Z0-9]+[A-Z0-9]*\.[A-Z0-9]+', re.IGNORECASE)

# ---------------------------------------------------------------------------
# Event-based DOM walker
# ---------------------------------------------------------------------------

def collect_events(node, events):
    """
    Recursively walk a BeautifulSoup node, appending to `events` as either:
        ('img', filename)   — for <img> tags
        ('text', string)    — for non-empty text nodes

    Walking depth-first preserves document order, which matters for detecting
    the second hr.gif that separates metadata from the description.
    """
    if isinstance(node, NavigableString):
        text = str(node).strip()
        if text:
            events.append(('text', text))
    elif isinstance(node, Tag):
        if node.name == 'img':
            src = node.get('src', '')
            fname = os.path.basename(src)
            events.append(('img', fname))
        else:
            for child in node.children:
                collect_events(child, events)


# ---------------------------------------------------------------------------
# Course block parser
# ---------------------------------------------------------------------------

def parse_course_block(h3, siblings):
    """
    Given an <h3> tag and all siblings up to the next <h3>, extract:
        number, title, units, prereqs, attributes, description
    """
    # --- Number and title ---
    raw = re.sub(r'\s+', ' ', h3.get_text(' ', strip=True)).strip()
    parts = raw.split(' ', 1)
    number = parts[0].strip()
    title  = parts[1].strip() if len(parts) > 1 else ''

    # --- Flatten all sibling content into an ordered event list ---
    events = []
    for sib in siblings:
        collect_events(sib, events)

    # --- Split events into pre-description and description at 2nd hr.gif ---
    hr_count   = 0
    pre_events = []
    desc_events = []

    for event in events:
        etype, content = event
        if etype == 'img' and content == 'hr.gif':
            hr_count += 1
            # Don't add hr.gif itself to either list
            continue
        if hr_count >= 1:
            desc_events.append(event)
        else:
            pre_events.append(event)

    # --- Extract requirement attributes from icons in pre-description block ---
    attributes = []
    for etype, content in pre_events:
        if etype == 'img' and content in ICON_MAP:
            attr = ICON_MAP[content]
            if attr not in attributes:
                attributes.append(attr)

    # --- Extract prereqs and units from pre-description text ---
    pre_text = ' '.join(c for t, c in pre_events if t == 'text')

    prereq = ''
    m = re.search(r'Prereq(?:uisite)?s?:\s*(.+?)(?=Units?:|$)', pre_text, re.IGNORECASE)
    if m:
        prereq = re.sub(r'\s+', ' ', m.group(1)).strip().rstrip('.')

    units = ''
    m = re.search(r'Units?:\s*([\d\-]+(?:\s+or\s+[\d\-]+)?)', pre_text, re.IGNORECASE)
    if m:
        units = m.group(1).strip()

    # --- Build description from post-hr2 text events ---
    description = ' '.join(c for t, c in desc_events if t == 'text')
    description = re.sub(r'\s+', ' ', description).strip()

    return {
        'number':      number,
        'title':       title,
        'units':       units,
        'prereqs':     prereq,
        'attributes':  attributes,
        'description': description,
    }


# ---------------------------------------------------------------------------
# Page parser
# ---------------------------------------------------------------------------

def parse_department_page(html):
    """Parse all course entries from one department catalog page."""
    soup = BeautifulSoup(html, 'html.parser')
    courses = []

    h3_tags = soup.find_all('h3')
    for h3 in h3_tags:
        first_word = h3.get_text(' ', strip=True).split()[0] if h3.get_text(strip=True) else ''

        # Skip non-course h3s (section headers, navigation, etc.)
        if not COURSE_NUMBER_RE.match(first_word):
            continue

        # Collect all siblings until the next h3
        siblings = []
        for sib in h3.next_siblings:
            if isinstance(sib, Tag) and sib.name == 'h3':
                break
            siblings.append(sib)

        course = parse_course_block(h3, siblings)
        courses.append(course)

    return courses

def fetch_hydrant_data():
    """Fetch schedule and rating data from Hydrant."""
    resp = requests.get("https://hydrant.mit.edu/latest.json", timeout=20)
    data = resp.json()
    return data.get("classes", {})

def decode_schedule(raw_sections: list) -> str:
    """
    Decode Hydrant raw section strings.
    Format: "room/days/isEvening/times"
    Days: M=Mon, T=Tue, W=Wed, R=Thu, F=Fri
    times: human-readable time string like "1-2" or "11"
    """
    if not raw_sections or raw_sections == ["TBA"]:
        return "TBA"

    DAY_MAP = {"M": "Mon", "T": "Tue", "W": "Wed", "R": "Thu", "F": "Fri"}
    schedules = []

    for raw in raw_sections:
        parts = raw.split("/")
        if len(parts) < 4:
            continue
        room = parts[0]
        days_str = parts[1]
        is_evening = parts[2] == "1"
        times = parts[3]

        days = "".join(DAY_MAP.get(d, d) for d in days_str)

        if is_evening:
            schedules.append(f"{days} EVE ({times}) ({room})")
        else:
            schedules.append(f"{days} {times} ({room})")

    return "; ".join(schedules) if schedules else "TBA"

GIR_MAP = {
    "CAL1": "Calculus I (GIR)",
    "CAL2": "Calculus II (GIR)", 
    "PHY1": "Physics I (GIR)",
    "PHY2": "Physics II (GIR)",
    "CHEM": "Chemistry (GIR)",
    "BIO":  "Biology (GIR)",
    "REST": "REST (GIR)",
}

def merge_hydrant(courses, hydrant_data):
    """Merge Hydrant data into scraped courses."""
    matched = 0
    for course in courses:
        num = course['number']
        # Try exact match first, then without [J] suffix
        h = hydrant_data.get(num) or hydrant_data.get(num.replace('[J]', '').strip())
        if h:
            matched += 1
            course['schedule'] = decode_schedule(h.get('lectureRawSections', []))
            course['instructors'] = h.get('inCharge', '')
            course['rating'] = h.get('rating')
            course['hours'] = h.get('hours')
            course['enrollment'] = h.get('size')
            course['level'] = 'undergrad' if h.get('level') == 'U' else 'graduate'
            gir = h.get('gir', '')
            if gir and gir in GIR_MAP:
                if GIR_MAP[gir] not in course['attributes']:
                    course['attributes'].append(GIR_MAP[gir])
    print(f"Hydrant matched {matched}/{len(courses)} courses")
    return courses
# ---------------------------------------------------------------------------
# Main scraper
# ---------------------------------------------------------------------------

def scrape_all(output_path='data/courses.json', delay=0.5):
    """Scrape all departments and write to output_path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_courses = []
    seen_numbers = set()
    failed_depts = []

    current_prefix = None
    for dept in all_dept_pages():
        # Detect when we move to a new department prefix
        prefix = dept[:-1]
        if prefix != current_prefix:
            current_prefix = prefix

        url = f"{BASE_URL}{dept}.html"
        try:
            resp = requests.get(url, timeout=20, headers={'User-Agent': 'Mozilla/5.0'})
            if resp.status_code == 404:
                continue  # This sub-page doesn't exist, try the next letter
            resp.raise_for_status()
        except requests.HTTPError:
            continue
        except Exception as e:
            print(f"  {dept}: FAILED — {e}")
            failed_depts.append(dept)
            time.sleep(delay)
            continue

        courses = parse_department_page(resp.text)
        added = 0
        for course in courses:
            num = course['number']
            if num not in seen_numbers:
                seen_numbers.add(num)
                all_courses.append(course)
                added += 1

        print(f"  {dept}: {added} new courses (total: {len(all_courses)})")
        time.sleep(delay)

    print("Fetching Hydrant data...")
    hydrant_data = fetch_hydrant_data()
    all_courses = merge_hydrant(all_courses, hydrant_data)

    # --- Write output ---
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_courses, f, indent=2, ensure_ascii=False)

    # --- Sanity checks ---
    total        = len(all_courses)
    with_desc    = sum(1 for c in all_courses if c['description'])
    with_units   = sum(1 for c in all_courses if c['units'])
    rest_count   = sum(1 for c in all_courses if 'REST'  in c['attributes'])
    cih_count    = sum(1 for c in all_courses if 'CI-H'  in c['attributes'])
    hass_count   = sum(1 for c in all_courses if any(a.startswith('HASS') for a in c['attributes']))
    grad_count   = sum(1 for c in all_courses if 'graduate' in c['attributes'])
    nooff_count  = sum(1 for c in all_courses if 'not-offered-2526' in c['attributes'])

    print(f"\n{'='*50}")
    print(f"Saved {total} courses → {output_path}")
    print(f"{'='*50}")
    print(f"  With description : {with_desc}/{total} ({100*with_desc//total}%)")
    print(f"  With units       : {with_units}/{total} ({100*with_units//total}%)")
    print(f"  REST             : {rest_count}")
    print(f"  CI-H             : {cih_count}")
    print(f"  HASS (any)       : {hass_count}")
    print(f"  Graduate only    : {grad_count}")
    print(f"  Not offered 25-26: {nooff_count}")

    if failed_depts:
        print(f"\nFailed departments: {failed_depts}")
    else:
        print("\nAll departments scraped successfully.")

    return all_courses


if __name__ == '__main__':
    scrape_all()
