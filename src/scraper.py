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

DEPARTMENTS = [
    "m1a", "m2a", "m3a", "m4a", "m5a", "m6a", "m7a", "m8a", "m9a",
    "m10a", "m11a", "m12a", "m14a", "m15a", "m16a", "m17a", "m18a",
    "m20a", "m21a", "m21Aa", "mCMSa", "m21Wa", "m21Ga", "m21Ha",
    "m21La", "m21Ma", "m21Ta", "mWGSa", "m22a", "m24a", "mCCa",
    "mCGa", "mCSBa", "mCSEa", "mECa", "mEMa", "mESa", "mHSTa",
    "mIDSa", "mMASa", "mSCMa", "mASa", "mMSa", "mNSa", "mSTSa",
    "mSWEa", "mSPa",
]

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


# ---------------------------------------------------------------------------
# Main scraper
# ---------------------------------------------------------------------------

def scrape_all(output_path='data/courses.json', delay=0.5):
    """Scrape all departments and write to output_path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_courses = []
    seen_numbers = set()
    failed_depts = []

    for dept in DEPARTMENTS:
        url = f"{BASE_URL}{dept}.html"
        print(f"Scraping {dept}...", end=' ', flush=True)
        try:
            resp = requests.get(url, timeout=20, headers={'User-Agent': 'Mozilla/5.0'})
            resp.raise_for_status()
            courses = parse_department_page(resp.text)

            added = 0
            for course in courses:
                num = course['number']
                if num not in seen_numbers:
                    seen_numbers.add(num)
                    all_courses.append(course)
                    added += 1

            print(f"{added} new courses (running total: {len(all_courses)})")
        except Exception as e:
            print(f"FAILED — {e}")
            failed_depts.append(dept)

        time.sleep(delay)

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
