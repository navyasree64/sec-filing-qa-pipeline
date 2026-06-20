"""
chunk_filing.py

Parses the raw 10-K HTML and splits it into meaningful chunks for Q&A generation.

WHAT THIS DOES (plain English):
1. 10-K filings are HTML documents, but full of formatting noise (tags, scripts, styles)
   that we don't want an LLM to see. We strip all that out and get clean text.
2. Every 10-K, by SEC law, is organized into numbered "Items" (Item 1, Item 1A, Item 7,
   etc.) -- these are standardized across ALL companies, which is what makes automatic
   section-splitting possible without per-company custom logic.
3. We find these "Item X" headings in the text and use them as chunk boundaries.
4. Because some Items (especially Item 7 - MD&A, and Item 8 - Financial Statements) are
   very long, we further split big sections into smaller sub-chunks by paragraph,
   capped at a max character length -- this keeps each chunk small enough to fit
   comfortably in an LLM prompt and focused enough to generate good, grounded questions.

WHY THIS APPROACH:
- Regex-based section detection on the standardized "Item N" headers is simple, requires
  no ML, and works across virtually any 10-K because the structure is SEC-mandated.
- This is a known limitation we're upfront about in the README: real-world 10-Ks have
  messy/inconsistent HTML (extra whitespace, tables, nested divs), so the section
  detection isn't perfect on every filing -- but it's a solid, explainable baseline.
"""

import re
import json
import os
from bs4 import BeautifulSoup

MAX_CHUNK_CHARS = 4500  # larger chunks = fewer total chunks = fewer API calls needed
MIN_CHUNK_CHARS = 200    # skip tiny fragments (e.g. table-of-contents leftovers)

# Only chunk these sections -- the ones that actually contain useful Q&A material.
# Skips boilerplate/legal sections (Mine Safety, Properties, Exhibits, signatures, etc.)
# that rarely contain meaningful financial content, to keep the total chunk count (and
# therefore API call count) manageable under free-tier rate limits.
ALLOWED_ITEM_NUMBERS = {"1", "1a", "7", "7a", "8"}

# Max chunks taken from EACH section, rather than one global cap. This guarantees
# section diversity in the final dataset -- without a per-section cap, one very long
# section (e.g. Financial Statements, full of dense tables) can produce so many
# sub-chunks that it alone fills the entire chunk budget, crowding out every other
# section. Item 7A (Market Risk) is usually short, so it gets a smaller cap; MD&A and
# Financial Statements get the most since they're richest in extractable facts.
MAX_CHUNKS_PER_SECTION = {"7": 4, "8": 4, "1a": 4, "1": 2, "7a": 1}

# Standard 10-K item headings we look for as section boundaries.
# Order matters for some downstream logic but not strictly required here.
ITEM_PATTERN = re.compile(
    r"(Item\s+\d{1,2}[A-Z]?\.?\s*[-–—.]?\s*[A-Z][A-Za-z,;:&'()\- ]{3,160}?)(?=\n|\.\s|$)",
    re.IGNORECASE,

)

# Human-friendly labels for the sections we most care about (used for question-type hints)
SECTION_LABELS = {
    "1a": "Risk Factors",
    "1": "Business",
    "7": "MD&A (Management Discussion and Analysis)",
    "7a": "Market Risk",
    "8": "Financial Statements",
}


def html_to_clean_text(html_path: str) -> str:
    """Strip HTML tags/scripts/styles and collapse whitespace into clean plain text."""
    with open(html_path, "r", encoding="utf-8") as f:
        raw_html = f.read()

    soup = BeautifulSoup(raw_html, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Collapse excessive blank lines / whitespace left over from HTML layout
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_items(text: str) -> list:
    """
    Split the full document text into (heading, body_text) pairs using 'Item N' headings
    as boundaries. Returns a list of dicts: {"heading": ..., "text": ...}
    """
    matches = list(ITEM_PATTERN.finditer(text))
    sections = []

    for i, match in enumerate(matches):
        heading = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if len(body) >= MIN_CHUNK_CHARS:
            sections.append({"heading": heading, "text": body})

    return sections


def get_item_number(heading: str) -> str:
    """Extract just the item number (e.g. '1A', '7') from a heading string for labeling."""
    m = re.search(r"Item\s+(\d{1,2}[A-Z]?)", heading, re.IGNORECASE)
    return m.group(1).lower() if m else ""


def split_long_section(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list:
    """
    Split a long section into smaller pieces along paragraph boundaries, keeping each
    piece under max_chars. We avoid splitting mid-paragraph so we don't cut a sentence
    (and its supporting numbers) in half.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= max_chars:
            current += (" " + para if current else para)
        else:
            if len(current) >= MIN_CHUNK_CHARS:
                chunks.append(current)
            current = para

    if len(current) >= MIN_CHUNK_CHARS:
        chunks.append(current)

    return chunks if chunks else [text[:max_chars]]


def build_chunks(html_path: str) -> list:
    """
    Main entry point: turn a raw 10-K HTML file into a list of chunk dicts, each with:
      - chunk_id
      - section_heading (raw heading text, e.g. "Item 7. Management's Discussion...")
      - section_label (friendly name, e.g. "MD&A")
      - text (the actual chunk content fed to the LLM)
    """
    print("Cleaning HTML...")
    text = html_to_clean_text(html_path)

    print("Splitting into Item sections...")
    sections = split_into_items(text)
    print(f"Found {len(sections)} raw sections before sub-chunking.")

    # Filter to only the sections worth generating Q&A from. This keeps the total
    # number of chunks (and therefore API calls) manageable under free-tier rate
    # limits, and avoids wasting calls on boilerplate (mine safety, exhibits, etc.)
    sections = [s for s in sections if get_item_number(s["heading"]) in ALLOWED_ITEM_NUMBERS]
    print(f"Kept {len(sections)} sections after filtering to: "
          f"Business, Risk Factors, MD&A, Market Risk, Financial Statements.")

    chunks = []
    chunk_id = 0
    # Track how many chunks we've taken from each section so far, so one long section
    # (e.g. Financial Statements, full of dense tables) can't crowd out the others.
    section_chunk_counts = {}

    for section in sections:
        item_num = get_item_number(section["heading"])
        label = SECTION_LABELS.get(item_num, section["heading"][:60])
        per_section_cap = MAX_CHUNKS_PER_SECTION.get(item_num, 2)

        sub_chunks = split_long_section(section["text"])
        for sub_text in sub_chunks:
            taken_so_far = section_chunk_counts.get(item_num, 0)
            if taken_so_far >= per_section_cap:
                break  # this section has hit its quota; move on to the next section

            chunk_id += 1
            section_chunk_counts[item_num] = taken_so_far + 1
            chunks.append({
                "chunk_id": f"chunk_{chunk_id:03d}",
                "section_heading": section["heading"],
                "section_label": label,
                "text": sub_text,
            })

    return chunks


def main():
    # Find the downloaded 10-K html file in ../data
    data_dir = "../data"
    html_files = [f for f in os.listdir(data_dir) if f.endswith(".html")]
    if not html_files:
        raise FileNotFoundError("No .html filing found in ../data. Run fetch_filing.py first.")

    html_path = os.path.join(data_dir, html_files[0])
    print(f"Parsing {html_path}...")

    chunks = build_chunks(html_path)
    print(f"Produced {len(chunks)} chunks total (after sub-splitting long sections).")

    out_path = os.path.join(data_dir, "chunks.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"Saved chunks to {out_path}")

    # Print a quick summary so you can sanity-check the split
    label_counts = {}
    for c in chunks:
        label_counts[c["section_label"]] = label_counts.get(c["section_label"], 0) + 1
    print("\nChunk counts by section:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
