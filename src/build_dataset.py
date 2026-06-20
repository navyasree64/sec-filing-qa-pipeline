"""
build_dataset.py

Takes the verified Q&A pairs and writes the final, clean dataset in the exact column
format the assignment requires, as both CSV and JSON.

WHAT THIS DOES (plain English):
1. Load qa_pairs_verified.json
2. Rename/select fields to match exactly what was requested:
   question, ground_truth_answer, source_passage, question_type, difficulty
   (plus a few extra provenance columns: section, chunk_id -- useful, not required,
   and clearly additive rather than replacing any required column)
3. Write out dataset.csv (for quick viewing in Excel/Sheets) and dataset.json
4. Print summary stats (counts by type/difficulty) so you can include them in your README

WHY THIS APPROACH:
- Keeping the raw/verified/final files separate (rather than overwriting in place) means
  the full pipeline is auditable -- anyone can trace a final row back through
  verification to the exact generation call and source chunk.
"""

import json
import csv
import os

REQUIRED_COLUMNS = [
    "question",
    "ground_truth_answer",
    "source_passage",
    "question_type",
    "difficulty",
]

EXTRA_COLUMNS = [
    "section",
    "chunk_id",
]


def main():
    data_dir = "../data"
    output_dir = "../output"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(data_dir, "qa_pairs_verified.json"), "r", encoding="utf-8") as f:
        verified = json.load(f)

    rows = []
    for qa in verified:
        rows.append({
            "question": qa["question"],
            "ground_truth_answer": qa["answer"],
            "source_passage": qa["source_passage"],
            "question_type": qa["question_type"],
            "difficulty": qa["difficulty"],
            "section": qa.get("section_label", ""),
            "chunk_id": qa.get("chunk_id", ""),
        })

    # Write JSON (preserves structure cleanly)
    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # Write CSV (easy to open in Excel/Sheets for a quick look)
    csv_path = os.path.join(output_dir, "dataset.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS + EXTRA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Final dataset: {len(rows)} verified Q&A pairs")
    print(f"Saved to {json_path} and {csv_path}\n")

    if len(rows) < 100:
        print(f"WARNING: only {len(rows)} rows -- assignment asks for at least 100.")
        print("Fix: increase QUESTIONS_PER_CHUNK in generate_qa.py, or chunk more finely,")
        print("or run the pipeline on a second filing and merge the datasets.\n")

    # Summary stats -- useful to paste into the README
    type_counts = {}
    diff_counts = {}
    for r in rows:
        type_counts[r["question_type"]] = type_counts.get(r["question_type"], 0) + 1
        diff_counts[r["difficulty"]] = diff_counts.get(r["difficulty"], 0) + 1

    print("By question type:")
    for k, v in sorted(type_counts.items()):
        print(f"  {k}: {v}")

    print("\nBy difficulty:")
    for k, v in sorted(diff_counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
