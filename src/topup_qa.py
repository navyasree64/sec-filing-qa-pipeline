"""
topup_qa.py

Generates ADDITIONAL Q&A pairs from the sections that verified well in the first run
(Business, Risk Factors, Market Risk -- prose sections, not dense tables), then
verifies just the new batch and merges everything into the final dataset.

WHY THIS EXISTS:
After a first full pipeline run, you may end up with fewer than 100 verified pairs --
for example, if Financial Statements chunks (HTML tables flattened to plain text)
produced a lot of ambiguous "which year does this number belong to" rejections, while
prose sections like Business and Risk Factors verified at a much higher rate. Rather
than re-running fetch + chunk + generate + verify from scratch (slow, and would
re-encounter the exact same table-structure issue), this script:

1. Reuses the existing chunks.json (no need to re-download or re-chunk the filing)
2. Generates a fresh, ADDITIONAL batch of questions from just the
   highest-verification-rate sections (edit TOPUP_SECTIONS below if needed)
3. Verifies only this new batch (not the pairs you already verified)
4. Merges the new verified pairs with your existing qa_pairs_verified.json
5. Rebuilds the final dataset.csv / dataset.json from the merged, larger set

This is a deliberate, documented design choice: topping up from sections that already
demonstrated high signal quality, rather than lowering the verification bar or padding
with weaker pairs from sections known to produce more ambiguous results.

USAGE:
  cd src
  set GROQ_API_KEY=your-key-here     (Windows; reuse the same key as before)
  python topup_qa.py
  python build_dataset.py             (rebuild final dataset.csv/json from the merged set)
"""

import json
import os
import time
from generate_qa import call_groq_for_chunk, SLEEP_BETWEEN_CALLS
from verify_qa import fuzzy_contains, llm_verify

# Only top up from sections that verified well in the first run. Edit this if your
# own first-run results showed a different section performing best.
TOPUP_SECTIONS = {"Business", "Risk Factors", "Market Risk"}

# How many EXTRA questions to generate per qualifying chunk this round.
EXTRA_QUESTIONS_PER_CHUNK = 12


def main():
    data_dir = "../data"

    with open(os.path.join(data_dir, "chunks.json"), "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    topup_chunks = [c for c in all_chunks if c["section_label"] in TOPUP_SECTIONS]
    print(f"Found {len(topup_chunks)} chunks in top-up sections: {TOPUP_SECTIONS}")
    print(f"Generating ~{EXTRA_QUESTIONS_PER_CHUNK} additional pairs from each...\n")

    new_raw_pairs = []
    for i, chunk in enumerate(topup_chunks):
        print(f"[{i+1}/{len(topup_chunks)}] Generating extra pairs for {chunk['chunk_id']} ({chunk['section_label']})...")
        qa_list = call_groq_for_chunk(chunk, n=EXTRA_QUESTIONS_PER_CHUNK)

        for qa in qa_list:
            qa["chunk_id"] = chunk["chunk_id"]
            qa["section_label"] = chunk["section_label"]
            qa["source_chunk_text"] = chunk["text"]
            new_raw_pairs.append(qa)

        time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"\nGenerated {len(new_raw_pairs)} new raw pairs. Verifying...\n")

    new_verified = []
    new_rejected = []
    for i, qa in enumerate(new_raw_pairs):
        print(f"[{i+1}/{len(new_raw_pairs)}] Checking: {qa['question'][:70]}...")

        if not fuzzy_contains(qa["source_chunk_text"], qa["source_passage"]):
            qa["rejection_reason"] = "source_passage not found in original chunk (failed quote-grounding check)"
            new_rejected.append(qa)
            print("    REJECTED: quote not found in source chunk")
            continue

        verdict = llm_verify(qa)
        if verdict.get("is_supported") is True:
            qa["verification_note"] = verdict.get("reason", "")
            new_verified.append(qa)
            print("    PASSED")
        else:
            qa["rejection_reason"] = verdict.get("reason", "LLM judged answer not supported")
            new_rejected.append(qa)
            print(f"    REJECTED: {qa['rejection_reason']}")

        time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"\nNew batch: {len(new_verified)} passed, {len(new_rejected)} rejected.")

    # Merge with existing verified/rejected pairs from the first run
    existing_verified_path = os.path.join(data_dir, "qa_pairs_verified.json")
    existing_rejected_path = os.path.join(data_dir, "qa_pairs_rejected.json")

    existing_verified = []
    if os.path.exists(existing_verified_path):
        with open(existing_verified_path, "r", encoding="utf-8") as f:
            existing_verified = json.load(f)

    existing_rejected = []
    if os.path.exists(existing_rejected_path):
        with open(existing_rejected_path, "r", encoding="utf-8") as f:
            existing_rejected = json.load(f)

    merged_verified = existing_verified + new_verified
    merged_rejected = existing_rejected + new_rejected

    with open(existing_verified_path, "w", encoding="utf-8") as f:
        json.dump(merged_verified, f, indent=2)
    with open(existing_rejected_path, "w", encoding="utf-8") as f:
        json.dump(merged_rejected, f, indent=2)

    print(f"\nMerged total: {len(merged_verified)} verified pairs (was {len(existing_verified)}, added {len(new_verified)}).")
    print("Now run: python build_dataset.py")


if __name__ == "__main__":
    main()
