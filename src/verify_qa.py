"""
verify_qa.py

Independently verifies each generated Q&A pair against its source passage, to catch
hallucinated or unsupported answers before they make it into the final dataset.

WHAT THIS DOES (plain English):
This is the "verification step" the assignment explicitly asks for. The idea: the model
that GENERATED a question/answer might be wrong or might have invented something not
actually in the text. So we don't just trust it -- we run a SEPARATE check:

1. For each Q&A pair, we give a fresh model call ONLY the question, the answer, and
   the quoted source_passage (NOT the reasoning that produced it).
2. We ask it: "Is this answer fully and correctly supported by this passage? Does the
   quoted source_passage actually appear in / match the chunk it claims to come from?"
3. We also do a CHEAP, NON-LLM check first: confirm the "source_passage" string the
   generator gave us is an actual substring (or near-exact match) of the original
   chunk text. This catches outright fabricated quotes without spending any API calls.
4. Only pairs that pass BOTH checks go into the final dataset. Failed pairs are saved
   separately for transparency/debugging, not silently discarded.

WHY THIS APPROACH (and why it's a real verification, not just asking the same model
to "double check itself"):
- Using a programmatic substring/fuzzy-match check first is a deterministic,
  non-hallucinate-able way to catch the most common failure mode: a "source_passage"
  that the generation step invented or paraphrased instead of quoting exactly.
- The LLM verification call is a SEPARATE, independent prompt with a narrower task
  (just judging support, not generating) which reduces the risk of the model
  rubber-stamping its own prior output -- it's not even told this was machine-generated.
- This two-layer approach (cheap deterministic check + targeted LLM check) is a known,
  practical pattern for scalable hallucination detection. It is not perfect (an LLM
  judge can still be wrong) -- this limitation is called out explicitly in the README.

WHY GROQ: same free-tier rationale as generate_qa.py -- a 14,400 requests/day quota
comfortably covers both generation and verification calls for a 100+ pair dataset in
one run, with no need to wait for or manage a local model on this run.
"""

import json
import os
import time
import re
from difflib import SequenceMatcher
from groq import Groq

MODEL = "llama-3.1-8b-instant"
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 6  # seconds; free tier is ~30 requests/minute AND has a tokens-per-minute cap, so pace conservatively
FUZZY_MATCH_THRESHOLD = 0.85  # how close source_passage must be to a real substring

_api_key = os.environ.get("GROQ_API_KEY")
if not _api_key:
    raise RuntimeError(
        "No API key found. Set it before running, e.g. on Windows:\n"
        "    set GROQ_API_KEY=your-key-here\n"
        "then run this script in the SAME terminal window."
    )
client = Groq(api_key=_api_key)

VERIFY_PROMPT = """You are a strict fact-checker reviewing a question-answer pair that claims to be supported by a passage from a financial document (a 10-K SEC filing).

PASSAGE:
\"\"\"
{passage}
\"\"\"

QUESTION: {question}
PROPOSED ANSWER: {answer}

Determine whether the proposed answer is fully and correctly supported by the passage above. The answer must be derivable using ONLY information in the passage -- no outside knowledge, no assumptions, no invented numbers.

Respond with ONLY a valid JSON object, no markdown, no preamble:
{{
  "is_supported": true or false,
  "reason": "one short sentence explaining your judgment"
}}
"""


def fuzzy_contains(haystack: str, needle: str, threshold: float = FUZZY_MATCH_THRESHOLD) -> bool:
    """
    Checks whether 'needle' (the claimed source_passage) genuinely appears in 'haystack'
    (the original chunk text) -- exact substring match OR a close fuzzy match, to allow
    for minor whitespace/punctuation differences without allowing fabricated quotes.
    """
    haystack_norm = re.sub(r"\s+", " ", haystack).strip()
    needle_norm = re.sub(r"\s+", " ", needle).strip()

    if needle_norm in haystack_norm:
        return True

    if len(needle_norm) < 10:
        return False

    best_ratio = 0.0
    window = len(needle_norm)
    step = max(1, window // 4)
    for start in range(0, max(1, len(haystack_norm) - window), step):
        candidate = haystack_norm[start:start + window]
        ratio = SequenceMatcher(None, candidate, needle_norm).ratio()
        best_ratio = max(best_ratio, ratio)
        if best_ratio >= threshold:
            return True

    return best_ratio >= threshold


def llm_verify(qa: dict) -> dict:
    prompt = VERIFY_PROMPT.format(
        passage=qa["source_passage"],
        question=qa["question"],
        answer=qa["answer"],
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            raw_text = response.choices[0].message.content.strip()
            raw_text = re.sub(r"^```(json)?", "", raw_text).strip()
            raw_text = re.sub(r"```$", "", raw_text).strip()
            return json.loads(raw_text)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                print(f"  [retry {attempt+1}] Rate limit hit, waiting 30s...")
                time.sleep(30)
            else:
                print(f"  [retry {attempt+1}] verification error: {e}")
                time.sleep(5)

    return {"is_supported": False, "reason": "verification call failed after retries"}


def main():
    data_dir = "../data"
    raw_path = os.path.join(data_dir, "qa_pairs_raw.json")

    with open(raw_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    print(f"Verifying {len(qa_pairs)} Q&A pairs...\n")

    verified = []
    rejected = []

    for i, qa in enumerate(qa_pairs):
        print(f"[{i+1}/{len(qa_pairs)}] Checking: {qa['question'][:70]}...")

        # Layer 1: deterministic quote-grounding check (no API call, catches fabricated quotes)
        if not fuzzy_contains(qa["source_chunk_text"], qa["source_passage"]):
            qa["rejection_reason"] = "source_passage not found in original chunk (failed quote-grounding check)"
            rejected.append(qa)
            print("    REJECTED: quote not found in source chunk")
            continue

        # Layer 2: independent LLM judgment on whether the answer is actually supported
        verdict = llm_verify(qa)
        if verdict.get("is_supported") is True:
            qa["verification_note"] = verdict.get("reason", "")
            verified.append(qa)
            print("    PASSED")
        else:
            qa["rejection_reason"] = verdict.get("reason", "LLM judged answer not supported")
            rejected.append(qa)
            print(f"    REJECTED: {qa['rejection_reason']}")

        time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"\n{len(verified)} pairs PASSED verification.")
    print(f"{len(rejected)} pairs REJECTED.")

    with open(os.path.join(data_dir, "qa_pairs_verified.json"), "w", encoding="utf-8") as f:
        json.dump(verified, f, indent=2)

    with open(os.path.join(data_dir, "qa_pairs_rejected.json"), "w", encoding="utf-8") as f:
        json.dump(rejected, f, indent=2)

    print("Saved qa_pairs_verified.json and qa_pairs_rejected.json")


if __name__ == "__main__":
    main()
