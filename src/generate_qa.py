"""
generate_qa.py

For each text chunk, calls Groq (a free, fast LLM API) to generate question-answer
pairs grounded in that chunk, tagged with question type and difficulty.

WHAT THIS DOES (plain English):
1. Load the chunks we produced in chunk_filing.py
2. For each chunk, send it to the model with detailed instructions: "read this passage,
   write N questions that can ONLY be answered using this passage, give the exact
   answer, quote the exact sentence(s) that prove the answer, and label the
   question's type and difficulty."
3. We ask for the response in strict JSON so it's easy to parse programmatically.
4. We save every generated Q&A pair, along with which chunk it came from, so later
   the verification step can re-check it against the same source text.

WHY GROQ (and not a paid API):
- Groq's free tier requires no credit card and gives a daily quota of 14,400 requests
  on "llama-3.1-8b-instant" -- comfortably enough for generating 100+ Q&A pairs AND
  verifying every one of them, in a single run, on a single free account. This was
  switched to from Google's Gemini free tier after hitting a much tighter daily quota
  (as low as 20 requests/day on some accounts) that made same-day completion
  impossible at this dataset's scale.
- A local model via Ollama was also considered (zero rate limits at all, runs on your
  own machine) but was set aside in favor of Groq for this run: local CPU inference on
  a laptop is roughly 10-30x slower per call than Groq's hosted API, which would turn a
  ~15-20 minute pipeline run into multiple hours for ~150 total calls. Groq's actual
  rate limits (30 requests/minute, 14,400/day) are comfortably above what this
  pipeline needs once paced correctly, making it the faster and more practical choice
  under a tight deadline. The pipeline's provider-agnostic design means switching to
  Ollama later is a small, contained change if rate limits ever become a blocker again.
- Tradeoff, documented here and in the README: Groq serves open-weight models
  (Llama, in this case) rather than a frontier closed model. Quality is good for this
  structured-extraction task, but may be a notch below GPT/Gemini/Claude-class models
  on subtler reasoning questions -- an acceptable tradeoff for free, high-volume use.

WHY THIS APPROACH (independent of provider):
- Asking for structured JSON output (rather than free-form text) makes the pipeline
  reliable and machine-readable -- no fragile regex-parsing of prose answers needed.
- We explicitly instruct the model to quote its supporting text VERBATIM. This is what
  lets the verification step (verify_qa.py) later check that the quote is real and not
  fabricated, and that it actually appears in the source chunk.
- We deliberately ask for a *mix* of question types per chunk (not just simple fact
  lookup) so the final dataset has good coverage across difficulty levels, as the
  assignment requires.
"""

import json
import os
import time
import re
from groq import Groq

MODEL = "llama-3.1-8b-instant"  # Groq's most generous free-tier daily quota
QUESTIONS_PER_CHUNK = 9          # tuned to comfortably clear 100+ pairs after verification losses
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 6  # seconds; free tier is ~30 requests/minute AND has a tokens-per-minute cap, so pace conservatively

_api_key = os.environ.get("GROQ_API_KEY")
if not _api_key:
    raise RuntimeError(
        "No API key found. Set it before running, e.g. on Windows:\n"
        "    set GROQ_API_KEY=your-key-here\n"
        "then run this script in the SAME terminal window."
    )
client = Groq(api_key=_api_key)

GENERATION_PROMPT = """You are creating a benchmark dataset of question-answer pairs from a section of a company's 10-K SEC filing. Your questions will be used to test whether an AI model can correctly read and reason about financial documents.

SOURCE PASSAGE (from section: "{section_label}"):
\"\"\"
{chunk_text}
\"\"\"

Generate exactly {n} question-answer pairs based ONLY on information contained in the passage above. Do not use any outside knowledge about this company.

Requirements for each pair:
- The question must be answerable using ONLY the passage above.
- The answer must be the exact correct answer, concise (a number, short phrase, or 1-2 sentences).
- "source_passage" must be an EXACT, VERBATIM quote of the sentence(s) from the passage that prove/support the answer. Do not paraphrase this field -- copy it exactly as it appears above.
- "question_type" must be one of: "fact_extraction", "numeric_calculation", "comparison", "multi_step_reasoning"
  - fact_extraction: answer is directly stated in the passage
  - numeric_calculation: answer requires doing math on numbers in the passage (e.g. computing a difference, percentage, ratio)
  - comparison: answer requires comparing two or more values/items mentioned in the passage
  - multi_step_reasoning: answer requires combining multiple pieces of information from the passage in a logical chain
- "difficulty" must be one of: "easy", "medium", "hard"
  - easy: single fact, directly stated, no computation
  - medium: requires one calculation step or combining two nearby facts
  - hard: requires multiple steps, careful reading, or synthesizing scattered information
- Try to generate a MIX of question types and difficulties across the {n} questions, not all the same type.
- If the passage doesn't contain enough numeric data to support a numeric_calculation or comparison question, generate fact_extraction or multi_step_reasoning questions instead. Do not invent numbers.

Respond with ONLY a valid JSON array, no preamble, no markdown code fences, no explanation. Format:
[
  {{
    "question": "...",
    "answer": "...",
    "source_passage": "...",
    "question_type": "...",
    "difficulty": "..."
  }}
]
"""


def salvage_partial_json_array(raw_text: str) -> list:
    """
    If the model's response gets cut off mid-array (e.g. hit a token limit partway
    through writing the JSON), this tries to recover whichever complete {...} objects
    came before the cutoff, rather than discarding the entire batch. This matters
    because regenerating a whole chunk from scratch costs another API call and more
    rate-limit pressure -- salvaging what's usable is cheaper and still produces
    correctly-grounded Q&A pairs, since each {...} object is self-contained valid JSON
    even if the surrounding array got cut short.
    """
    objects = []
    depth = 0
    start = None
    for i, ch in enumerate(raw_text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = raw_text[start:i + 1]
                try:
                    obj = json.loads(candidate)
                    objects.append(obj)
                except json.JSONDecodeError:
                    pass  # this object itself was malformed/cut off; skip it
                start = None
    return objects


def call_groq_for_chunk(chunk: dict, n: int = QUESTIONS_PER_CHUNK) -> list:
    prompt = GENERATION_PROMPT.format(
        section_label=chunk["section_label"],
        chunk_text=chunk["text"],
        n=n,
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4096,
            )
            raw_text = response.choices[0].message.content.strip()

            # Defensive: strip markdown fences if the model adds them anyway
            raw_text = re.sub(r"^```(json)?", "", raw_text).strip()
            raw_text = re.sub(r"```$", "", raw_text).strip()

            try:
                qa_list = json.loads(raw_text)
                return qa_list
            except json.JSONDecodeError:
                # Response likely got cut off mid-array. Try to salvage whichever
                # complete Q&A objects we can, rather than losing the whole batch.
                salvaged = salvage_partial_json_array(raw_text)
                if salvaged:
                    print(f"  Recovered {len(salvaged)}/{n} pairs from a truncated response for {chunk['chunk_id']}.")
                    return salvaged
                raise  # nothing salvageable; fall through to the except below and retry

        except json.JSONDecodeError as e:
            print(f"  [retry {attempt+1}] JSON parse failed for {chunk['chunk_id']}: {e}")
            time.sleep(2)
        except Exception as e:
            # 429 = rate limit hit (too many requests/tokens this minute). Back off much
            # longer than a generic error, since retrying immediately just fails again.
            if "429" in str(e) or "rate" in str(e).lower():
                print(f"  [retry {attempt+1}] Rate limit hit for {chunk['chunk_id']}, waiting 30s...")
                time.sleep(30)
            else:
                print(f"  [retry {attempt+1}] API error for {chunk['chunk_id']}: {e}")
                time.sleep(5)

    print(f"  Giving up on {chunk['chunk_id']} after {MAX_RETRIES} attempts.")
    return []


def main():
    data_dir = "../data"
    chunks_path = os.path.join(data_dir, "chunks.json")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks. Generating ~{QUESTIONS_PER_CHUNK} Q&A pairs each...")
    print(f"Target total: ~{len(chunks) * QUESTIONS_PER_CHUNK} pairs (before verification filtering)\n")

    all_qa_pairs = []
    for i, chunk in enumerate(chunks):
        print(f"[{i+1}/{len(chunks)}] Generating for {chunk['chunk_id']} ({chunk['section_label']})...")
        qa_list = call_groq_for_chunk(chunk)

        for qa in qa_list:
            qa["chunk_id"] = chunk["chunk_id"]
            qa["section_label"] = chunk["section_label"]
            qa["source_chunk_text"] = chunk["text"]  # kept for verification step
            all_qa_pairs.append(qa)

        time.sleep(SLEEP_BETWEEN_CALLS)

    out_path = os.path.join(data_dir, "qa_pairs_raw.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_qa_pairs, f, indent=2)

    print(f"\nGenerated {len(all_qa_pairs)} raw Q&A pairs (pre-verification).")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
