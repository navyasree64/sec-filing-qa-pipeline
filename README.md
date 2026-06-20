# Financial Document Q&A Pipeline

A pipeline that takes a 10-K SEC filing and automatically produces a verified dataset
of question-answer pairs, suitable for benchmarking AI models on financial document
understanding. Built for the Caliper Lab internship technical assignment.

Uses Groq's free API (no credit card required, generous daily quota) so the entire
pipeline can be run at zero cost.

## What it does

```
SEC EDGAR 10-K  -->  parse & chunk  -->  generate Q&A (LLM)  -->  verify (LLM + rules)  -->  clean dataset
```

1. **`fetch_filing.py`** — downloads the most recent 10-K for a given company directly
   from SEC EDGAR's public JSON API (no API key needed for SEC; just a polite
   User-Agent header, per their access policy).
2. **`chunk_filing.py`** — strips HTML noise and splits the filing into sections using
   the standardized "Item N" headings that SEC mandates for every 10-K (Item 1 Business,
   Item 1A Risk Factors, Item 7 MD&A, Item 8 Financial Statements, etc.). Long sections
   are further split into smaller sub-chunks along paragraph boundaries so each chunk
   comfortably fits in one LLM call.
3. **`generate_qa.py`** — sends each chunk to Groq (running Llama 3.1) with instructions
   to produce several question-answer pairs *grounded only in that chunk*, each tagged
   with a question type (fact extraction / numeric calculation / comparison /
   multi-step reasoning) and a difficulty estimate, plus a verbatim quote of the
   supporting text.
4. **`verify_qa.py`** — the verification step. Every pair goes through two independent
   checks before being accepted:
   - **Quote-grounding check (deterministic, no LLM):** confirms the "source_passage"
     the generator returned is an actual (fuzzy-matched) substring of the original
     chunk text. This catches the most common failure mode — a fabricated or
     paraphrased "quote" — without spending any API calls or trusting any model.
   - **Independent LLM verification:** a *separate* Groq call, given only the
     question/answer/passage (not the generation reasoning), is asked to judge
     whether the answer is genuinely supported. This is a distinct prompt and task
     from generation, which reduces (but doesn't eliminate) the risk of the model
     simply agreeing with itself.
   Only pairs passing both checks make it into the final dataset. Rejected pairs are
   kept in a separate file for transparency, not silently dropped.
5. **`build_dataset.py`** — formats verified pairs into the required columns and writes
   `output/dataset.csv` and `output/dataset.json`.
6. **`topup_qa.py`** (optional, run if the main pass falls short of 100 pairs) — reuses
   the existing `chunks.json` and generates an additional batch of questions from
   whichever sections verified reliably in the main run, verifies just that new batch,
   and merges it into `qa_pairs_verified.json` for `build_dataset.py` to pick up. This
   is how the actual sample dataset in `output/` reached 111 pairs after the main run
   alone produced 62 (see Known Limitations for why Financial Statements chunks
   under-verified relative to prose sections).

Run everything with one command via `run_pipeline.py`, or run each script individually
(useful for debugging or re-running just one stage without re-spending API credits).

## Design choices

- **Why "Item N" regex chunking instead of an ML-based section classifier:** 10-K
  structure is legally standardized by the SEC, so a lightweight regex on heading
  patterns is simple, fast, free, fully explainable, and works across virtually any
  company's filing without training data or fine-tuning. A learned classifier would be
  more robust to messy edge cases but is overkill for this scope.
- **Why a separate verification step instead of trusting the generator:** the
  assignment explicitly requires this, and it matters in practice — LLMs generating
  Q&A from a passage will occasionally invent a plausible-sounding number or
  misattribute a fact from a *different* part of the document. Decoupling "generate"
  and "judge" into separate calls, plus a non-LLM quote check, catches a meaningful
  share of these without making either step bear that whole burden alone.
- **Why JSON-structured LLM output:** asking the model to return strict JSON (rather
  than free-text it has to parse with fragile regex) makes the whole pipeline far more
  reliable and easy to validate programmatically.
- **Why Groq as the model provider:** its free tier requires no credit card and gives
  a daily quota (14,400 requests/day on `llama-3.1-8b-instant`) that comfortably covers
  both generation and verification calls for a 100+ pair dataset in a single run.
  **This was a deliberate pivot from an earlier version of this pipeline that used
  Google's Gemini free tier** — Gemini's free-tier daily quota turned out to be far
  tighter in practice (as low as 20 requests/day on the account this was tested with)
  than published documentation suggested, which made completing a 100+ pair dataset in
  one day impossible. This is a useful lesson in its own right: free-tier quotas vary
  by account/region/time and are worth verifying empirically rather than trusting any
  single published number. The pipeline is provider-agnostic in design — swapping in a
  different model/provider only requires changing the API call in `generate_qa.py` and
  `verify_qa.py`.
  **Documented tradeoff:** Groq serves open-weight models (Llama) rather than a
  frontier closed model. Quality is good for structured extraction tasks like this one,
  but may lag a notch behind GPT/Gemini/Claude-class models on the subtlest
  multi-step-reasoning questions — an acceptable tradeoff for free, high-volume use.
- **Why a section allowlist and a hard cap on total chunks:** `chunk_filing.py` only
  chunks the five sections richest in extractable facts (Business, Risk Factors, MD&A,
  Market Risk, Financial Statements) and caps the total chunk count, prioritizing MD&A
  and Financial Statements first. This was added after discovering that an unfiltered
  10-K produces 80+ chunks — far more than needed for a 100+ pair dataset, and more
  than some free-tier daily quotas can support in one run. Capping chunks (rather than
  processing everything) keeps the pipeline's API usage predictable and the dataset
  focused on the sections that produce the best questions.
- **CSV + JSON output:** CSV for quick human review in Excel/Sheets, JSON for clean
  programmatic downstream use.

## Known limitations

- **Section detection is regex-based and not bulletproof.** Some filings have
  inconsistent HTML structure (extra nested tags, unusual whitespace, tables rendered
  as text) that can occasionally cause a heading to be missed or a chunk boundary to
  land awkwardly. This is a reasonable baseline, not a fully robust parser for every
  10-K ever filed.
- **Tables are flattened to plain text.** Financial statements are often in HTML
  tables; converting them to plain text loses some structure (e.g. column alignment),
  which can make it harder for the LLM to generate clean numeric-calculation questions
  from dense tables. A more advanced version would parse tables structurally (e.g. with
  pandas `read_html`) rather than relying on flattened text.
  **This showed up concretely in the actual dataset run**: Financial Statements chunks
  (table-derived) had a meaningfully higher verification rejection rate than prose
  sections, with the verifier's most common objection being that a number's
  fiscal-year attribution was ambiguous once the table's column structure was lost in
  flattening (e.g. "the passage does not specify the year ended September 27, 2025").
  Rather than loosen the verifier to force more Financial Statements pairs through,
  additional pairs were generated from sections that already demonstrated reliable
  grounding (Business, Risk Factors, Market Risk) to reach the 100+ pair target — see
  `topup_qa.py`. This was a deliberate choice to preserve dataset quality over forcing
  volume from a structurally weaker source.
- **Question type and difficulty distribution is skewed toward easy fact extraction.**
  The final dataset is roughly 94% `fact_extraction` and 73% `easy`, with only a
  handful of `comparison`, `multi_step_reasoning`, and `numeric_calculation` pairs.
  This is a direct downstream consequence of the table-flattening issue above: the
  sections richest in the numbers needed for genuine calculation/comparison questions
  (Financial Statements) verified poorly, while the sections that verified reliably
  (Business, Risk Factors) are mostly prose describing single facts rather than
  numbers that invite calculation or comparison. All four required question types are
  present in the dataset, but not in a well-balanced mix. Improving table handling
  (see above) is the most direct way to fix this, since it would let
  numeric/comparison questions from Financial Statements survive verification at a
  rate comparable to prose sections.
- **LLM verification is not infallible.** It catches clear-cut hallucinations and
  unsupported claims well, but a sufficiently subtle wrong answer could pass if the
  verifier model makes the same mistake the generator did. This is a real, acknowledged
  ceiling on this approach, not just this implementation.
- **Single-document only in this version.** The pipeline processes one filing at a time
  end-to-end; see the scaling note below for how this extends.
- **No deduplication across questions.** With multiple chunks discussing similar
  numbers (e.g. total revenue appearing in both MD&A and the financial statements),
  some near-duplicate questions may appear. A similarity-based dedup pass would be a
  natural next addition.
- **Only a subset of the filing is used.** To keep total API calls predictable under
  free-tier quotas, `chunk_filing.py` caps the number of chunks taken from each section
  (see `MAX_CHUNKS_PER_SECTION`) rather than processing every section of the 10-K (e.g.
  Legal Proceedings, Properties, and exhibits are skipped entirely via
  `ALLOWED_ITEM_NUMBERS`). For a research or production dataset, raising these caps and
  accepting a longer/more expensive run would give broader document coverage.
- **Open-weight model quality ceiling.** Using a free, open-weight model (Llama 3.1 8B
  via Groq) trades a small amount of quality, particularly on the subtlest
  multi-step-reasoning questions, for zero cost and a high-volume daily quota. A
  production version benchmarking frontier models would likely want to upgrade the
  generation and/or verification model.

## How to run it

1. **Get a free Groq API key** (no credit card required):
   - Go to [console.groq.com](https://console.groq.com)
   - Sign up (email, Google, or GitHub)
   - Click "API Keys" in the left sidebar → "Create API Key"

2. **Install and run:**
   ```bash
   pip install -r requirements.txt
   export GROQ_API_KEY="your-key-here"

   cd src
   # Edit COMPANY_CIK and USER_AGENT at the top of fetch_filing.py first
   python run_pipeline.py
   ```

Output lands in `output/dataset.csv` and `output/dataset.json`.

**Note on rate limits:** Groq's free tier is generous (thousands of requests/day), but
`generate_qa.py` and `verify_qa.py` still pace their calls with a short delay between
each to stay comfortably within per-minute limits. If you hit a rate-limit error
anyway, the scripts retry automatically with backoff.

**On chunk/question volume:** `chunk_filing.py` caps chunks per section via
`MAX_CHUNKS_PER_SECTION` (e.g. 4 each from MD&A, Financial Statements, and Risk
Factors), and `generate_qa.py` asks for `QUESTIONS_PER_CHUNK` (default 9) pairs per
chunk. In practice, the main run alone did not reliably clear 100 verified pairs,
because Financial Statements chunks had a notably higher verification rejection rate
than prose sections (see Known Limitations). If your main run falls short of 100:

```bash
python topup_qa.py      # generates additional pairs from the sections that verified well
python build_dataset.py # rebuilds the final dataset.csv/json from the merged set
```

`topup_qa.py` reuses the existing `chunks.json` (no need to re-fetch or re-chunk) and
generates more pairs specifically from whichever sections proved reliable in the main
run, then verifies and merges them in rather than lowering the verification bar to
force more pairs from a weaker source.

## Scaling to multiple documents / 1000+ pairs

- **Multiple documents:** wrap `run_pipeline.py` in an outer loop over a list of CIKs
  (or tickers resolved to CIKs via SEC's `company_tickers.json`), writing each
  company's output to its own subfolder, then concatenate all `dataset.json` files at
  the end with a `company`/`filing_date` column added for provenance.
- **Higher volume per document:** increase `QUESTIONS_PER_CHUNK` in `generate_qa.py`,
  and/or chunk more finely (lower `MAX_CHUNK_CHARS`) to get more, narrower chunks —
  more chunks means more generation calls means more pairs, at the cost of more API
  spend.
- **Parallelization:** chunk-level generation and verification calls are independent
  of each other, so they're naturally parallelizable — e.g. with a thread pool or
  `asyncio` batching multiple chunks' API calls concurrently, respecting the API
  provider's rate limits. This would be the single biggest speed win at scale.
- **Cost/latency control at scale:** Groq's free tier (14,400 requests/day) has
  significantly more headroom than this single-document pipeline uses, so scaling to
  several documents in one day is realistic without paid tiers. Beyond that volume,
  enabling a paid tier (on Groq or another provider) would remove the daily ceiling
  entirely for a modest cost.
- **Quality monitoring at scale:** track the rejection rate from `verify_qa.py` per
  section type over time — a rising rejection rate for a given section (e.g. dense
  financial tables) is an early signal that chunking or prompting needs tuning for that
  content type before scaling further.
- **Deduplication at scale:** with many chunks likely producing semantically similar
  questions, an embedding-based similarity filter across the final dataset would catch
  near-duplicates that simple per-chunk generation can't see.

## Sample output

See `output/dataset.csv` / `output/dataset.json` for a real generated dataset from
Apple's FY2025 10-K filing (see filing details in `data/filing_metadata.json`).

The included sample contains 111 verified Q&A pairs: 104 fact_extraction, 4
numeric_calculation, 2 comparison, 1 multi_step_reasoning; 81 easy, 21 medium, 9 hard.
See Known Limitations above for why the type/difficulty mix is skewed toward easy
fact-extraction rather than evenly distributed — this is a direct, documented
consequence of how table-derived sections (rich in calculation/comparison material)
verified less reliably than prose sections once flattened to plain text.
