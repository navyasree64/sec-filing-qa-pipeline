"""
run_pipeline.py

Runs the full pipeline end-to-end, in order:
  1. fetch_filing.py    -- download a 10-K from SEC EDGAR
  2. chunk_filing.py     -- parse + split into sections/chunks
  3. generate_qa.py      -- generate Q&A pairs per chunk via Groq
  4. verify_qa.py         -- independently verify each pair
  5. build_dataset.py     -- write the final clean dataset

USAGE:
  cd src
  python run_pipeline.py

You can also run each step individually (useful while debugging, or to avoid
re-spending API credits on steps that already succeeded -- e.g. if generation
finished fine but you want to re-run only verification).
"""

import subprocess
import sys

STEPS = [
    ("fetch_filing.py", "Fetching 10-K filing from SEC EDGAR"),
    ("chunk_filing.py", "Parsing and chunking the document"),
    ("generate_qa.py", "Generating Q&A pairs with Groq"),
    ("verify_qa.py", "Verifying Q&A pairs against source text"),
    ("build_dataset.py", "Building final dataset"),
]


def run_step(script_name: str, description: str):
    print("=" * 70)
    print(f"STEP: {description}")
    print("=" * 70)
    result = subprocess.run([sys.executable, script_name])
    if result.returncode != 0:
        print(f"\nStep '{script_name}' failed (exit code {result.returncode}). Stopping pipeline.")
        sys.exit(1)
    print()


def main():
    for script_name, description in STEPS:
        run_step(script_name, description)

    print("=" * 70)
    print("Pipeline complete! Final dataset is in ../output/dataset.csv and dataset.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
