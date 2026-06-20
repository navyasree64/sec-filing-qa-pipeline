"""
fetch_filing.py

Downloads the most recent 10-K filing for a given company from SEC EDGAR.

WHAT THIS DOES (plain English):
1. SEC EDGAR identifies companies by a number called a "CIK" (Central Index Key).
   Apple's CIK is 320193. You can find any company's CIK by searching:
   https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company=YOUR_COMPANY&type=10-K
2. We ask SEC's API: "give me the list of all filings this company has made"
3. We filter that list down to just 10-K filings, and grab the most recent one
4. We download the actual filing document (it's a big HTML file) and save it locally

WHY THIS APPROACH:
- SEC EDGAR requires a "User-Agent" header on every request -- basically you must say
  who you are (name + email) so they can contact you if your script misbehaves.
  No API key needed, it's all free and public.
- We use the official JSON API (data.sec.gov) instead of scraping HTML search pages --
  it's more reliable and is what SEC actually recommends for automated access.
"""

import requests
import time
import json
import os

# ---- CONFIG: edit these two things ----
COMPANY_CIK = "320193"   # Apple Inc. Find others at https://www.sec.gov/cgi-bin/browse-edgar
USER_AGENT = "CaliperLabAssignment madalanavyasree@gmail.com"  # <-- put YOUR real name/email here
# ----------------------------------------

HEADERS = {"User-Agent": USER_AGENT}


def pad_cik(cik: str) -> str:
    """SEC wants CIK numbers as 10 digits with leading zeros, e.g. 320193 -> 0000320193"""
    return cik.zfill(10)


def get_filing_list(cik: str) -> dict:
    """Fetch the full list of this company's SEC filings (metadata only, not documents)."""
    url = f"https://data.sec.gov/submissions/CIK{pad_cik(cik)}.json"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def find_latest_10k(filings_json: dict) -> dict:
    """
    The filings list is structured as parallel arrays (one list of forms, one list of
    dates, one list of accession numbers, etc. -- all the same length, indexed together).
    We loop through and find the first entry where form == "10-K".
    """
    recent = filings_json["filings"]["recent"]
    forms = recent["form"]
    accession_numbers = recent["accessionNumber"]
    primary_docs = recent["primaryDocument"]
    filing_dates = recent["filingDate"]

    for i, form in enumerate(forms):
        if form == "10-K":
            return {
                "accession_number": accession_numbers[i],
                "primary_document": primary_docs[i],
                "filing_date": filing_dates[i],
            }
    raise ValueError("No 10-K filing found for this company.")


def build_document_url(cik: str, accession_number: str, primary_document: str) -> str:
    """
    Construct the URL to the actual filing document.
    Accession numbers come formatted like '0000320193-23-000106' but the URL path
    needs them WITHOUT dashes.
    """
    acc_no_dashes = accession_number.replace("-", "")
    cik_no_pad = str(int(cik))  # folder uses CIK without leading zeros
    return f"https://www.sec.gov/Archives/edgar/data/{cik_no_pad}/{acc_no_dashes}/{primary_document}"


def download_filing(url: str, output_path: str):
    """Download the filing HTML and save it to disk."""
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(resp.text)
    print(f"Saved filing to {output_path}")


def main():
    print(f"Looking up filings for CIK {COMPANY_CIK}...")
    filings_json = get_filing_list(COMPANY_CIK)
    company_name = filings_json.get("name", "Unknown Company")
    print(f"Company: {company_name}")

    time.sleep(0.2)  # be polite to SEC's servers, stay well under their rate limit

    latest_10k = find_latest_10k(filings_json)
    print(f"Found 10-K filed on {latest_10k['filing_date']}")

    doc_url = build_document_url(
        COMPANY_CIK,
        latest_10k["accession_number"],
        latest_10k["primary_document"],
    )
    print(f"Document URL: {doc_url}")

    time.sleep(0.2)

    os.makedirs("../data", exist_ok=True)
    output_path = f"../data/{company_name.replace(' ', '_')}_10K_{latest_10k['filing_date']}.html"
    download_filing(doc_url, output_path)

    # Save metadata too, useful for the README / dataset provenance
    metadata = {
        "company_name": company_name,
        "cik": COMPANY_CIK,
        "filing_date": latest_10k["filing_date"],
        "source_url": doc_url,
    }
    with open("../data/filing_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Done. Metadata saved to ../data/filing_metadata.json")


if __name__ == "__main__":
    main()
