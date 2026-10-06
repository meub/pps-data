"""Download and parse PPS's annual Language Immersion Enrollment report.

The PDF is published yearly by PPS Research, Assessment & Accountability
and gives per-school counts of immersion-strand students vs. non-immersion
students, which ODE's Fall Membership cannot separate. Source page:
https://www.pps.net/departments/research-assessment-and-accountability/data-and-reports/enrollment-reports-and-school-profiles
"""
import json
import re
import requests
import pdfplumber
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT / "data/raw/pps_immersion_details_2526.pdf"
JSON_PATH = ROOT / "data/raw/pps_immersion_details_2526.json"

# 2025-26 report resource UUID (pps.net/fs/resource-manager).
URL = "https://www.pps.net/fs/resource-manager/view/87c13af2-3974-4efd-930c-d897fae34e4f"


DLI_LANGUAGES = ["Spanish", "Mandarin", "Japanese", "Russian", "Vietnamese"]


def _split_count_pct(cell):
    """PDF cells like '287\n51%' → (287, 0.51). Empty/None → (None, None)."""
    if not cell or cell.strip() in ("", "-"):
        return None, None
    parts = [p.strip() for p in cell.split("\n") if p.strip()]
    count = None
    pct = None
    for p in parts:
        if p.endswith("%"):
            try:
                pct = round(float(p.rstrip("%")) / 100, 4)
            except ValueError:
                pass
        else:
            try:
                count = int(p.replace(",", ""))
            except ValueError:
                pass
    return count, pct


def parse_pdf(pdf_path):
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for row in table:
                    if not row or not row[0]:
                        continue
                    name = row[0].strip()
                    if not name or name.lower() == "name":
                        continue
                    total, _ = _split_count_pct(row[1])
                    if total is None:
                        continue
                    dli_count, _ = _split_count_pct(row[4])
                    dli_pct = round(dli_count / total, 4) if dli_count and total else None
                    # Columns 5-9 split the immersion total by language, in the
                    # report's fixed order. Strands are taught as separate
                    # classes, so the rightsizing class model needs them.
                    by_lang = {}
                    for col, lang in zip(range(5, 10), DLI_LANGUAGES):
                        n, _ = _split_count_pct(row[col]) if len(row) > col else (None, None)
                        if n:
                            by_lang[lang] = n
                    rows.append({
                        "name": name,
                        "total_enrollment": total,
                        "dli_students": dli_count,
                        "dli_pct": dli_pct,
                        "dli_by_language": by_lang,
                    })
    return rows


def main():
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not PDF_PATH.exists():
        r = requests.get(URL, timeout=60)
        r.raise_for_status()
        PDF_PATH.write_bytes(r.content)
        print(f"Downloaded {len(r.content):,} bytes to {PDF_PATH}")
    rows = parse_pdf(PDF_PATH)
    JSON_PATH.write_text(json.dumps(rows, indent=2))
    print(f"Parsed {len(rows)} schools → {JSON_PATH}")
    for r in rows[:5]:
        print(f"  {r['name']:<22} total={r['total_enrollment']:<5} "
              f"dli={r['dli_students']:<4} pct={r['dli_pct']}")


if __name__ == "__main__":
    main()
