#!/usr/bin/env python3
"""Parse Table 5.5 (per-school enrollment forecast) from the 2025 PRC PDF.

Source PDF:
  data/raw/pps_enrollment_forecast_2025.pdf

Table 5.5 spans pages 37-40 in the PDF. Each row is:
  Name | Type | Program | Grades | 2022-23 | 2023-24 | 2024-25 |
  2025-26 | 2026-27 | 2027-28 | 2028-29 | 2029-30 |
  2030-31 | 2031-32 | 2032-33 | 2033-34 | 2034-35

Type ∈ {ES, MS, HS, K8, K12, G28, -}. Program ∈ {Total, Neighborhood,
Mandarin, Spanish, Russian, Japanese, Vietnamese, PISA}. We parse all
rows (including non-Total program rows) so downstream code can pick
Total-per-school or slice by program.

Output:
  data/raw/pps_enrollment_forecast.csv
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "data" / "raw" / "pps_enrollment_forecast_2025.pdf"
OUT = ROOT / "data" / "raw" / "pps_enrollment_forecast.csv"

TABLE_PAGES = (37, 38, 39, 40)  # 1-indexed
TYPE_TOKENS = {"ES", "MS", "HS", "K8", "K12", "G28", "-"}
YEAR_COLS = [
    "hist_2022_23", "hist_2023_24", "hist_2024_25",
    "fcst_2025_26", "fcst_2026_27", "fcst_2027_28", "fcst_2028_29",
    "fcst_2029_30", "fcst_2030_31", "fcst_2031_32", "fcst_2032_33",
    "fcst_2033_34", "fcst_2034_35",
]

# Rows we skip (summary lines, not schools)
SKIP_NAMES = {
    "Elementary Schools Subtotal",
    "Middle Schools Subtotal",
    "High Schools Subtotal",
    "Other (K-8, 2-8, and K-12)",
    "TOTAL",
}

# pdfplumber often concatenates multi-word names (no space). Expand them.
NAME_FIXUPS = {
    "HarrietTubman": "Harriet Tubman",
    "OckleyGreen": "Ockley Green",
    "HarrisonPark": "Harrison Park",
    "RosewayHeights": "Roseway Heights",
    "WestSylvan": "West Sylvan",
    "MtTabor": "Mt Tabor",
    "BeverlyCleary": "Beverly Cleary",
    "BridgerCreativeScience": "Bridger Creative Science",
    "SunnysideEnvironmental": "Sunnyside Environmental",
    "RoseCityPark": "Rose City Park",
    "IdaBWells-Barnett": "Ida B Wells-Barnett",
    "BensonPolytechnic": "Benson Polytechnic",
    "CapitolHill": "Capitol Hill",
    "ChiefJoseph": "Chief Joseph",
    "CésarChávez": "César Chávez",
    "JamesJohn": "James John",
    "Boise-Eliot/Humboldt": "Boise-Eliot/Humboldt",
    "MLKJr": "MLK Jr",
    "ForestPark": "Forest Park",
    "RosaParks": "Rosa Parks",
    "daVinci": "da Vinci",
    "Other(incl.Charters)": "Other (incl. Charters)",
}


def group_words_into_rows(words, y_tolerance: int = 2) -> list[list[dict]]:
    """Bucket words into rows by their top y-coordinate."""
    rows: dict[int, list[dict]] = {}
    for w in words:
        key = round(w["top"] / y_tolerance) * y_tolerance
        rows.setdefault(key, []).append(w)
    return [
        sorted(rows[k], key=lambda w: w["x0"])
        for k in sorted(rows.keys())
    ]


def parse_row(words: list[dict]) -> dict | None:
    """Parse a single data row. Returns None for non-data rows (headers, etc.)."""
    texts = [w["text"] for w in words]

    # Find the Type token. It's the first word matching TYPE_TOKENS after
    # at least one name token.
    type_idx = None
    for i, t in enumerate(texts):
        if i >= 1 and t in TYPE_TOKENS:
            type_idx = i
            break
    if type_idx is None:
        return None

    # Must have Type + Program + Grades + 13 values after the name.
    if len(texts) < type_idx + 1 + 2 + len(YEAR_COLS):
        return None

    name = " ".join(texts[:type_idx])
    name = NAME_FIXUPS.get(name, name)

    type_ = texts[type_idx]
    program = texts[type_idx + 1]
    grades = texts[type_idx + 2]
    value_tokens = texts[type_idx + 3 : type_idx + 3 + len(YEAR_COLS)]

    values: list[int | None] = []
    for t in value_tokens:
        t2 = t.replace(",", "")
        if t2 in ("-", ""):
            values.append(None)
        else:
            try:
                values.append(int(t2))
            except ValueError:
                return None

    if name in SKIP_NAMES:
        return None

    row = {
        "school_name": name,
        "type": type_,
        "program": program,
        "grades": grades,
    }
    for col, val in zip(YEAR_COLS, values):
        row[col] = val
    return row


def main() -> int:
    if not PDF.exists():
        print(f"ERROR: {PDF} missing — run fetch_pps_enrollment_forecast.py first")
        return 1

    all_rows: list[dict] = []
    with pdfplumber.open(PDF) as pdf:
        for page_num in TABLE_PAGES:
            pg = pdf.pages[page_num - 1]
            words = pg.extract_words()
            for line in group_words_into_rows(words):
                parsed = parse_row(line)
                if parsed:
                    all_rows.append(parsed)

    # Sanity check: Total rows per (name, type) should be unique.
    totals = [r for r in all_rows if r["program"] == "Total"]
    seen = set()
    dupes = []
    for r in totals:
        key = (r["school_name"], r["type"], r["grades"])
        if key in seen:
            dupes.append(key)
        seen.add(key)
    if dupes:
        print(f"WARN: duplicate Total rows: {dupes}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["school_name", "type", "program", "grades", *YEAR_COLS]
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print(f"wrote {OUT}: {len(all_rows)} rows ({len(totals)} Total rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
