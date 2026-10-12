#!/usr/bin/env python3
"""Parse PPS's 2026-27 Individual School Reports (budget Volume 2).

Source PDF ("2026-27 Proposed Budget Volume #2", updated May 6, 2026):
  data/raw/pps_budget_2026-27_proposed_vol2.pdf

Two parts are read:
  - The School Summary heat map (two rotated pages): dollars per student
    for 2022-23 through 2026-27, students per FTE, and the direct-
    certification share, grouped by grade band.
  - Each school's two-page report: the enrollment "Total" row (five years
    of October counts plus three projected years).

Dollars per student here are PPS's own figure: school General Fund,
Title I, and grant dollars divided by enrollment. They exclude centrally
allocated staff (special education, ESL, nutrition, custodial), so they
undercount what a school costs to run. The page says so.

Text comes from poppler's `pdftotext -layout`, because the heat map is
rotated and pdfplumber returns it reversed. The parser fails if the heat
map row count or the report count differs from EXPECTED_SCHOOLS, so a
revised PDF cannot silently drop schools.

Output:
  data/raw/pps_budget_school_reports.json
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "data" / "raw" / "pps_budget_2026-27_proposed_vol2.pdf"
OUT = ROOT / "data" / "raw" / "pps_budget_school_reports.json"

SOURCE = {
    "title": "2026-27 Proposed Budget, Volume 2: Individual School Reports",
    "publisher": "Portland Public Schools",
    "updated": "2026-05-06",
    "url": "https://www.pps.net/departments/budget",
}

# The table of contents lists 83 reports; the heat map has the same 83 rows.
EXPECTED_SCHOOLS = 83

DOLLAR_YEARS = ["2022_23", "2023_24", "2024_25", "2025_26", "2026_27"]
ENROLL_YEARS = ["2021_22", "2022_23", "2023_24", "2024_25", "2025_26",
                "2026_27", "2027_28", "2028_29"]

BAND_HEADERS = {
    "K-5 SCHOOLS": "K-5",
    "K-8 SCHOOLS": "K-8",
    "MIDDLE SCHOOLS": "Middle",
    "HIGH SCHOOLS": "High",
}

# Heat-map name -> report-page name, where the two pages spell it differently.
REPORT_NAME = {
    "MLK Jr": "Martin Luther King Jr",
    "da Vinci": "Da Vinci",
}

TOC = re.compile(r"^\s*(?P<name>[A-Za-z].*?)\s*\.{3,}\s*\d+\s*$")
TOC_SKIP = {"Individual School Reports", "School Summary"}

ROW = re.compile(r"^\s{2,}(?P<name>[A-Za-z][A-Za-z .'\-/éá]+?)\s{2,}(?P<rest>\S.*)$")
MONEY = re.compile(r"\$\s?([\d,]+)")
PCT = re.compile(r"(\d+)%")
RATIO = re.compile(r"(?<![\d$,])(\d+\.\d)(?![\d%])")


def pdf_pages() -> list[str]:
    text = subprocess.run(
        ["pdftotext", "-layout", str(PDF), "-"],
        check=True, capture_output=True, text=True,
    ).stdout
    return text.split("\f")


def parse_heat_map(pages: list[str]) -> list[dict]:
    rows: list[dict] = []
    band = None
    for page in pages:
        if "Dollars per Student" not in page or "School Name" not in page:
            continue
        for line in page.splitlines():
            stripped = line.strip()
            if stripped in BAND_HEADERS:
                band = BAND_HEADERS[stripped]
                continue
            # Metro. Learning Center sits under MIDDLE SCHOOLS but is K-12.
            m = ROW.match(line)
            if not m or band is None or "$" not in m["rest"]:
                continue
            rest = m["rest"]
            money_start = rest.index("$")
            before, after = rest[:money_start], rest[money_start:]
            dollars = [int(x.replace(",", "")) for x in MONEY.findall(after)]
            after_money = MONEY.sub("", after)
            dc = PCT.search(after_money)
            school_type = after_money[dc.end():].strip() if dc else after_money.strip()
            pcts = [int(x) for x in PCT.findall(before)]
            ratios = [float(x) for x in RATIO.findall(before)]
            # Newer schools have blank early years; values are right-aligned.
            dollars = [None] * (len(DOLLAR_YEARS) - len(dollars)) + dollars
            ratios = [None] * (len(DOLLAR_YEARS) - len(ratios)) + ratios
            rows.append({
                "name": m["name"].strip(),
                "band": band,
                "school_type": school_type,
                "title_i": "Title I" in school_type,
                "proficient_math_2425": pcts[0] if len(pcts) == 2 else None,
                "proficient_ela_2425": pcts[1] if len(pcts) == 2 else None,
                "pct_direct_cert_2425": int(dc.group(1)) if dc else None,
                "dollars_per_student": dict(zip(DOLLAR_YEARS, dollars)),
                "students_per_fte": dict(zip(DOLLAR_YEARS, ratios)),
            })
    return rows


def report_names(pages: list[str]) -> list[str]:
    """School names from the table of contents, in report order."""
    names = []
    for page in pages:
        if "TABLE OF CONTENTS" not in page and not names:
            continue
        for line in page.splitlines():
            m = TOC.match(line)
            if m and m["name"].rstrip(" .") not in TOC_SKIP:
                names.append(m["name"].rstrip(" ."))
        if names and "Woodstock" in page:
            return names
    return names


def parse_enrollment(pages: list[str]) -> dict[str, dict]:
    """Report name -> {year: total enrollment}.

    Report headers wrap or carry ODE notes in the same row, so names come
    from the table of contents. Reports appear in TOC order; each one's
    header is checked to contain the TOC name's first word.
    """
    names = report_names(pages)
    # The intro page also says "Enrollment Data"; only reports have a
    # Projected column.
    reports = [p for p in pages if "Enrollment Data" in p and "Projected" in p]
    if len(names) != len(reports):
        sys.exit(f"TOC lists {len(names)} schools but found {len(reports)} reports")
    out: dict[str, dict] = {}
    for name, page in zip(names, reports):
        lines = page.splitlines()
        head = " ".join(lines[:8]).upper()
        if name.split()[0].upper().rstrip(".") not in head:
            sys.exit(f"report order mismatch: expected {name!r}")
        header = next(l for l in lines if "Projected" in l)
        # Column centers come from the Actual/Projected header row, so a
        # school with blank early years still lands in the right columns.
        centers = [m.start() + len(m.group()) / 2
                   for m in re.finditer(r"Actual|Projected", header)]
        if len(centers) != len(ENROLL_YEARS):
            sys.exit(f"{name}: expected {len(ENROLL_YEARS)} enrollment columns, got {len(centers)}")
        total = next((l for l in lines if re.match(r"^\s+Total\s+\d", l)), None)
        if total is None:
            sys.exit(f"{name}: no enrollment Total row")
        nums = list(re.finditer(r"\d+", total))
        if len(nums) == len(ENROLL_YEARS):
            # A full row maps in order. Lincoln's Total row is offset from
            # its header, so positions alone would misplace it.
            out[name] = dict(zip(ENROLL_YEARS, (int(m.group()) for m in nums)))
            continue
        values: dict[str, int | None] = dict.fromkeys(ENROLL_YEARS)
        for m in nums:
            mid = (m.start() + m.end()) / 2
            col = min(range(len(centers)), key=lambda i: abs(centers[i] - mid))
            values[ENROLL_YEARS[col]] = int(m.group())
        out[name] = values
    return out


def main() -> None:
    pages = pdf_pages()
    schools = parse_heat_map(pages)
    enrollment = parse_enrollment(pages)
    if len(schools) != EXPECTED_SCHOOLS:
        sys.exit(f"heat map: expected {EXPECTED_SCHOOLS} rows, got {len(schools)}")
    if len(enrollment) != EXPECTED_SCHOOLS:
        sys.exit(f"reports: expected {EXPECTED_SCHOOLS} schools, got {len(enrollment)}")
    for s in schools:
        report = REPORT_NAME.get(s["name"], s["name"])
        if report not in enrollment:
            sys.exit(f"no report page for heat-map row {s['name']!r}")
        s["enrollment"] = enrollment[report]
    OUT.write_text(json.dumps({"source": SOURCE, "schools": schools}, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(schools)} schools")


if __name__ == "__main__":
    main()
