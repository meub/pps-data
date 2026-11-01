#!/usr/bin/env python3
"""Parse PPS's October 1, 2026 enrollment summary (by grade and program).

Source:
  Portland Public Schools, "Enrollment by Grade and Program Type", 2026-27,
  October 1st Enrollment (Data and Reporting page, pps.net/fs/pages/46481,
  "2026-27 Academic Year Datasets"). Five pages, created 2026-10-27. Counts
  come from the Period 1 Cumulative ADM state report and exclude students
  receiving only special education services who are not regularly attending.

PPS posts this about four weeks after October 1; ODE's Fall Membership file
for the same date follows around February. Until then this is the only
per-school count for the year. It lists every school's grades PK-12 and,
for immersion schools, one row per strand (Spanish Immersion, Neighborhood
Program, ...), which ODE does not publish.

Words are read with `pdftotext -bbox` and assigned to the grade column whose
header is nearest, because empty grades are blank cells. The parser fails
unless every row's grades add to its printed total, strands add to their
school total, schools add to each section total, and sections add to the
district total.

Output:
  data/raw/pps_october_enrollment_2026-27.json
"""
from __future__ import annotations

import html
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = "https://www.pps.net/fs/resource-manager/view/88b9f6d2-92a2-469d-b6d3-b62323a9a67b"
PDF = ROOT / "data" / "raw" / "pps_october_enrollment_2026-27.pdf"
OUT = ROOT / "data" / "raw" / "pps_october_enrollment_2026-27.json"
SCHOOL_YEAR = "2026-27"

HEADERS = ["PK", "KG", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "Total"]
KEYS = ["pk", "k"] + [f"g{i}" for i in range(1, 13)]
SECTION_TOTALS = {
    "Elementary Schools Total": "Elementary",
    "Middle School Total": "Middle",
    "High School Total": "High",
    "Alt. Programs Total": "Alternative",
    "CBOs Total": "Community-based",
    "Special Services Total": "Special services",
    "Public Charter Schools Total": "Charter",
}
# Benson is printed between the middle-school total and the high-school page,
# so it belongs to the next section like every other row.
WORD_RE = re.compile(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="[\d.]+">(.*?)</word>')
PAGE_RE = re.compile(r"<page .*?</page>", re.S)


def fetch() -> None:
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    if not data.startswith(b"%PDF"):
        sys.exit(f"{URL} did not return a PDF")
    PDF.write_bytes(data)


def page_rows(page: str) -> list[dict]:
    words = [(float(a), float(b), float(c), html.unescape(t)) for a, b, c, t in WORD_RE.findall(page)]
    header_y = next(y for x0, y, x1, t in words if t == "PK")
    cols = {t: (x0 + x1) / 2 for x0, y, x1, t in words if abs(y - header_y) < 1 and t in HEADERS}
    assert sorted(cols) == sorted(HEADERS), f"page header columns: {list(cols)}"
    program_x = next(x0 for x0, y, x1, t in words if t == "School" and abs(y - header_y) < 1)
    first_col = next(x0 for x0, y, x1, t in words if t == "PK" and abs(y - header_y) < 1)

    lines: dict[float, list] = {}
    for w in words:
        if w[1] <= header_y + 1:
            continue
        key = next((k for k in lines if abs(k - w[1]) < 2), w[1])
        lines.setdefault(key, []).append(w)

    rows = []
    for y in sorted(lines):
        ws = sorted(lines[y])
        name = " ".join(t for x0, _, x1, t in ws if x1 < program_x - 2)
        program = " ".join(t for x0, _, x1, t in ws if x0 >= program_x - 2 and x1 < first_col - 2)
        nums = [(x0, x1, t) for x0, _, x1, t in ws if x1 >= first_col - 2]
        if not nums or not all(re.fullmatch(r"[\d,]+", t) for _, _, t in nums):
            continue  # footnote text runs across the grade columns
        cells: dict[str, int] = {}
        for x0, x1, t in nums:
            col = min(cols, key=lambda c: abs(cols[c] - (x0 + x1) / 2))
            assert col not in cells, f"two numbers in column {col}: {name} {program}"
            cells[col] = int(t.replace(",", ""))
        grades = {k: cells.get(h, 0) for k, h in zip(KEYS, HEADERS)}
        total = cells["Total"]
        assert sum(grades.values()) == total, f"{name} {program}: grades {sum(grades.values())} != {total}"
        rows.append({"name": name, "program": program, "grades": grades, "total": total})
    return rows


def parse() -> dict:
    text = subprocess.run(["pdftotext", "-bbox", str(PDF), "-"], capture_output=True, text=True, check=True).stdout
    rows = [r for page in PAGE_RE.findall(text) for r in page_rows(page)]

    schools: list[dict] = []
    sections: dict[str, dict] = {}
    pending: list[dict] = []  # schools since the last section total
    current = None  # multi-strand school waiting for its "X Total" row
    district = None
    for r in rows:
        name, program = r["name"], r["program"]
        if name == "District Total":
            district = r
        elif name in SECTION_TOTALS:
            section = SECTION_TOTALS[name]
            for k in KEYS + ["total"]:
                got = sum((s["grades"][k] if k != "total" else s["total"]) for s in pending)
                want = r["grades"][k] if k != "total" else r["total"]
                assert got == want, f"{name} {k}: schools add to {got}, PPS prints {want}"
            for s in pending:
                s["section"] = section
            sections[section] = {"grades": r["grades"], "total": r["total"]}
            schools += pending
            pending, current = [], None
        elif name.endswith(" Total"):
            assert current and current["name"] == name[:-len(" Total")], f"unexpected total row {name}"
            for k in KEYS:
                got = sum(p["grades"][k] for p in current["programs"])
                assert got == r["grades"][k], f"{name} {k}: strands add to {got}, PPS prints {r['grades'][k]}"
            current["grades"], current["total"] = r["grades"], r["total"]
            current = None
        elif program:
            if name:
                current = {"name": name, "programs": []}
                pending.append(current)
            assert current, f"strand row with no school: {program}"
            current["programs"].append({"program": program, "grades": r["grades"], "total": r["total"]})
        else:
            assert name, f"row with no name: {r}"
            current = None
            pending.append({"name": name, "programs": [], "grades": r["grades"], "total": r["total"]})

    assert not pending, f"rows after the last section total: {[s['name'] for s in pending]}"
    assert district, "no District Total row"
    for k in KEYS + ["total"]:
        got = sum((s["grades"][k] if k != "total" else s["total"]) for s in sections.values())
        want = district["grades"][k] if k != "total" else district["total"]
        assert got == want, f"District Total {k}: sections add to {got}, PPS prints {want}"
    assert all("total" in s for s in schools), "a multi-strand school had no total row"

    return {
        "source": URL,
        "school_year": SCHOOL_YEAR,
        "count_date": "2026-10-01",
        "district_total": district["total"],
        "district_total_k12": district["total"] - district["grades"]["pk"],
        "district_grades": district["grades"],
        "sections": sections,
        "schools": schools,
    }


def main() -> int:
    if not PDF.exists():
        fetch()
    data = parse()
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(data['schools'])} schools, district {data['district_total']:,} "
          f"(K-12 {data['district_total_k12']:,}) -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
