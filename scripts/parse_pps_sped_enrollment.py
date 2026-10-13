#!/usr/bin/env python3
"""Parse PPS's October 2025 Enrollment Characteristics (SpEd / TAG / LEP).

Source PDF (5 pages, one row per school or per program within a school):
  data/raw/pps_sped_tag_lep_2025-26.pdf
  https://resources.finalsite.net/images/v1761768730/ppsnet/urzqafgv7wzzprexxjvw/SchoolProfiles-SPEDTAGLEP25-26.pdf

SpEd is PPS's count of students with an IEP flag in Synergy on October 1,
2025, out of PK-12 enrollment. Schools with more than one program (a
neighborhood program plus an immersion strand) have a row per program and
a "<School> Total" row; single-program schools have one row. Both are kept:
the totals feed the dashboard, the program rows let the rightsizing model
move an immersion strand with its own IEP rate.

Each row's printed percentage is checked against count / enrollment, and
the elementary, middle, and high subtotals against the sum of their school
rows, so a parsing slip fails loudly.

Output:
  data/raw/pps_sped_enrollment_2526.json
    {"source": {...}, "schools": [{"name", "sped", "tag", "lep", "total",
     "programs": [{"program", "sped", "tag", "lep", "total"}, ...]}, ...]}
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "data" / "raw" / "pps_sped_tag_lep_2025-26.pdf"
OUT = ROOT / "data" / "raw" / "pps_sped_enrollment_2526.json"

SOURCE = {
    "title": "Enrollment Characteristics, October 2025 (SpEd / TAG / LEP)",
    "publisher": "Portland Public Schools",
    "as_of": "2025-10-01",
    "url": "https://resources.finalsite.net/images/v1761768730/ppsnet/urzqafgv7wzzprexxjvw/SchoolProfiles-SPEDTAGLEP25-26.pdf",
}

# Subtotal rows PPS prints after each band. Rows before each one are checked
# against it; the bands after high school (focus, community, special services,
# charters) are kept but not checked.
SUBTOTALS = ["Elementary Schools Total", "Middle Schools Total", "High Schools Total"]

# name [program]  sped# sped%  tag# tag%  lep# lep%  female% male% nb%  total
ROW = re.compile(
    r"^(?P<label>\S.*?)\s{2,}(?:(?P<program>\S.*?)\s{2,})?"
    r"(?P<sped>[\d,]+)\s+(?P<sped_pct>\d+)%\s+(?P<tag>[\d,]+)\s+\d+%\s+"
    r"(?P<lep>[\d,]+)\s+[\d.]+%\s+\d+%\s+\d+%\s+\d+%\s+(?P<total>[\d,]+)\s*$")
CONT = re.compile(r"^\s{20,}(?P<program>\S.*?)\s{2,}(?P<sped>[\d,]+)\s+(?P<sped_pct>\d+)%\s+"
                  r"(?P<tag>[\d,]+)\s+\d+%\s+(?P<lep>[\d,]+)\s+[\d.]+%\s+\d+%\s+\d+%\s+\d+%\s+(?P<total>[\d,]+)\s*$")


def num(s: str) -> int:
    return int(s.replace(",", ""))


def counts(m: re.Match) -> dict:
    row = {k: num(m[k]) for k in ("sped", "tag", "lep", "total")}
    # PPS rounds the printed share to a whole percent.
    if row["total"] and abs(100 * row["sped"] / row["total"] - int(m["sped_pct"])) > 0.51:
        sys.exit(f"SpEd % mismatch on line: {m.group(0).strip()}")
    return row


def parse(text: str) -> tuple[list[dict], dict[str, dict]]:
    schools: list[dict] = []
    subtotals: dict[str, dict] = {}
    current: dict | None = None
    for line in text.splitlines():
        if (m := CONT.match(line)) and current is not None:
            current["programs"].append({"program": m["program"], **counts(m)})
            continue
        m = ROW.match(line)
        if not m:
            continue
        label = m["label"].strip()
        if label in SUBTOTALS or label.endswith("Programs Total") or label.endswith("Services Total"):
            subtotals[label] = {**counts(m), "after": len(schools)}
            current = None
            continue
        if label.endswith(" Total") and current is not None and label[:-6] == current["name"]:
            current.update(counts(m))
            continue
        current = {"name": label, "programs": []}
        if m["program"]:
            current["programs"].append({"program": m["program"], **counts(m)})
        else:
            current.update(counts(m))
        schools.append(current)
    return schools, subtotals


def main() -> None:
    text = subprocess.run(["pdftotext", "-layout", str(PDF), "-"],
                          check=True, capture_output=True, text=True).stdout
    schools, subtotals = parse(text)
    for s in schools:
        if "total" not in s:
            sys.exit(f"{s['name']}: program rows but no Total row")
        if s["programs"] and sum(p["sped"] for p in s["programs"]) != s["sped"]:
            sys.exit(f"{s['name']}: program SpEd counts don't add to the school total")
    start = 0
    for label in SUBTOTALS:
        if label not in subtotals:
            sys.exit(f"missing subtotal row {label!r}")
        band = schools[start:subtotals[label]["after"]]
        for key in ("sped", "total"):
            got = sum(s[key] for s in band)
            if got != subtotals[label][key]:
                sys.exit(f"{label}: schools add to {got:,} {key}, PPS prints {subtotals[label][key]:,}")
        start = subtotals[label]["after"]
    OUT.write_text(json.dumps({"source": SOURCE, "schools": schools}, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(schools)} schools")


if __name__ == "__main__":
    main()
