#!/usr/bin/env python3
"""Parse PPS's rightsizing scenario comparison PDF into JSON.

Source PDF (two pages, "Updated October 4, 2026"):
  data/raw/pps_rightsizing_scenarios_2026-10-04.pdf

Page 1: district-wide measures for Scenario A, Scenario B, and Status Quo
(projections for 2031-32) plus the sustainability thresholds.
Page 2: every affected school, per scenario, in four groups: schools
closing, attendance boundary changes, program moves, grade-level changes.

PPS prints a count beside each group header. The parser checks its own
counts against those numbers and fails on a mismatch, so a layout change
in a revised PDF cannot silently drop schools. PPS's school names are
kept verbatim; mapping to master-CSV names happens in
merge_rightsizing_scenarios.py.

Output:
  data/raw/pps_rightsizing_scenarios.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "data" / "raw" / "pps_rightsizing_scenarios_2026-10-04.pdf"
OUT = ROOT / "data" / "raw" / "pps_rightsizing_scenarios.json"

SOURCE = {
    "title": "Rightsizing: Comparing three scenarios",
    "publisher": "Portland Public Schools",
    "updated": "2026-10-04",
    "released": "2026-10-05",
    "projection_year": "2031-32",
    "url": "https://drive.google.com/file/d/1ueJVC-Jo6MRKw56Jedihc6YbVdBlSB2z/view",
}

# Page 1 district table. Labels as printed; the label column ends before
# x=800pt and the three value columns sit to its right in this order.
MEASURES = [
    ("closures", "Schools or sites that would close"),
    ("students_change_schools", "Students who would change schools"),
    ("schools_above_threshold", "Schools above the sustainability threshold"),
    ("students_above_threshold", "Students attending a school above the threshold"),
    ("poverty_in_sustainable", "Students Experiencing Poverty in Sustainable Schools"),
    ("underserved_in_sustainable", "Historically Underserved Students in Sustainable Schools"),
    ("sped_in_sustainable", "Students Receiving Special Education Services in Sustainable Schools"),
    ("multilingual_in_sustainable", "Multilingual Learners in Sustainable Schools"),
]
LABEL_X_MAX = 800
SCENARIO_KEYS = ["A", "B", "status_quo"]

# Page 2 group headers -> output keys. Left column is Scenario A, right is B.
GROUPS = {
    "SCHOOLS CLOSING": "closing",
    "ATTENDANCE BOUNDARY CHANGES": "boundary_changes",
    "PROGRAM MOVES": "program_moves",
    "GRADE-LEVEL CHANGES": "grade_changes",
}
PAGE2_SPLIT_X = 612  # page midpoint between the two scenario columns


def parse_value(tok: str):
    tok = tok.strip()
    if tok.endswith("%"):
        return float(tok[:-1]) / 100
    return int(tok)


def parse_district(page) -> dict:
    """Return {measure_key: {A, B, status_quo}} from the page-1 table.

    Long labels wrap onto two lines with the values on the middle line, so
    rows are rebuilt by matching the known label words in reading order
    and taking the value row nearest that label vertically.
    """
    words = page.extract_words()
    value_rows: dict[int, list] = {}
    for w in words:
        if w["x0"] >= LABEL_X_MAX and re.fullmatch(r"\d+%?", w["text"]):
            value_rows.setdefault(round(w["top"]), []).append(w)
    rows = [sorted(ws, key=lambda w: w["x0"]) for _, ws in sorted(value_rows.items())]
    rows = [r for r in rows if len(r) == 3]

    label_words = [w for w in words if w["x0"] < LABEL_X_MAX]
    out = {}
    for key, label in MEASURES:
        first = label.split()[0]
        # The label's first word, below the table header, locates the measure.
        cands = [w for w in label_words if w["text"] == first and w["top"] > 330]
        if not cands:
            raise ValueError(f"label not found on page 1: {label!r}")
        anchor = min(cands, key=lambda w: min(abs(r[0]["top"] - w["top"]) for r in rows))
        row = min(rows, key=lambda r: abs(r[0]["top"] - anchor["top"]))
        if abs(row[0]["top"] - anchor["top"]) > 20:
            raise ValueError(f"no value row near label {label!r}")
        out[key] = {"label": label, **{k: parse_value(w["text"]) for k, w in zip(SCENARIO_KEYS, row)}}
        rows.remove(row)
    return out


def parse_thresholds(text: str) -> dict:
    # Printed as one row of labels ("K–5:  K–8 and middle school:  High
    # school:") above one row of values, so match the labels, then the values.
    m = re.search(r"K–5:\s*K–8 and middle school:\s*High school:.*?"
                  r"(\d[\d,]*) students\s+(\d[\d,]*) students\s+(\d[\d,]*) students", text, re.S)
    if not m:
        raise ValueError("sustainability thresholds not found")
    k5, k8ms, hs = (int(v.replace(",", "")) for v in m.groups())
    return {"k5": k5, "k8_and_middle": k8ms, "high": hs}


def split_items(body: str) -> list[str]:
    return [i.strip() for i in re.split(r"\s+·\s+|\s·|·\s", body) if i.strip()]


def parse_program_move(item: str) -> dict:
    """'Spanish: Beach, James John, and Sitton to Cesar Chavez' ->
    {text, program, from: [...], to}. 'Odyssey moves to the MLC site' is
    the one phrasing without a plain 'X to Y'."""
    program = None
    m = re.match(r"^(Spanish|Mandarin|Vietnamese|Japanese|Russian):\s*(.*)$", item)
    rest = item
    if m:
        program, rest = m.group(1), m.group(2)
    m = re.match(r"^(.*?)\s+(?:moves\s+)?to\s+(?:the\s+)?(.+?)$", rest)
    if not m:
        raise ValueError(f"unparsed program move: {item!r}")
    src, dst = m.group(1), m.group(2)
    if src.startswith("Creston Deaf and Hard of Hearing"):
        program, src = "Deaf and Hard of Hearing", "Creston"
    sources = [s.strip() for s in re.split(r",\s*(?:and\s+)?|\s+and\s+", src) if s.strip()]
    return {"text": item, "program": program, "from": sources, "to": dst.removesuffix(" site")}


def parse_grade_change(item: str) -> dict:
    m = re.match(r"^(.*?)\s+(\d[–-]\d)\s+to\s+(.+)$", item)
    if m:
        return {"text": item, "from": m.group(1), "grades": m.group(2).replace("–", "-"), "to": m.group(3)}
    m = re.match(r"^(.*?)\s+becomes\s+(.+)$", item)
    if m:
        return {"text": item, "from": m.group(1), "grades": None, "to": None, "becomes": m.group(2)}
    raise ValueError(f"unparsed grade change: {item!r}")


def parse_page2(page) -> dict:
    scenarios = {"A": {}, "B": {}}
    for table in page.find_tables():
        cell = (table.extract()[0][-1] or "").strip()
        header, _, body = cell.partition("\n")
        m = re.match(r"^([A-Z\- ]+?)\s+(\d+)$", header.strip())
        if not m or m.group(1) not in GROUPS:
            continue
        group, printed = GROUPS[m.group(1)], int(m.group(2))
        scen = "A" if table.bbox[0] < PAGE2_SPLIT_X else "B"
        body = " ".join(body.split())
        notes = []
        if group == "closing":
            note_at = body.find("The MLC")
            if note_at >= 0:
                notes.append(body[note_at:])
                body = body[:note_at]
        items = split_items(body)
        if group == "program_moves":
            moves = [parse_program_move(i) for i in items]
            counted = sum(len(mv["from"]) for mv in moves)
            value = moves
        elif group == "grade_changes":
            value = [parse_grade_change(i) for i in items]
            counted = len(value)
        else:
            value = items
            counted = len(items)
        if counted != printed:
            raise ValueError(f"Scenario {scen} {group}: parsed {counted}, PDF says {printed}: {items}")
        scenarios[scen][group] = value
        scenarios[scen][f"{group}_count"] = printed
        if notes:
            scenarios[scen]["notes"] = notes
    for scen, groups in scenarios.items():
        missing = [g for g in GROUPS.values() if g not in groups]
        if missing:
            raise ValueError(f"Scenario {scen}: groups not found: {missing}")
    return scenarios


def main() -> int:
    if not PDF.exists():
        print(f"ERROR: {PDF} missing; run fetch_rightsizing_scenarios.py first")
        return 1
    with pdfplumber.open(PDF) as pdf:
        p1, p2 = pdf.pages[0], pdf.pages[1]
        district = parse_district(p1)
        thresholds = parse_thresholds(p1.extract_text())
        scenarios = parse_page2(p2)

    for scen in ("A", "B"):
        if district["closures"][scen] != scenarios[scen]["closing_count"]:
            raise ValueError(f"Scenario {scen}: page 1 and page 2 closure counts differ")

    payload = {"source": SOURCE, "thresholds": thresholds, "district": district, "scenarios": scenarios}
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}")
    for scen in ("A", "B"):
        s = scenarios[scen]
        print(f"  Scenario {scen}: {s['closing_count']} close, {s['boundary_changes_count']} boundary, "
              f"{s['program_moves_count']} program moves, {s['grade_changes_count']} grade changes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
