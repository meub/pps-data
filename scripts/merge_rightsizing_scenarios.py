#!/usr/bin/env python3
"""Merge PPS's rightsizing scenarios (A and B, Oct 2026) into the master CSV.

Reads data/raw/pps_rightsizing_scenarios.json (parse_rightsizing_scenarios.py)
and maps PPS's short school names to master names via PPS_NAME_MAP plus a
suffix rule. Every name must resolve; an unknown name fails the run so a
revised PDF cannot silently drop a school.

Adds to data/pps_schools.csv, for each scenario s in {a, b}:
  rs_{s}_category  closes | program_change | boundary | (blank = no change)
                   One per school, highest of: closes > program or grade
                   change (moving out or receiving) > boundary change.
  rs_{s}_detail    Every change touching the school, PPS's wording, "; "-joined.

export_web.py also imports load_resolved() to emit the scenario payload
(district measures plus resolved program-move pairs for the map).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "data/pps_schools.csv"
SCENARIOS = ROOT / "data/raw/pps_rightsizing_scenarios.json"

# PPS short name -> master school_name, for names the suffix rule in
# resolve() gets wrong. Brentwood MS is Lane Middle School under the name
# the board approved in 2026; Robert Gray MS is Gray Middle School.
PPS_NAME_MAP = {
    "Beverly Cleary": "Beverly Cleary School ",  # master name has a trailing space
    "Boise-Eliot/Humboldt": "Boise-Eliot Elementary School",
    "Brentwood MS": "Lane Middle School",
    "Bridger Creative Science": "Bridger Creative Science School",
    "Cesar Chavez": "César Chávez K-8 School",
    "Chavez": "César Chávez K-8 School",
    "Chavez 6–8": "César Chávez K-8 School",
    "Dr. Martin Luther King Jr.": "Dr. Martin Luther King Jr. School",
    "George": "George Middle School",
    "Harrison Park MS": "Harrison Park School",
    "McDaniel": "Leodis V. McDaniel High School",
    "MLC": "Metropolitan Learning Center",
    "MLK Jr.": "Dr. Martin Luther King Jr. School",
    "Mt. Tabor MS": "Mt Tabor Middle School",
    "Odyssey": "Odyssey Program (K-8)",
    "Robert Gray MS": "Gray Middle School",
    "Rose City Park": "Rose City Park",
    "Roseway Heights MS": "Roseway Heights School",
    "Sunnyside Environmental": "Sunnyside Environmental School",
}

CATEGORY_RANK = {"closes": 3, "program_change": 2, "boundary": 1}


def resolve(name: str, master_names: set[str]) -> str:
    name = name.strip()
    if name in PPS_NAME_MAP:
        return PPS_NAME_MAP[name]
    if name.endswith(" MS"):
        cand = name[:-3] + " Middle School"
    elif name.endswith(" HS"):
        cand = name[:-3] + " High School"
    else:
        cand = name + " Elementary School"
    if cand not in master_names:
        raise KeyError(f"PPS name {name!r} does not resolve (tried {cand!r}); add it to PPS_NAME_MAP")
    return cand


def load_resolved(master_names: set[str]) -> dict:
    """Scenario JSON with every school name resolved to its master name."""
    raw = json.loads(SCENARIOS.read_text())
    out = {"source": raw["source"], "thresholds": raw["thresholds"],
           "district": raw["district"], "scenarios": {}}
    for scen, s in raw["scenarios"].items():
        moves = []
        for mv in s["program_moves"]:
            for src in mv["from"]:
                moves.append({"kind": "program", "text": mv["text"], "program": mv["program"],
                              "from": resolve(src, master_names), "to": resolve(mv["to"], master_names)})
        for gc in s["grade_changes"]:
            moves.append({"kind": "grade", "text": gc["text"], "program": None,
                          "from": resolve(gc["from"], master_names),
                          "to": resolve(gc["to"], master_names) if gc["to"] else None})
        out["scenarios"][scen] = {
            "closing": [resolve(n, master_names) for n in s["closing"]],
            "boundary_changes": [resolve(n, master_names) for n in s["boundary_changes"]],
            "moves": moves,
            "notes": s.get("notes", []),
            "counts": {k: s[f"{k}_count"] for k in
                       ("closing", "boundary_changes", "program_moves", "grade_changes")},
        }
    return out


def school_changes(scen: dict) -> dict[str, list[tuple[str, str]]]:
    """master name -> [(category, description)] for one scenario."""
    ch: dict[str, list[tuple[str, str]]] = {}
    for n in scen["closing"]:
        ch.setdefault(n, []).append(("closes", "Closes"))
    for mv in scen["moves"]:
        label = "Program move" if mv["kind"] == "program" else "Grade change"
        for n in {mv["from"], mv["to"]} - {None}:
            entry = ("program_change", f"{label}: {mv['text']}")
            if entry not in ch.setdefault(n, []):
                ch[n].append(entry)
    for note in scen["notes"]:
        if note.startswith("The MLC"):
            ch.setdefault("Metropolitan Learning Center", []).append(("program_change", note))
    for n in scen["boundary_changes"]:
        ch.setdefault(n, []).append(("boundary", "Attendance boundary changes"))
    return ch


def main() -> int:
    master = pd.read_csv(MASTER)
    master = master.drop(columns=[c for c in master.columns if c.startswith("rs_")])
    names = set(master["school_name"])
    resolved = load_resolved(names)

    for scen in ("A", "B"):
        ch = school_changes(resolved["scenarios"][scen])
        key = scen.lower()
        master[f"rs_{key}_category"] = master["school_name"].map(
            lambda n: max((c for c, _ in ch.get(n, [])), key=CATEGORY_RANK.get, default=None))
        master[f"rs_{key}_detail"] = master["school_name"].map(
            lambda n: "; ".join(d for _, d in ch.get(n, [])) or None)
        counts = master[f"rs_{key}_category"].value_counts().to_dict()
        print(f"Scenario {scen}: {counts}")

    master.to_csv(MASTER, index=False)
    print(f"wrote {MASTER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
