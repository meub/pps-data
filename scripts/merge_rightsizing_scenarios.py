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
(district measures, receiving schools for each closure, and resolved
program-move pairs for the map).
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

# Where each closing school's students go, from PPS's "Rightsizing Update
# Scenario Release" board memo for the 2026-10-06 meeting (pp. 2-4) and the
# matching slides 31-42 (data/raw/pps_rightsizing_board_2026-10-06/). The
# scenario comparison PDF does not name receiving schools, so this is
# transcribed by hand. Master names; share is PPS's proposed split where
# given. Scenario B keeps Lewis, Rose City Park, and Stephenson open and
# uses the same receivers for the other 11 (memo p. 3-4).
CLOSURE_RECEIVERS = {
    "Maplewood Elementary School": [
        ("Hayhurst Elementary School", 0.7, "Neighborhood students"),
        ("Rieke Elementary School", 0.3, "Neighborhood students")],
    "Stephenson Elementary School": [
        ("Markham Elementary School", 0.7, "Neighborhood students"),
        ("Capitol Hill Elementary School", 0.3, "Neighborhood students")],
    "Irvington Elementary School": [
        ("Beverly Cleary School ", None, "K-5 students"),
        ("Beaumont Middle School", None, "Middle school area")],
    "Rose City Park": [
        ("Scott Elementary School", None, "Neighborhood K-5 students"),
        ("Vestal Elementary School", None, "Vietnamese immersion")],
    "Buckman Elementary School": [
        ("Abernethy Elementary School", 0.5, "Neighborhood students"),
        ("Sunnyside Environmental School", 0.5, "Neighborhood students")],
    "Creston Elementary School": [
        ("Atkinson Elementary School", None, "Neighborhood K-5 students"),
        ("Glencoe Elementary School", None, "Deaf and Hard of Hearing program")],
    "Marysville Elementary School": [("Arleta Elementary School", None, "Neighborhood students")],
    "Woodmere Elementary School": [("Whitman Elementary School", None, "Neighborhood students")],
    "Lewis Elementary School": [("Duniway Elementary School", None, "Neighborhood students")],
    "Sellwood Middle School": [
        ("Hosford Middle School", None, "Llewellyn-area students"),
        ("Lane Middle School", None, "Duniway- and Lewis-area students")],
    "Beach Elementary School": [
        ("Chief Joseph Elementary School", None, "Neighborhood K-5 students"),
        ("César Chávez K-8 School", None, "Spanish immersion")],
    "James John Elementary School": [
        ("Sitton Elementary School", None, "Neighborhood K-5 students"),
        ("César Chávez K-8 School", None, "Spanish immersion")],
    "Peninsula Elementary School": [("Rosa Parks Elementary School", None, "Neighborhood students")],
    "Sabin Elementary School": [("Dr. Martin Luther King Jr. School", None, "Neighborhood students")],
}
RECEIVERS_SOURCE = ("PPS board memo, Rightsizing Update: Scenario Release, "
                    "October 6, 2026 board meeting")


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
           "district": raw["district"], "receivers_source": RECEIVERS_SOURCE, "scenarios": {}}
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
        closing = [resolve(n, master_names) for n in s["closing"]]
        missing = [n for n in closing if n not in CLOSURE_RECEIVERS]
        if missing:
            raise KeyError(f"Scenario {scen}: no receiving schools recorded for {missing}")
        receivers = []
        for n in closing:
            for to, share, group in CLOSURE_RECEIVERS[n]:
                if to not in master_names:
                    raise KeyError(f"receiving school {to!r} not in master")
                receivers.append({"from": n, "to": to, "share": share, "group": group})
        out["scenarios"][scen] = {
            "closing": closing,
            "receivers": receivers,
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
