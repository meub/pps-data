#!/usr/bin/env python3
"""Rough before/after enrollment model for PPS's rightsizing scenarios.

Holds fall 2025 enrollment fixed and moves students the way each scenario
says, to show how big each receiving school gets by grade and strand. The
dashboard turns these counts into estimated K-5 classes with a
user-adjustable class-size target, and compares totals with each building's
functional capacity.

Inputs:
  ODE Fall Membership 2025-26 grade counts (export_enrollment_by_grade.load_year)
  PPS immersion report Oct 2025, students per language (data/raw/pps_immersion_details_2526.json)
  Resolved scenarios with receiving schools (merge_rightsizing_scenarios.load_resolved)

What moves, per scenario:
  - Each closing school's students go to its receiving schools (PPS board memo).
    Immersion groups take that language's strand; neighborhood groups take
    the rest. Two receivers with no published split share students equally
    (only Sellwood MS: Hosford and Lane).
  - Grade-level changes ("X 6-8 to Y") move all of X's grades 6-8.
  - Program moves from schools that stay open move that school's immersion
    strand (its only language, unless PPS names one).
  - EXTRA_FLOWS: assignments in the memo but not in the scenario PDF.
  - Students with IEPs move with their group: each strand carries its
    school's October 2025 IEP rate (PPS SpEd report; immersion and
    neighborhood programs use their own rates where PPS prints them).

Assumptions and gaps (also shown on the dashboard):
  - A strand's students are spread across grades in proportion to the
    school's grade counts; PPS publishes strand totals, not strands by grade.
  - Attendance boundary changes are not modeled; PPS has not published how
    many students they move.
  - Program sizes PPS does not publish are not moved: Creston's Deaf and
    Hard of Hearing program, and Odyssey (no grade counts; ODE reports it
    under Hayhurst).
  - Focus classrooms (SES, CB, ISC) are not moved separately: students in a
    closing school's focus rooms are assumed to go where its other students go.
  - Hayhurst's ODE counts include Odyssey; they are scaled to PPS's
    Hayhurst-only total.
"""
from __future__ import annotations

import json
from pathlib import Path

from build_master import DLI_NAME_MAP, HAYHURST_ODYSSEY_SPLIT, pps_master_name
from export_enrollment_by_grade import GRADE_KEYS, load_year

ROOT = Path(__file__).resolve().parent.parent
DLI_JSON = ROOT / "data" / "raw" / "pps_immersion_details_2526.json"
SPED_JSON = ROOT / "data" / "raw" / "pps_sped_enrollment_2526.json"

GRADES = GRADE_KEYS[:9]  # K-8; the scenarios change only elementary and middle grades
K5 = GRADES[:6]
G68 = GRADES[6:9]
NEIGHBORHOOD = "Neighborhood"

# In PPS's Oct 6 board memo (p. 3) but not in the scenario comparison PDF.
EXTRA_FLOWS = [
    {"from": "César Chávez K-8 School", "to": "Rosa Parks Elementary School",
     "strand": NEIGHBORHOOD, "grades": K5, "share": 1.0,
     "text": "César Chávez non-immersion K-5 students to Rosa Parks (board memo)"},
]

# Closure groups whose size PPS does not publish; left in place, noted on the page.
UNMODELED_GROUPS = {"Deaf and Hard of Hearing program"}


def base_strands(master_names: set[str]) -> dict[str, dict[str, list[float]]]:
    """school -> strand -> [K..8 counts] for fall 2025."""
    grades = load_year("2025-26").set_index("school_name")
    dli = {DLI_NAME_MAP.get(r["name"], r["name"]).strip(): r for r in json.loads(DLI_JSON.read_text())}
    # ODE names are stripped; a few master names keep a trailing space
    # ("Beverly Cleary School "), so key the output by the master spelling.
    by_stripped = {n.strip(): n for n in master_names}
    out = {}
    for ode_name, row in grades.iterrows():
        name = by_stripped.get(ode_name)
        if name is None:
            continue
        counts = [float(row[g]) for g in GRADES]
        if name == "Hayhurst Elementary School":
            # ODE counts include Odyssey; scale K-5 to PPS's Hayhurst-only total.
            k5 = sum(counts[:6])
            target = HAYHURST_ODYSSEY_SPLIT["2025-26 Total Enrollment"][0]
            counts = [c * target / k5 if i < 6 else 0.0 for i, c in enumerate(counts)]
        total = sum(counts)
        if total == 0:
            continue
        strands = {}
        rep = dli.get(name.strip())
        if rep and rep.get("dli_by_language") and rep["total_enrollment"]:
            for lang, n in rep["dli_by_language"].items():
                share = n / rep["total_enrollment"]
                strands[lang] = [c * share for c in counts]
        used = [sum(v[i] for v in strands.values()) for i in range(len(GRADES))]
        strands[NEIGHBORHOOD] = [max(0.0, c - u) for c, u in zip(counts, used)]
        out[name] = strands
    return out


def iep_rates(master_names: set[str]) -> dict[str, dict[str, float]]:
    """school -> strand -> share of students with IEPs, October 2025.

    "*" is the whole-school rate, used for any strand PPS doesn't print a
    program row for.
    """
    names = sorted(master_names)
    out = {}
    for r in json.loads(SPED_JSON.read_text())["schools"]:
        name = pps_master_name(r["name"], names)
        if name is None or not r["total"]:
            continue
        rates = {"*": r["sped"] / r["total"]}
        for p in r["programs"]:
            strand = NEIGHBORHOOD if p["program"] == "Neighborhood Program" else p["program"].replace(" Immersion", "")
            if p["total"]:
                rates[strand] = p["sped"] / p["total"]
        out[name] = rates
    return out


def _strand_for(group: str, src: dict) -> list[str] | None:
    """Which of the source school's strands a closure group takes."""
    for lang in ("Spanish", "Mandarin", "Vietnamese", "Japanese", "Russian"):
        if lang.lower() in group.lower():
            return [lang] if lang in src else []
    return None  # neighborhood group


def _grades_for(group: str) -> list[str]:
    g = group.lower()
    if "k-5" in g:
        return K5
    if "middle school area" in g:
        return G68
    return GRADES


def build(resolved: dict, master_names: set[str], direct_cert: dict[str, float] | None = None) -> dict:
    """direct_cert: school -> share of students directly certified, moved
    like IEPs (one rate per school) so merged schools get a blended share."""
    base = base_strands(master_names)
    rates = iep_rates(master_names)
    rate = lambda n, strand: rates.get(n, {}).get(strand, rates.get(n, {}).get("*", 0.0))
    dc = direct_cert or {}
    # Students carrying each measure, by school and strand: a rate times the head count.
    measures = {
        "iep": {n: {k: [c * rate(n, k) for c in v] for k, v in st.items()} for n, st in base.items()},
        "dc": {n: {k: [c * dc.get(n, 0.0) for c in v] for k, v in st.items()} for n, st in base.items()},
    }
    result = {"assumptions": {
        "grade_distribution": "Strand totals spread across grades in proportion to each school's grade counts.",
        "not_modeled": ["Attendance boundary changes", "Creston Deaf and Hard of Hearing program (size not published)",
                        "Odyssey's move to MLC (no grade counts)"],
    }, "scenarios": {}}
    for scen, s in resolved["scenarios"].items():
        closing = set(s["closing"])
        flows = []
        # Closures. A school with an immersion flow sends only its other strands
        # with the neighborhood group.
        by_src: dict[str, list] = {}
        for r in s["receivers"]:
            by_src.setdefault(r["from"], []).append(r)
        for src, rs in by_src.items():
            strands = base.get(src, {})
            lang_taken = {l for r in rs for l in (_strand_for(r["group"], strands) or [])}
            hood = [k for k in strands if k not in lang_taken]
            unsplit = [r for r in rs if r["share"] is None and _strand_for(r["group"], strands) is None
                       and _grades_for(r["group"]) is GRADES]
            for r in rs:
                if r["group"] in UNMODELED_GROUPS:
                    continue
                langs = _strand_for(r["group"], strands)
                share = r["share"] if r["share"] is not None else (1 / len(unsplit) if r in unsplit else 1.0)
                for strand in (langs if langs is not None else hood):
                    flows.append({"from": src, "to": r["to"], "strand": strand,
                                  "grades": _grades_for(r["group"]), "share": share,
                                  "text": f"{src.strip()}: {r['group']}"})
        # Grade-level changes and program moves from schools that stay open.
        grade_moved = set()
        for mv in s["moves"]:
            if mv["kind"] == "grade" and mv["to"]:
                grade_moved.add(mv["from"])
                for strand in base.get(mv["from"], {}):
                    flows.append({"from": mv["from"], "to": mv["to"], "strand": strand, "grades": G68,
                                  "share": 1.0, "text": mv["text"]})
        for mv in s["moves"]:
            if mv["kind"] != "program" or mv["from"] in closing or mv["from"] in grade_moved:
                continue
            strands = base.get(mv["from"], {})
            langs = [mv["program"]] if mv["program"] in strands else [k for k in strands if k != NEIGHBORHOOD]
            for strand in langs:
                flows.append({"from": mv["from"], "to": mv["to"], "strand": strand, "grades": GRADES,
                              "share": 1.0, "text": mv["text"]})
        flows += EXTRA_FLOWS

        after = {n: {k: list(v) for k, v in st.items()} for n, st in base.items()}
        m_after = {m: {n: sum(sum(v) for v in st.values()) for n, st in mb.items()} for m, mb in measures.items()}
        for f in flows:
            src = base.get(f["from"], {}).get(f["strand"])
            if src is None or f["to"] not in after:
                continue
            for g in f["grades"]:
                i = GRADES.index(g)
                moved = src[i] * f["share"]
                after[f["from"]][f["strand"]][i] -= moved
                after[f["to"]].setdefault(f["strand"], [0.0] * len(GRADES))[i] += moved
                for m, mb in measures.items():
                    carried = mb[f["from"]][f["strand"]][i] * f["share"]
                    m_after[m][f["from"]] -= carried
                    m_after[m][f["to"]] += carried

        # Students are only moved, never created or lost.
        before_total = sum(sum(sum(v) for v in st.values()) for st in base.values())
        after_total = sum(sum(sum(v) for v in st.values()) for st in after.values())
        assert abs(before_total - after_total) < 0.5, f"Scenario {scen}: model lost students"
        for m, mb in measures.items():
            total = sum(sum(sum(v) for v in st.values()) for st in mb.values())
            assert abs(total - sum(m_after[m].values())) < 0.5, f"Scenario {scen}: model lost {m} students"
        dropped = [f for f in flows if f["from"] in base and f["to"] not in after]
        assert not dropped, f"Scenario {scen}: flows to unknown schools: {[f['to'] for f in dropped]}"

        touched = {f["from"] for f in flows} | {f["to"] for f in flows}
        rows = {}
        for n in sorted(touched):
            if n not in base:
                continue
            rnd = lambda st: {k: [round(x, 1) for x in v] for k, v in st.items() if sum(v) > 0.05}
            rows[n] = {"before": rnd(base[n]), "after": {} if n in closing else rnd(after[n]),
                       **{f"{m}_before": round(sum(sum(v) for v in mb[n].values()), 1) for m, mb in measures.items()},
                       **{f"{m}_after": 0.0 if n in closing else round(m_after[m][n], 1) for m in measures},
                       "closes": n in closing,
                       "inflows": sorted({f["text"] for f in flows if f["to"] == n})}
        result["scenarios"][scen] = rows
    result["grades"] = GRADES
    return result


if __name__ == "__main__":
    import pandas as pd
    from merge_rightsizing_scenarios import load_resolved
    master = pd.read_csv(ROOT / "data" / "pps_schools.csv")
    names = set(master["school_name"])
    m = build(load_resolved(names), names)
    cap = master.set_index("school_name")["functional_capacity_2021"]
    for scen in ("A", "B"):
        print(f"Scenario {scen}")
        for n, r in m["scenarios"][scen].items():
            b = sum(sum(v) for v in r["before"].values()); a = sum(sum(v) for v in r["after"].values())
            if a > b + 0.5:
                c = cap.get(n)
                print(f"  {n.strip():38s} {b:6.0f} -> {a:6.0f}   cap {c:>6}  util {'' if pd.isna(c) else f'{b/c:.0%} -> {a/c:.0%}'}")
