#!/usr/bin/env python3
"""Export PPS enrollment by grade and school (ODE Fall Membership, 4 years)
plus the rightsizing receiving-school table, as plain CSVs for sharing.

Sources:
  data/raw/fall_membership_{2022_23,2023_24,2024_25,2025_26}.xlsx
    ODE Fall Membership Reports, "School" tab, Portland SD 1J rows. Counts
    are students enrolled on the first school day in October. Older files
    print "-" for a grade with no students; 2025-26 prints 0. Both become 0.
  data/pps_schools.csv               rightsizing categories (merge_rightsizing_scenarios.py)
  merge_rightsizing_scenarios.CLOSURE_RECEIVERS   receiving schools (PPS Oct 6 board memo)

Outputs:
  data/exports/pps_enrollment_by_grade_2022-23_to_2025-26.csv
    One row per school per school year: K, grades 1-12, total.
  data/exports/pps_rightsizing_receiving_schools.csv
    One row per closing school -> receiving school flow, per scenario.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

from merge_rightsizing_scenarios import CLOSURE_RECEIVERS, RECEIVERS_SOURCE

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "exports"
MASTER = ROOT / "data" / "pps_schools.csv"
SCENARIOS_JSON = RAW / "pps_rightsizing_scenarios.json"

YEARS = ["2022-23", "2023-24", "2024-25", "2025-26"]
GRADE_WORDS = ["Kindergarten", "Grade One", "Grade Two", "Grade Three", "Grade Four", "Grade Five",
               "Grade Six", "Grade Seven", "Grade Eight", "Grade Nine", "Grade Ten",
               "Grade Eleven", "Grade Twelve"]
GRADE_KEYS = ["k"] + [f"g{i}" for i in range(1, 13)]

# ODE reports Odyssey's students under Hayhurst's institution ID even though
# Odyssey has been in the East Sylvan building since fall 2016.
NOTES = {
    "Hayhurst Elementary School": "ODE counts include the Odyssey Program (K-8), which has been in the East Sylvan building since fall 2016.",
}


def load_year(year: str) -> pd.DataFrame:
    path = RAW / f"fall_membership_{year.replace('-', '_')}.xlsx"
    xl = pd.ExcelFile(path)
    sheet = next(s for s in xl.sheet_names if s.startswith("School"))
    df = xl.parse(sheet)
    df = df[df["District Name"].astype(str).str.strip() == "Portland SD 1J"].copy()
    out = pd.DataFrame({
        "school_year": year,
        "ode_school_id": df["School Institution ID"].astype("Int64"),
        "school_name": df["School Name"].astype(str).str.strip(),
        # Not in every year's file (2022-23 has no School Type column).
        "school_type": df["School Type"] if "School Type" in df else pd.NA,
    })
    for word, key in zip(GRADE_WORDS, GRADE_KEYS):
        col = f"{year} {word}"
        out[key] = pd.to_numeric(df[col].replace("-", 0), errors="raise").astype(int)
    out["total_k12"] = out[GRADE_KEYS].sum(axis=1)
    # The 2024-25 file lists two contracted programs (Providence Pediatric
    # Nursing Facility, Rosemary Anderson) twice under one ID, once with zero
    # students; drop the empty duplicate so each ID appears once per year.
    dup = out.duplicated("ode_school_id", keep=False) & (out["total_k12"] == 0)
    out = out[~dup]
    assert not out["ode_school_id"].duplicated().any(), f"{year}: duplicate school IDs remain"
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = [load_year(y) for y in YEARS]
    allyrs = pd.concat(frames, ignore_index=True)

    # Current (2025-26) name by ODE ID, so renamed schools line up across years.
    current = frames[-1].set_index("ode_school_id")["school_name"]
    allyrs.insert(3, "school_name_2025_26", allyrs["ode_school_id"].map(current))

    master = pd.read_csv(MASTER)
    m = master.set_index("school_name")
    allyrs["level"] = allyrs["school_name_2025_26"].map(m["level"])
    allyrs["scenario_a"] = allyrs["school_name_2025_26"].map(m["rs_a_category"]).fillna("")
    allyrs["scenario_b"] = allyrs["school_name_2025_26"].map(m["rs_b_category"]).fillna("")
    allyrs.loc[allyrs["school_name_2025_26"].isna() | allyrs["level"].isna(), ["scenario_a", "scenario_b"]] = ""
    allyrs["note"] = allyrs["school_name_2025_26"].map(NOTES).fillna("")

    cols = ["school_year", "ode_school_id", "school_name", "school_name_2025_26", "school_type", "level",
            *GRADE_KEYS, "total_k12", "scenario_a", "scenario_b", "note"]
    enroll_path = OUT_DIR / "pps_enrollment_by_grade_2022-23_to_2025-26.csv"
    allyrs.sort_values(["school_name_2025_26", "school_name", "school_year"], na_position="last")[cols] \
        .to_csv(enroll_path, index=False)

    # Sanity: school rows should sum to ODE's own school totals.
    for f, y in zip(frames, YEARS):
        print(f"  {y}: {len(f)} PPS rows, {f['total_k12'].sum():,} students")

    # Receiving schools, per scenario. Scenario B keeps three of A's closures open.
    import json
    closing = {s: set(v["closing"]) for s, v in json.loads(SCENARIOS_JSON.read_text())["scenarios"].items()}
    from merge_rightsizing_scenarios import resolve
    names = set(master["school_name"])
    closing = {s: {resolve(n, names) for n in v} for s, v in closing.items()}
    ode_id = master.set_index("school_name")["ode_school_id"]
    recv_path = OUT_DIR / "pps_rightsizing_receiving_schools.csv"
    with recv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "closing_school", "closing_ode_id", "receiving_school", "receiving_ode_id",
                    "students", "proposed_share", "source"])
        for scen in ("A", "B"):
            for frm, flows in CLOSURE_RECEIVERS.items():
                if frm not in closing[scen]:
                    continue
                for to, share, group in flows:
                    w.writerow([scen, frm.strip(), _id(ode_id.get(frm)), to.strip(), _id(ode_id.get(to)),
                                group, "" if share is None else share, RECEIVERS_SOURCE])
    print(f"wrote {enroll_path}\nwrote {recv_path}")
    return 0


def _id(v):
    return "" if pd.isna(v) else int(v)


if __name__ == "__main__":
    raise SystemExit(main())
