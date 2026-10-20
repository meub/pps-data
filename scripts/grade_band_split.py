#!/usr/bin/env python3
"""Split K-12 schools that ODE and PPS report as one school into their K-8
and 9-12 parts, for the two measures that can be split.

Metropolitan Learning Center is one ODE school (ID 916) but two programs:
a K-8 and a 9-12 high school that both PPS scenarios close. Every other
source (demographics, special education, staffing, budget, capacity, PRC
forecast) reports MLC as a whole.

Sources:
  data/raw/fall_membership_{year}.xlsx   ODE Fall Membership, counts by grade
  data/raw/pagr_schools_{ela,math}_tot_gnd_2425.xlsx
    ODE OSAS 2024-25 by student group and grade. The "all" files we use
    elsewhere only carry All Grades; these carry grades 3-8 and HS (11).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from export_enrollment_by_grade import GRADE_KEYS, YEARS, load_year

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
SPLIT_SCHOOLS = {"Metropolitan Learning Center": 916}  # master name -> ODE ID
K8, HS = GRADE_KEYS[:9], GRADE_KEYS[9:]
OSAS_FILES = {"ela": RAW / "pagr_schools_ela_tot_gnd_2425.xlsx",
              "math": RAW / "pagr_schools_math_tot_gnd_2425.xlsx"}
OSAS_ALL = {"ela": RAW / "pagr_schools_ela_all_2425.xlsx",
            "math": RAW / "pagr_schools_math_all_2425.xlsx"}
TOTAL = "Total Population (All Students)"


def _osas(path: Path, ode_id: int) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df[(df["School ID"] == ode_id) & (df["Student Group"] == TOTAL)]
    return df.assign(n=pd.to_numeric(df["Number Proficient"]),
                     tested=pd.to_numeric(df["Number of Participants"]))


def split(name: str, ode_id: int) -> dict:
    enrollment = []
    for year in YEARS:
        row = load_year(year).set_index("ode_school_id").loc[ode_id]
        enrollment.append({"year": year, "k8": int(row[K8].sum()), "hs": int(row[HS].sum())})
    tests = {}
    for subj, path in OSAS_FILES.items():
        g = _osas(path, ode_id)
        hs = g[g["Grade Level"] == "Grade HS (11)"]
        lower = g[g["Grade Level"] != "Grade HS (11)"]
        # The grade rows must add back to ODE's all-grades row for the school.
        whole = _osas(OSAS_ALL[subj], ode_id)
        if (lower.n.sum() + hs.n.sum(), lower.tested.sum() + hs.tested.sum()) != (whole.n.iloc[0], whole.tested.iloc[0]):
            sys.exit(f"{name} {subj}: grade rows don't add to the all-grades total")
        tests[subj] = {"grades_3_8": {"proficient": int(lower.n.sum()), "tested": int(lower.tested.sum())},
                       "grade_11": {"proficient": int(hs.n.sum()), "tested": int(hs.tested.sum())}}
    return {"enrollment": enrollment, "osas_2425": tests}


def load() -> dict:
    return {name: split(name, ode_id) for name, ode_id in SPLIT_SCHOOLS.items()}


if __name__ == "__main__":
    import json
    print(json.dumps(load(), indent=1))
