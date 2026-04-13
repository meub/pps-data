"""Fetch CRDC 2021 (school year 2020-21) PPS data from the Urban Institute API.

Three series:
  * teachers-staff: counselor / social worker / psychologist / nurse FTE.
  * discipline-instances: out-of-school suspension instances.
  * chronic-absenteeism (race/sex breakdown): aggregate to school total.

Aggregates each to per-school totals and writes data/raw/pps_crdc_2021_agg.json
in the same shape as pps_crdc_agg.json. Cached locally; rerun anytime.
"""
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/raw/pps_crdc_2021_agg.json"
PPS_LEAID = "4110040"
BASE = "https://educationdata.urban.org/api/v1/schools/crdc"


def fetch_all(path):
    """Page through a paginated Urban Institute endpoint."""
    url = f"{BASE}/{path}?fips=41&leaid={PPS_LEAID}&page_size=500"
    rows = []
    while url:
        with urllib.request.urlopen(url) as r:
            d = json.loads(r.read())
        rows.extend(d.get("results", []))
        url = d.get("next")
    return rows


def main():
    print("Fetching enrollment/2021/race/sex (totals only) ...")
    enr = fetch_all("enrollment/2021/race/sex")
    enr_by_school = {}
    for r in enr:
        if (r.get("race") == 99 and r.get("sex") == 99 and r.get("disability") == 99
                and r.get("lep") == 99):
            v = _f(r.get("enrollment_crdc"))
            if v is not None and v >= 0:
                enr_by_school[r["ncessch"]] = v
    print(f"  {len(enr_by_school)} schools")

    print("Fetching teachers-staff/2021/ ...")
    staff = fetch_all("teachers-staff/2021")
    staff_by_school = {}
    for r in staff:
        staff_by_school[r["ncessch"]] = {
            "teachers_fte": _f(r.get("teachers_fte_crdc")),
            "counselors_fte": _f(r.get("counselors_fte")),
            "psychologists_fte": _f(r.get("psychologists_fte")),
            "social_workers_fte": _f(r.get("social_workers_fte")),
            "nurses_fte": _f(r.get("nurses_fte")),
            "security_guard_fte": _f(r.get("security_guard_fte")),
            "law_enforcement_fte": _f(r.get("law_enforcement_fte")),
        }
    print(f"  {len(staff_by_school)} schools")

    print("Fetching discipline-instances/2021/ ...")
    disc = fetch_all("discipline-instances/2021")
    # rows are per (ncessch, disability flag). Sum suspensions across rows; suppressed
    # values are negative, treat as 0.
    susp_by_school = {}
    for r in disc:
        n = r["ncessch"]
        v = _f(r.get("suspensions_instances"))
        if v is None or v < 0:
            v = 0
        susp_by_school[n] = susp_by_school.get(n, 0) + v
    print(f"  {len(susp_by_school)} schools (after aggregation)")

    print("Fetching chronic-absenteeism/2021/race/sex/ ...")
    absent = fetch_all("chronic-absenteeism/2021/race/sex")
    # Race=99 sex=99 is the unduplicated "Total" row in CRDC. Use that to avoid
    # double-counting; if missing, sum across race=99 sex=1+sex=2.
    abs_total = {}
    abs_sex_sum = {}
    for r in absent:
        n = r["ncessch"]
        v = _f(r.get("students_chronically_absent"))
        if v is None or v < 0:
            continue
        if r.get("race") == 99 and r.get("sex") == 99 and r.get("disability") == 99 and r.get("lep") == 99 and r.get("homeless") == 99:
            abs_total[n] = v
        elif r.get("race") == 99 and r.get("sex") in (1, 2) and r.get("disability") == 99 and r.get("lep") == 99 and r.get("homeless") == 99:
            abs_sex_sum[n] = abs_sex_sum.get(n, 0) + v
    abs_by_school = {n: abs_total.get(n, abs_sex_sum.get(n)) for n in (set(abs_total) | set(abs_sex_sum))}
    print(f"  {len(abs_by_school)} schools (race=99 totals)")

    payload = {
        "year": 2021,
        "enrollment": enr_by_school,
        "teachers_staff": staff_by_school,
        "suspensions_instances": susp_by_school,
        "chronic_absent": abs_by_school,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT}")


def _f(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
