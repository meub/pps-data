"""Fetch NCES CCD school-level data for PPS: 6 years of enrollment
(2018–2023) plus teacher FTE for the most recent year (2023).

Enrollment history feeds the long-term-sustainability analysis.
Teacher FTE feeds the student-to-teacher ratio; only the most recent
year is stored since older years add no analytical value here.
"""
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/raw/pps_ccd_enrollment_history.json"
OUT_TEACHERS = ROOT / "data/raw/pps_ccd_teachers_2023.json"
PPS_LEAID = "4110040"
BASE = "https://educationdata.urban.org/api/v1/schools/ccd/directory"
YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
TEACHER_YEAR = 2023


def fetch_year(year):
    url = f"{BASE}/{year}/?fips=41&leaid={PPS_LEAID}&page_size=500"
    rows = []
    while url:
        with urllib.request.urlopen(url) as r:
            d = json.loads(r.read())
        rows.extend(d.get("results", []))
        url = d.get("next")
    return rows


def main():
    payload = {}
    teachers = {}
    for year in YEARS:
        print(f"Fetching CCD directory {year} ...")
        rows = fetch_year(year)
        by_school = {}
        for r in rows:
            n = r.get("ncessch")
            if n is None:
                continue
            enr = r.get("enrollment")
            if enr is not None and enr >= 0:
                by_school[n] = enr
            if year == TEACHER_YEAR:
                fte = r.get("teachers_fte")
                if fte is not None and fte > 0:
                    teachers[n] = fte
        payload[str(year)] = by_school
        print(f"  {len(by_school)} schools with enrollment")

    OUT.write_text(json.dumps(payload, indent=2))
    OUT_TEACHERS.write_text(json.dumps(teachers, indent=2))
    print(f"Wrote {OUT}")
    print(f"Wrote {OUT_TEACHERS} ({len(teachers)} schools with teacher FTE)")


if __name__ == "__main__":
    main()
