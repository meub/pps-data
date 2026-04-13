"""Fetch 6 years of NCES CCD school-level enrollment for PPS (2018–2023).

Provides the historical context needed to compute multi-year enrollment
trends used in the long-term-sustainability analysis. Combined with the
2024-25 and 2025-26 ODE numbers already in the master CSV, this gives an
8-year window for each school.
"""
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/raw/pps_ccd_enrollment_history.json"
PPS_LEAID = "4110040"
BASE = "https://educationdata.urban.org/api/v1/schools/ccd/directory"
YEARS = [2018, 2019, 2020, 2021, 2022, 2023]


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
    for year in YEARS:
        print(f"Fetching CCD directory {year} ...")
        rows = fetch_year(year)
        # Filter to operational schools only.
        by_school = {}
        for r in rows:
            n = r.get("ncessch")
            if n is None:
                continue
            enr = r.get("enrollment")
            if enr is not None and enr >= 0:
                by_school[n] = enr
        payload[str(year)] = by_school
        print(f"  {len(by_school)} schools with enrollment")

    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
