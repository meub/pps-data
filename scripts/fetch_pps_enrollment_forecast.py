#!/usr/bin/env python3
"""Fetch the 2026 PRC enrollment forecast PDF for Portland Public Schools.

Source:
  Portland State University, Population Research Center (PRC)
  "Portland Public Schools Enrollment Forecasts 2026-27 to 2035-36"
  Published July 24, 2026 (appendix tables dated May 19, 2026).

PRC publishes a new edition each summer. To move to the next one, update
URL and OUT here, then the year constants in parse_ and
merge_pps_enrollment_forecast.py. The 2025 edition stays in data/raw/
as the historical record.

URL is stable on PPS's finalsite CDN. Re-run with --force to re-download.

Output:
  data/raw/pps_enrollment_forecast_2026.pdf
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

URL = (
    "https://resources.finalsite.net/images/v1785171463/ppsnet/"
    "ejdr22voeddvux8qwlob/PPS_Forecast_2026_27.pdf"
)
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "pps_enrollment_forecast_2026.pdf"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if file exists")
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and not args.force:
        print(f"already fetched: {OUT} ({OUT.stat().st_size:,} bytes)")
        return 0

    print(f"fetching {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    OUT.write_bytes(data)
    print(f"wrote {OUT} ({len(data):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
