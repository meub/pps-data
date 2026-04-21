#!/usr/bin/env python3
"""Fetch the 2025 PRC enrollment forecast PDF for Portland Public Schools.

Source:
  Portland State University, Population Research Center (PRC)
  "Portland Public Schools Enrollment Forecasts 2025-26 to 2034-35"
  Published July 21, 2025 (appendix tables refreshed May 30, 2025).

URL is stable on PPS's finalsite CDN. Re-run with --force to re-download.

Output:
  data/raw/pps_enrollment_forecast_2025.pdf
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

URL = (
    "https://resources.finalsite.net/images/v1759783181/ppsnet/"
    "xzghnbm55ogovbz4eisd/PPS_Forecast_2025.pdf"
)
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "pps_enrollment_forecast_2025.pdf"


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
