#!/usr/bin/env python3
"""Fetch PPS's rightsizing scenario comparison PDF.

Source:
  Portland Public Schools, "Rightsizing: Comparing three scenarios"
  (Scenario A, Scenario B, Status Quo). Two pages, footer "Updated
  October 4, 2026". Shared by PPS as a Google Drive file on 2026-10-05.

If PPS posts a revised version, update FILE_ID and OUT (OUT carries the
PDF's "Updated" date), then rerun parse_rightsizing_scenarios.py.

Output:
  data/raw/pps_rightsizing_scenarios_2026-10-04.pdf
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

FILE_ID = "1ueJVC-Jo6MRKw56Jedihc6YbVdBlSB2z"
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "pps_rightsizing_scenarios_2026-10-04.pdf"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if file exists")
    args = ap.parse_args()

    if OUT.exists() and not args.force:
        print(f"already fetched: {OUT} ({OUT.stat().st_size:,} bytes)")
        return 0

    print(f"fetching {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    # Drive serves an HTML interstitial instead of the file when sharing
    # changes; fail loudly rather than saving it as a .pdf.
    if not data.startswith(b"%PDF"):
        print("ERROR: response is not a PDF (Drive sharing may have changed)")
        return 1
    OUT.write_bytes(data)
    print(f"wrote {OUT} ({len(data):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
