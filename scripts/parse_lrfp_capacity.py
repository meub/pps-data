"""Parse per-school functional capacity from the 2021 PPS Long-Range Facility Plan.

The LRFP Vol 1 presents "Projected Utilization" tables for each grade band
(elementary / K-8 / middle / high) with columns:
    Site | Classrooms | Modular Classrooms | Functional Capacity | five years of utilization pcts

"Functional capacity" is PPS's own planning number: gross capacity
(classrooms × student-station size by area) minus set-aside rooms
(SPED focus, art/music, computer lab, DLI co-location, etc.), with a
configuration-specific utilization rate applied (85% at middle/high,
slightly higher at K-5/K-8), plus Title I / TSI / CSI reductions.

Output: data/raw/pps_functional_capacity_2021.json — [{"name", "classrooms",
"modular_classrooms", "functional_capacity"}, ...] keyed off the school's
short name as it appears in the LRFP.
"""
import json
import re
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parent.parent
LRFP = ROOT / "data/raw/LRFP_Vol1_2021.pdf"
OUT = ROOT / "data/raw/pps_functional_capacity_2021.json"

# Pages containing the tables (1-indexed as printed; PDF indices below).
PAGES = {
    "elementary": 53,
    "k8": 57,
    "middle": 60,
    "high": 63,
}

# Row: NAME  CLASSROOMS  MODULAR  FUNCTIONAL  PCT PCT PCT PCT PCT
# The capacity can carry a thousands-separator comma (e.g. Harrison Park: 1,006).
ROW_RE = re.compile(
    r"([A-Z][A-Z0-9 \.\-'\u2019/]+?)"          # name
    r"\s+(\d{1,2})"                             # classrooms
    r"\s+(\d{1,2})"                             # modular classrooms
    r"\s+(\d[\d,]{2,4})"                        # functional capacity (comma optional)
    r"\s+\d+%\s+\d+%\s+\d+%\s+\d+%\s+\d+%"      # 5 years of %
)


def parse_page(text, level):
    rows = []
    for line in text.split("\n"):
        for m in ROW_RE.finditer(line):
            rows.append({
                "name": m.group(1).strip(),
                "level": level,
                "classrooms": int(m.group(2)),
                "modular_classrooms": int(m.group(3)),
                "functional_capacity": int(m.group(4).replace(",", "")),
            })
    return rows


def main():
    out = []
    with pdfplumber.open(LRFP) as pdf:
        for level, page in PAGES.items():
            text = pdf.pages[page - 1].extract_text() or ""
            rows = parse_page(text, level)
            out.extend(rows)
            print(f"  {level}: {len(rows)} schools (p.{page})")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {len(out)} schools to {OUT}")


if __name__ == "__main__":
    main()
