"""Merge the PRC 2025 per-school enrollment forecast (Table 5.5, medium
scenario) into the master CSV.

Joins on school name. PRC uses short names (e.g. "MLC", "MLK Jr") which
are mapped to the master's canonical names via NAME_MAP below. Programs
other than Total are collapsed into the Total row per school.

Adds to data/pps_schools.csv:
  enrollment_forecast_2025_26 ... enrollment_forecast_2034_35   (10 cols)
  enrollment_forecast_pct_change_10yr  (2024-25 baseline → 2034-35)
  enrollment_forecast_2034_35_low  (district Low/Medium ratio × school's 2034 med)
  enrollment_forecast_2034_35_high (district High/Medium ratio × school's 2034 med)

PRC publishes per-school forecasts in the Medium scenario only (Table 5.5).
Tables 5.3/5.4 give Low/High at the district-and-grade level. We scale each
school's 2034-35 medium forecast by the district-wide Low-over-Medium and
High-over-Medium ratios at its grade band (K-5 vs 6-8), giving a rough
uncertainty band. Elementaries show wider bands than middle schools because
K/1 cohort recovery is the biggest scenario lever.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "data/pps_schools.csv"
FORECAST = ROOT / "data/raw/pps_enrollment_forecast.csv"

# PRC short name → master canonical name.
NAME_MAP = {
    "MLC": "Metropolitan Learning Center",
    "MLK Jr": "Dr. Martin Luther King Jr. School",
    "Boise-Eliot/Humboldt": "Boise-Eliot Elementary School",
    # These PRC rows have no master counterpart — skipped:
    #   OLA, Other (incl. Charters)
}

FORECAST_YEARS = [
    ("2025_26", "fcst_2025_26"),
    ("2026_27", "fcst_2026_27"),
    ("2027_28", "fcst_2027_28"),
    ("2028_29", "fcst_2028_29"),
    ("2029_30", "fcst_2029_30"),
    ("2030_31", "fcst_2030_31"),
    ("2031_32", "fcst_2031_32"),
    ("2032_33", "fcst_2032_33"),
    ("2033_34", "fcst_2033_34"),
    ("2034_35", "fcst_2034_35"),
]

# District-wide 2034-35 grade-band totals from PRC Tables 5.2 (Medium),
# 5.3 (Low), and 5.4 (High). Used to construct per-school low/high bands
# by scaling each school's medium forecast by its band's ratio.
PRC_2034_35_DISTRICT = {
    # (K-5): sum of K-2 + 3-5 from each table
    "k5": {"low": 7702 + 7412, "med": 8488 + 8908, "high": 11079 + 10153},
    # (6-8)
    "m68": {"low": 6774, "med": 7890, "high": 8479},
}


def scenario_ratios(level: str) -> tuple[float, float]:
    """Return (low_over_med, high_over_med) for a given school level."""
    if level == "middle":
        d = PRC_2034_35_DISTRICT["m68"]
    else:
        # elementary, k8, alternative, other — pooled K-5 ratio
        d = PRC_2034_35_DISTRICT["k5"]
    return d["low"] / d["med"], d["high"] / d["med"]


def canonical(name: str) -> str:
    return (
        name.replace(".", "")
        .replace(",", "")
        .replace(" ", "")
        .lower()
    )


def match_forecast_to_master(forecast_df: pd.DataFrame, master_names: list[str]) -> dict[str, str]:
    """Return {forecast_name → master_name}. Unmatched forecast rows are omitted."""
    canon_to_master = {canonical(m): m for m in master_names}
    mapping: dict[str, str] = {}
    for fc_name in forecast_df["school_name"].unique():
        if fc_name in NAME_MAP:
            mapping[fc_name] = NAME_MAP[fc_name]
            continue
        fck = canonical(fc_name)
        if fck in canon_to_master:
            mapping[fc_name] = canon_to_master[fck]
            continue
        # Substring fallback (e.g. "Abernethy" vs "Abernethy Elementary School")
        for mk, m in canon_to_master.items():
            if fck in mk or mk in fck:
                mapping[fc_name] = m
                break
    return mapping


def main() -> int:
    master = pd.read_csv(MASTER)
    forecast = pd.read_csv(FORECAST)

    # Keep only Total rows (per-school totals across programs).
    totals = forecast[forecast["program"] == "Total"].copy()

    name_map = match_forecast_to_master(totals, master["school_name"].tolist())
    unmatched = sorted(set(totals["school_name"]) - set(name_map))
    if unmatched:
        print(f"Skipped (no master row): {unmatched}")

    totals["_master_name"] = totals["school_name"].map(name_map)
    totals = totals.dropna(subset=["_master_name"])

    # Collapse duplicates (a school shouldn't appear twice as Total, but be safe
    # — e.g. the old "CreativeScience" and new "Bridger Creative Science" rows
    # both map to one master row; summing works because the old row is all zeros).
    sum_cols = ["hist_2024_25"] + [col for _, col in FORECAST_YEARS]
    collapsed = (
        totals.groupby("_master_name", as_index=False)[sum_cols].sum(min_count=1)
    )

    # Rename to final column names.
    rename = {src: f"enrollment_forecast_{suffix}" for suffix, src in FORECAST_YEARS}
    rename["hist_2024_25"] = "prc_baseline_2024_25"
    collapsed = collapsed.rename(columns=rename)
    collapsed = collapsed.rename(columns={"_master_name": "school_name"})

    # Drop any pre-existing forecast columns so re-runs don't accumulate.
    drop_cols = [
        c for c in master.columns
        if c.startswith("enrollment_forecast_")
        or c == "prc_baseline_2024_25"
    ]
    if drop_cols:
        master = master.drop(columns=drop_cols)

    merged = master.merge(collapsed, on="school_name", how="left")

    # Derived: 10-year percent change using PRC's own historic 2024-25 baseline
    # (NOT the master's enrollment_2024_25). This matters for co-located
    # programs like Odyssey-at-Hayhurst where master enrollment includes both
    # schools but PRC's forecast is Hayhurst-proper only.
    baseline = merged["prc_baseline_2024_25"]
    future = merged["enrollment_forecast_2034_35"]
    merged["enrollment_forecast_pct_change_10yr"] = (
        (future - baseline) / baseline
    ).where(baseline > 0)

    # Low / High scenario bands for 2034-35: scale medium by district-wide
    # Low-over-Medium and High-over-Medium ratios at the school's grade band.
    low_vals = []
    high_vals = []
    for _, row in merged.iterrows():
        med = row.get("enrollment_forecast_2034_35")
        if pd.isna(med):
            low_vals.append(pd.NA)
            high_vals.append(pd.NA)
            continue
        lo_r, hi_r = scenario_ratios(row.get("level", ""))
        low_vals.append(round(med * lo_r))
        high_vals.append(round(med * hi_r))
    merged["enrollment_forecast_2034_35_low"] = low_vals
    merged["enrollment_forecast_2034_35_high"] = high_vals

    merged.to_csv(MASTER, index=False)

    matched_count = collapsed.shape[0]
    print(f"Merged {matched_count} schools into {MASTER}")

    # Summary: biggest projected declines among in-scope schools.
    in_scope = merged[merged["level"] != "high"].copy()
    in_scope = in_scope.dropna(subset=["enrollment_forecast_2034_35"])
    if "enrollment_forecast_pct_change_10yr" in in_scope.columns:
        worst = in_scope.nsmallest(10, "enrollment_forecast_pct_change_10yr")
        print("\nBiggest projected declines by 2034-35 (in-scope):")
        print(
            worst[[
                "school_name",
                "enrollment_forecast_2025_26",
                "enrollment_forecast_2034_35",
                "enrollment_forecast_pct_change_10yr",
            ]].to_string(index=False)
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
