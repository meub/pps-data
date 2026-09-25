"""Merge the PRC per-school enrollment forecast (Table 5.5, medium
scenario) into the master CSV. Currently the 2026 edition (2026-27 to
2035-36); the years below derive from BASELINE_YEAR.

Joins on school name. PRC uses short names (e.g. "MLC", "MLK Jr") which
are mapped to the master's canonical names via NAME_MAP below. Programs
other than Total are collapsed into the Total row per school.

Adds to data/pps_schools.csv (for the 2026 edition):
  prc_baseline_2025_26  (PRC's own count for the baseline year)
  enrollment_forecast_2026_27 ... enrollment_forecast_2035_36   (10 cols)
  enrollment_forecast_pct_change_10yr  (2025-26 baseline → 2035-36)
  enrollment_forecast_2035_36_low  (district Low/Medium ratio × school's horizon med)
  enrollment_forecast_2035_36_high (district High/Medium ratio × school's horizon med)

PRC publishes per-school forecasts in the Medium scenario only (Table 5.5).
Tables 5.3/5.4 give Low/High at the district-and-grade level. We scale each
school's horizon-year medium forecast by the district-wide Low-over-Medium and
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

# Fall of PRC's last actual-enrollment year (must match LAST_HISTORIC_YEAR
# in parse_pps_enrollment_forecast.py). Forecast runs 10 years after it.
BASELINE_YEAR = 2025
N_FORECAST = 10


def _sy(fall: int) -> str:
    """2025 -> '2025_26'."""
    return f"{fall}_{(fall + 1) % 100:02d}"


BASELINE = _sy(BASELINE_YEAR)
FORECAST_YEARS = [
    (_sy(y), f"fcst_{_sy(y)}")
    for y in range(BASELINE_YEAR + 1, BASELINE_YEAR + 1 + N_FORECAST)
]
HORIZON = FORECAST_YEARS[-1][0]

# District-wide horizon-year (2035-36) grade-band totals from PRC 2026
# Tables 5.2 (Medium), 5.3 (Low), and 5.4 (High). Used to construct
# per-school low/high bands by scaling each school's medium forecast by
# its band's ratio. Update these with each new PRC edition.
PRC_HORIZON_DISTRICT = {
    # (K-5): sum of K-2 + 3-5 from each table
    "k5": {"low": 7695 + 7729, "med": 8195 + 8371, "high": 9500 + 8945},
    # (6-8)
    "m68": {"low": 7763, "med": 8650, "high": 9230},
}


def scenario_ratios(level: str) -> tuple[float, float]:
    """Return (low_over_med, high_over_med) for a given school level."""
    if level == "middle":
        d = PRC_HORIZON_DISTRICT["m68"]
    else:
        # elementary, k8, alternative, other — pooled K-5 ratio
        d = PRC_HORIZON_DISTRICT["k5"]
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
    sum_cols = [f"hist_{BASELINE}"] + [col for _, col in FORECAST_YEARS]
    collapsed = (
        totals.groupby("_master_name", as_index=False)[sum_cols].sum(min_count=1)
    )

    # Rename to final column names.
    rename = {src: f"enrollment_forecast_{suffix}" for suffix, src in FORECAST_YEARS}
    rename[f"hist_{BASELINE}"] = f"prc_baseline_{BASELINE}"
    collapsed = collapsed.rename(columns=rename)
    collapsed = collapsed.rename(columns={"_master_name": "school_name"})

    # Drop any pre-existing forecast columns so re-runs don't accumulate.
    drop_cols = [
        c for c in master.columns
        if c.startswith("enrollment_forecast_")
        or c.startswith("prc_baseline_")
    ]
    if drop_cols:
        master = master.drop(columns=drop_cols)

    merged = master.merge(collapsed, on="school_name", how="left")

    # Derived: 10-year percent change using PRC's own historic baseline-year
    # count (NOT the master's ODE enrollment). This matters for co-located
    # programs like Odyssey-at-Hayhurst where master enrollment includes both
    # schools but PRC's forecast is Hayhurst-proper only.
    baseline = merged[f"prc_baseline_{BASELINE}"]
    future = merged[f"enrollment_forecast_{HORIZON}"]
    merged["enrollment_forecast_pct_change_10yr"] = (
        (future - baseline) / baseline
    ).where(baseline > 0)

    # Low / High scenario bands for the horizon year: scale medium by district-wide
    # Low-over-Medium and High-over-Medium ratios at the school's grade band.
    low_vals = []
    high_vals = []
    for _, row in merged.iterrows():
        med = row.get(f"enrollment_forecast_{HORIZON}")
        if pd.isna(med):
            low_vals.append(pd.NA)
            high_vals.append(pd.NA)
            continue
        lo_r, hi_r = scenario_ratios(row.get("level", ""))
        low_vals.append(round(med * lo_r))
        high_vals.append(round(med * hi_r))
    merged[f"enrollment_forecast_{HORIZON}_low"] = low_vals
    merged[f"enrollment_forecast_{HORIZON}_high"] = high_vals

    merged.to_csv(MASTER, index=False)

    matched_count = collapsed.shape[0]
    print(f"Merged {matched_count} schools into {MASTER}")

    # Summary: biggest projected declines among in-scope schools.
    in_scope = merged[merged["level"] != "high"].copy()
    in_scope = in_scope.dropna(subset=[f"enrollment_forecast_{HORIZON}"])
    if "enrollment_forecast_pct_change_10yr" in in_scope.columns:
        worst = in_scope.nsmallest(10, "enrollment_forecast_pct_change_10yr")
        print(f"\nBiggest projected declines by {HORIZON} (in-scope):")
        print(
            worst[[
                "school_name",
                f"prc_baseline_{BASELINE}",
                f"enrollment_forecast_{HORIZON}",
                "enrollment_forecast_pct_change_10yr",
            ]].to_string(index=False)
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
