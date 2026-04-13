"""Export master CSV and column metadata to web/data.json for the static site."""
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "data/pps_schools.csv"
OUT_DATA = ROOT / "web/data.json"

# Column metadata: label, description, source, format.
# Format values: int, pct_0_100, pct_0_1, usd, text, bool, year, ratio.
META = {
    "school_name": {"label": "School", "desc": "School name (PPS Directory 2025-26).", "source": "PPS Directory", "fmt": "text"},
    "level": {"label": "Level", "desc": "Grade band served: elementary, k8, middle, high, alternative.", "source": "PPS Directory + ODE", "fmt": "text"},
    "is_closure_candidate": {"label": "Likely closure candidate", "desc": "One of the 15 schools Willamette Week ranked as most likely to be closed (2026-03-18). PPS has not released its own shortlist — that is expected Nov 2026.", "source": "Willamette Week", "fmt": "bool"},
    "closure_rank": {"label": "WW rank", "desc": "Willamette Week's published rank (1 = smallest), based solely on enrollment numbers. Not a PPS rank.", "source": "Willamette Week", "fmt": "int"},
    "enrollment_2025_26": {"label": "Enrollment 25-26", "desc": "Total students, fall 2025.", "source": "Oregon ODE Fall Membership", "fmt": "int"},
    "enrollment_2024_25": {"label": "Enrollment 24-25", "desc": "Total students, fall 2024.", "source": "Oregon ODE Fall Membership", "fmt": "int"},
    "enrollment_pct_change": {"label": "Enrollment Δ% YoY", "desc": "Year-over-year enrollment change (2024-25 → 2025-26).", "source": "Derived from ODE", "fmt": "pct_0_1"},
    "students_per_sqft": {"label": "Students / sqft", "desc": "Building crowding: 2025-26 enrollment ÷ 2009 KPFF square footage. Lower = more underused space.", "source": "Derived: ODE + KPFF 2009", "fmt": "ratio"},
    "year_built": {"label": "Year built", "desc": "Year the main building was constructed.", "source": "KPFF Seismic Report 2009", "fmt": "year"},
    "square_feet": {"label": "Square feet", "desc": "Building square footage (2009 inventory — may predate recent bond expansions).", "source": "KPFF Seismic Report 2009", "fmt": "int"},
    "construction_type_2009": {"label": "Construction", "desc": "Structural type: URM (unreinforced masonry), LRCW (reinforced concrete wall), Wood, Concrete, Steel, Masonry.", "source": "KPFF 2009", "fmt": "text"},
    "is_urm_building": {"label": "URM?", "desc": "Unreinforced masonry — highest seismic risk category.", "source": "Holmes Engineering 2024, via WW 2025-05-10", "fmt": "bool"},
    "urm_retrofit_cost_usd": {"label": "URM retrofit $", "desc": "Estimated cost to retrofit this URM building (only populated for URM schools).", "source": "Holmes Engineering 2024", "fmt": "usd"},
    "seismic_retrofit_status": {"label": "Seismic retrofit", "desc": "'full' = full modernization, 'targeted' = roof/partial retrofit done, 'planned_*' = 2025 bond scheduled, blank = none.", "source": "PPS bond.pps.net/seismic-improvements", "fmt": "text"},
    "is_title_i": {"label": "Title I?", "desc": "Receives federal Title I-A schoolwide funding (high-poverty designation).", "source": "PPS Funded Programs 2025-26", "fmt": "bool"},
    "pct_ela_prof_2425": {"label": "% ELA prof 24-25", "desc": "Percent of students meeting/exceeding on Oregon ELA state test (all grades, all students).", "source": "Oregon OSAS 2024-25", "fmt": "pct_0_100"},
    "pct_math_prof_2425": {"label": "% Math prof 24-25", "desc": "Percent of students meeting/exceeding on Oregon Math state test (all grades, all students).", "source": "Oregon OSAS 2024-25", "fmt": "pct_0_100"},
    "pct_ela_prof_2324": {"label": "% ELA prof 23-24", "desc": "Prior-year ELA proficiency.", "source": "Oregon OSAS 2023-24", "fmt": "pct_0_100"},
    "pct_math_prof_2324": {"label": "% Math prof 23-24", "desc": "Prior-year Math proficiency.", "source": "Oregon OSAS 2023-24", "fmt": "pct_0_100"},
    "crdc_chronic_absent_2020": {"label": "Chronic absent (CRDC)", "desc": "Count of students chronically absent (missed ≥15 days). Based on 2020 COVID year — likely suppressed.", "source": "US Dept of Ed CRDC 2020", "fmt": "int"},
    "crdc_lep_2020": {"label": "LEP students", "desc": "English Learner (Limited English Proficient) count.", "source": "US Dept of Ed CRDC 2020", "fmt": "int"},
    "crdc_idea_2020": {"label": "IDEA (SPED) students", "desc": "Students on IEPs under IDEA (special education).", "source": "US Dept of Ed CRDC 2020", "fmt": "int"},
    "frl_free_lunch": {"label": "Free lunch (count)", "desc": "Students eligible for free meals — raw count.", "source": "NCES CCD 2022", "fmt": "int"},
    "frl_reduced_lunch": {"label": "Reduced lunch (count)", "desc": "Students eligible for reduced-price meals — raw count.", "source": "NCES CCD 2022", "fmt": "int"},
    "pct_free_lunch": {"label": "% Free lunch", "desc": "Share of students on free meals (free ÷ 2022 enrollment) — proxy for poverty.", "source": "Derived: NCES CCD 2022", "fmt": "pct_0_1"},
    "pct_frl": {"label": "% FRL", "desc": "Share of students eligible for free OR reduced-price meals — standard poverty proxy.", "source": "Derived: NCES CCD 2022", "fmt": "pct_0_1"},
    "pct_direct_cert": {"label": "% Direct cert", "desc": "Share directly certified for free meals via SNAP/TANF/foster — proxy for deep poverty.", "source": "Derived: NCES CCD 2022", "fmt": "pct_0_1"},
    "pct_lep": {"label": "% English learners", "desc": "Share of students classified LEP.", "source": "Derived: CRDC 2020 ÷ current enrollment", "fmt": "pct_0_1"},
    "pct_idea": {"label": "% SPED (IDEA)", "desc": "Share of students on IEPs.", "source": "Derived: CRDC 2020 ÷ current enrollment", "fmt": "pct_0_1"},
    "pct_asian": {"label": "% Asian", "desc": "Share of students identifying as Asian, 2025-26.", "source": "Oregon ODE", "fmt": "pct_0_1"},
    "pct_black": {"label": "% Black", "desc": "Share of students identifying as Black/African American, 2025-26.", "source": "Oregon ODE", "fmt": "pct_0_1"},
    "pct_hispanic": {"label": "% Hispanic", "desc": "Share of students identifying as Hispanic/Latino, 2025-26.", "source": "Oregon ODE", "fmt": "pct_0_1"},
    "pct_white": {"label": "% White", "desc": "Share of students identifying as White, 2025-26.", "source": "Oregon ODE", "fmt": "pct_0_1"},
    "pct_multiracial": {"label": "% Multiracial", "desc": "Share identifying as two or more races, 2025-26.", "source": "Oregon ODE", "fmt": "pct_0_1"},
    "pct_bipoc": {"label": "% BIPOC", "desc": "Share of students identifying as any race/ethnicity other than White (1 − % White), 2025-26.", "source": "Derived from Oregon ODE", "fmt": "pct_0_1"},
    "avg_prof_2425": {"label": "Avg proficiency 24-25", "desc": "Average of % ELA + % Math meeting/exceeding (2024-25). Single number for cross-school performance comparison.", "source": "Derived from Oregon OSAS 2024-25", "fmt": "pct_0_100"},
    "service_load": {"label": "Service load (weighted)", "desc": "Cost-weighted composite: 1.0×% deep poverty + 2.5×% SPED + 1.5×% English learners. Weights approximate per-pupil cost ratios from weighted-student-funding formulas (Boston, Hawaii). A higher index means heavier per-student support load.", "source": "Derived: NCES CCD + CRDC", "fmt": "ratio"},
    "prof_residual": {"label": "Proficiency residual", "desc": "Each school's avg proficiency minus what a linear fit on % BIPOC would predict. Positive = outperforms demographics; negative = underperforms.", "source": "Derived: OSAS 2024-25 + ODE", "fmt": "ratio"},
    "affordable_units_within_1mi": {"label": "Afford. units in catchment", "desc": "Total existing affordable housing units inside the school's PPS attendance area (or a 1-mile radius for schools without a published catchment).", "source": "OAHI + Metro RLIS", "fmt": "int"},
    "pipeline_affordable_units_within_1mi": {"label": "Pipeline afford. units", "desc": "Affordable units in projects currently in development inside the school's PPS attendance area (2023–2027).", "source": "OAHI", "fmt": "int"},
    "pipeline_family_units_within_1mi": {"label": "Pipeline family units", "desc": "2+BR pipeline units inside the school's PPS attendance area — proxy for future families with kids.", "source": "OAHI", "fmt": "int"},
    "n_pipeline_projects_within_1mi": {"label": "Pipeline projects", "desc": "Number of affordable housing projects in development inside the school's PPS attendance area.", "source": "OAHI", "fmt": "int"},
    "permits_units_within_1mi_since_2022": {"label": "Permitted units (2022+)", "desc": "New residential units on building permits issued since 2022-01-01 inside the school's PPS attendance area — single-family, ADUs, and multifamily (all tenures). Schools without a published catchment fall back to a 1-mile radius.", "source": "Portland BDS via PortlandMaps", "fmt": "int"},
    "n_permits_within_1mi_since_2022": {"label": "Permits (2022+)", "desc": "Number of residential building permits issued since 2022-01-01 inside the school's PPS attendance area. Permits = approved to build; not all reach completion.", "source": "Portland BDS via PortlandMaps", "fmt": "int"},
    "nearest_alt_school_mi": {"label": "Miles to nearest alt.", "desc": "Great-circle distance from this closure candidate to the nearest non-candidate school of the same grade band (elementary, k8, middle, or alternative). A rough proxy for transportation impact: families would have to travel at least this far if their school closes (district reassignment may differ).", "source": "Derived from NCES CCD coordinates", "fmt": "miles"},
    "nearest_alt_school_name": {"label": "Nearest alt. school", "desc": "Name of the closest non-candidate school of the same grade band.", "source": "Derived from NCES CCD coordinates", "fmt": "text"},
    "street_address": {"label": "Address", "desc": "Street address of the building.", "source": "NCES CCD + manual", "fmt": "text"},
    "latitude": {"label": "Latitude", "desc": "Geocoded latitude.", "source": "NCES CCD", "fmt": "text"},
    "longitude": {"label": "Longitude", "desc": "Geocoded longitude.", "source": "NCES CCD", "fmt": "text"},
}

# Columns to surface in the default table (order matters).
TABLE_COLS = [
    "school_name", "level", "closure_rank",
    "enrollment_2025_26", "enrollment_pct_change", "students_per_sqft",
    "year_built", "square_feet", "pct_ela_prof_2425", "pct_math_prof_2425",
    "is_urm_building", "seismic_retrofit_status", "is_title_i", "pct_bipoc",
    "pipeline_family_units_within_1mi", "affordable_units_within_1mi",
    "permits_units_within_1mi_since_2022",
]

# Pre-defined scatter plots.
SCATTERS = [
    {
        "id": "enrollment_vs_sqft",
        "title": "Enrollment vs. building crowding",
        "x": "enrollment_2025_26",
        "y": "students_per_sqft",
        "subtitle": "Closure candidates cluster in the bottom-left (small + underused buildings).",
    },
    {
        "id": "math_vs_frl",
        "title": "Math proficiency vs. % low-income",
        "x": "pct_frl",
        "y": "pct_math_prof_2425",
        "subtitle": "Classic income-achievement gradient. Schools above the trend line outperform expectations; below, underperform.",
        "trendline": True,
    },
    {
        "id": "prof_vs_direct_cert",
        "title": "Avg proficiency vs. deep poverty (direct certification)",
        "x": "pct_direct_cert",
        "y": "avg_prof_2425",
        "subtitle": "Strongest correlation in the dataset (r = -0.90). Direct certification (SNAP/TANF/foster) predicts test scores more tightly than free/reduced lunch or any race/ethnicity variable.",
        "trendline": True,
    },
    {
        "id": "enrollment_trend_vs_permits",
        "title": "Enrollment change vs. nearby residential permits (2022+)",
        "x": "permits_units_within_1mi_since_2022",
        "y": "enrollment_pct_change",
        "subtitle": "Schools losing students while their neighborhood is actively being built warrant a closer look before closure.",
    },
    {
        "id": "year_vs_students",
        "title": "Year built vs. enrollment",
        "x": "year_built",
        "y": "enrollment_2025_26",
        "subtitle": "Many candidates are mid-century buildings with low current enrollment.",
    },
    {
        "id": "enrollment_vs_bipoc",
        "title": "Enrollment vs. % BIPOC students",
        "x": "enrollment_2025_26",
        "y": "pct_bipoc",
        "subtitle": "Do the 15 smallest-enrollment schools serve a disproportionately BIPOC student body compared with the rest of PPS?",
    },
    {
        "id": "urm_cost_vs_enrollment",
        "title": "URM retrofit cost vs. enrollment",
        "x": "enrollment_2025_26",
        "y": "urm_retrofit_cost_usd",
        "subtitle": "URM buildings only. The slope is cost-per-student to save; schools well above the trend are expensive to retrofit per retained seat.",
        "trendline": True,
    },
    {
        "id": "service_load_vs_enrollment",
        "title": "Weighted service load vs. enrollment",
        "x": "service_load",
        "y": "enrollment_2025_26",
        "subtitle": "Cost-weighted index: 1.0×% deep poverty + 2.5×% SPED + 1.5×% English learners (weights from WSF formulas). Top-right schools absorb the most need; closing them concentrates services elsewhere.",
    },
]

# Features used for k-means clustering + PCA projection.
CLUSTER_FEATURES = [
    "enrollment_2025_26", "enrollment_pct_change", "students_per_sqft",
    "year_built", "avg_prof_2425", "pct_direct_cert", "pct_bipoc",
    "pct_lep", "pct_idea",
    "permits_units_within_1mi_since_2022", "pipeline_family_units_within_1mi",
]

# Human-readable archetype labels assigned post-fit, ordered by cluster_id.
CLUSTER_LABELS = {
    0: {"name": "High-need urban", "desc": "Mid-poverty, BIPOC-majority elementaries, mostly stable enrollment, lower test scores."},
    1: {"name": "Large + growing", "desc": "Larger schools with strong nearby housing pipeline; no closure candidates."},
    2: {"name": "West-side advantaged", "desc": "Higher-performing, lower-poverty, mostly white schools."},
    3: {"name": "Aging, high-SPED", "desc": "Tiny cluster: very old buildings, very high SPED %, declining enrollment."},
}


def derive_columns(df):
    """Add computed fields used by the dashboard's analytical charts."""
    df["pct_bipoc"] = 1 - df["pct_white"]
    df["avg_prof_2425"] = (df["pct_ela_prof_2425"] + df["pct_math_prof_2425"]) / 2

    # Weighted service load. Weights approximate per-pupil cost multipliers used in
    # weighted-student-funding (WSF) formulas: 1.0× base for deep poverty, ~2.5× for
    # SPED (IDEA average across tiers, Hawaii/Boston), ~1.5× for English Learner
    # (Boston/Hawaii ELL weight). Only valid when all three components are present.
    sl_mask = df[["pct_direct_cert", "pct_idea", "pct_lep"]].notna().all(axis=1)
    df["service_load"] = pd.NA
    df.loc[sl_mask, "service_load"] = (
        1.0 * df.loc[sl_mask, "pct_direct_cert"]
        + 2.5 * df.loc[sl_mask, "pct_idea"]
        + 1.5 * df.loc[sl_mask, "pct_lep"]
    ).round(4)

    # Proficiency residual from OLS fit of avg_prof_2425 ~ pct_bipoc.
    pr_mask = df[["avg_prof_2425", "pct_bipoc"]].notna().all(axis=1)
    x = df.loc[pr_mask, "pct_bipoc"].to_numpy()
    y = df.loc[pr_mask, "avg_prof_2425"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    df["prof_residual"] = pd.NA
    df.loc[pr_mask, "prof_residual"] = (y - (slope * x + intercept)).round(2)

    # K-means clusters + PCA coords on standardized feature space.
    cl_mask = df[CLUSTER_FEATURES].notna().all(axis=1)
    X = StandardScaler().fit_transform(df.loc[cl_mask, CLUSTER_FEATURES])
    km = KMeans(n_clusters=4, n_init=20, random_state=42)
    df["cluster_id"] = pd.NA
    df.loc[cl_mask, "cluster_id"] = km.fit_predict(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    df["pca_x"] = pd.NA
    df["pca_y"] = pd.NA
    df.loc[cl_mask, "pca_x"] = coords[:, 0].round(3)
    df.loc[cl_mask, "pca_y"] = coords[:, 1].round(3)

    # Transportation impact: haversine miles from each closure candidate to the
    # nearest non-candidate school of the same grade band.
    R_MI = 3958.7613  # earth radius in miles
    df["nearest_alt_school_mi"] = pd.NA
    df["nearest_alt_school_name"] = pd.NA
    cand_idx = df.index[df["is_closure_candidate"].astype(bool)]
    for i in cand_idx:
        cand_lat = df.at[i, "latitude"]
        cand_lon = df.at[i, "longitude"]
        cand_level = df.at[i, "level"]
        if pd.isna(cand_lat) or pd.isna(cand_lon):
            continue
        peers = df[
            (df["level"] == cand_level)
            & (~df["is_closure_candidate"].astype(bool))
            & df["latitude"].notna()
            & df["longitude"].notna()
            & (df.index != i)
        ]
        if len(peers) == 0:
            continue
        lat1 = math.radians(float(cand_lat))
        lon1 = math.radians(float(cand_lon))
        lat2 = np.radians(peers["latitude"].to_numpy(dtype=float))
        lon2 = np.radians(peers["longitude"].to_numpy(dtype=float))
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + math.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        d_mi = R_MI * c
        nearest = int(np.argmin(d_mi))
        df.at[i, "nearest_alt_school_mi"] = round(float(d_mi[nearest]), 2)
        df.at[i, "nearest_alt_school_name"] = peers.iloc[nearest]["school_name"]
    return df


def clean_val(v):
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if pd.isna(v):
        return None
    return v


def main():
    df = pd.read_csv(MASTER)
    # PPS's closure announcement covers only elementary, K-8, middle, and
    # alternative schools (not high schools). Drop high schools from the
    # dashboard so every view reflects the in-scope set.
    df = df[df["level"] != "high"].reset_index(drop=True)
    df = derive_columns(df)
    schools = []
    for _, row in df.iterrows():
        schools.append({c: clean_val(row[c]) for c in df.columns})

    payload = {
        "schools": schools,
        "meta": META,
        "table_cols": TABLE_COLS,
        "scatters": SCATTERS,
        "cluster_labels": CLUSTER_LABELS,
        "n_schools": len(schools),
        "n_candidates": int(df["is_closure_candidate"].sum()),
    }
    OUT_DATA.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {len(schools)} schools to {OUT_DATA}")


if __name__ == "__main__":
    main()
