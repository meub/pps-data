"""Build master PPS schools dataset from ODE enrollment + closure-candidate list,
merged with NCES CCD 2022 (addresses, lat/lon, FRL) and 2009 KPFF facility data
(year built, square footage, construction type)."""
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ODE = ROOT / "data/raw/fall_membership_2025_26.xlsx"
CCD = ROOT / "data/raw/pps_ccd_directory_2022.json"
FACILITY = ROOT / "data/pps_facility_2009.csv"
CRDC = ROOT / "data/raw/pps_crdc_agg.json"
CRDC_2021 = ROOT / "data/raw/pps_crdc_2021_agg.json"
CCD_HISTORY = ROOT / "data/raw/pps_ccd_enrollment_history.json"
OSAS_ELA_25 = ROOT / "data/raw/pagr_schools_ela_all_2425.xlsx"
OSAS_MATH_25 = ROOT / "data/raw/pagr_schools_math_all_2425.xlsx"
OSAS_ELA_24 = ROOT / "data/raw/pagr_schools_ela_all_2324.xlsx"
OSAS_MATH_24 = ROOT / "data/raw/pagr_schools_math_all_2324.xlsx"
OUT = ROOT / "data/pps_schools.csv"

# Closure candidates from WW article (3/18/2026). Keys are ODE school names.
CANDIDATES = {
    "Rosa Parks Elementary School": (1, 160),
    "Whitman Elementary School": (2, 194),
    "Skyline Elementary School": (3, 207),       # listed as K-8 in PPS directory
    "Creston Elementary School": (4, 213),
    # Odyssey Program — not in ODE (embedded program). Added manually below.
    "Woodmere Elementary School": (6, 227),
    "Arleta Elementary School": (7, 230),
    "Peninsula Elementary School": (8, 239),
    "Chief Joseph Elementary School": (9, 242),
    "Lee Elementary School": (10, 251),
    "Lent Elementary School": (11, 256),
    "Irvington Elementary School": (12, 259),
    "Lewis Elementary School": (13, 259),
    "Forest Park Elementary School": (14, 264),
    "Rieke Elementary School": (15, 264),
}

# PPS-directory level classifications where they differ from ODE's "Elementary"
# default. ODE classifies many K-8s as "Elementary"; the PPS directory is the
# authoritative source for school configuration.
# Unreinforced masonry buildings (Holmes Engineering 2024 assessment,
# reported by WW 2025-05-10). Value = estimated URM retrofit cost in USD.
URM_BUILDINGS = {
    "Ainsworth Elementary School": 7_870_000,
    "Beach Elementary School": 1_350_000,
    "Beverly Cleary School": 14_400_000,
    "Buckman Elementary School": 19_750_000,
    "Capitol Hill Elementary School": 3_430_000,
    "Creston Elementary School": 10_970_000,
    "George Middle School": 11_180_000,
    "Hayhurst Elementary School": 520_000,
    "James John Elementary School": 1_020_000,
    "Kelly Elementary School": 1_930_000,
    "Marysville Elementary School": 1_400_000,
    "Richmond Elementary School": 1_720_000,
    "Rieke Elementary School": 4_160_000,
    "Rigler Elementary School": 2_850_000,
    "Rose City Park": 20_910_000,
    "Sabin Elementary School": 2_600_000,
    "Vernon Elementary School": 2_450_000,
    "Winterhaven School": 810_000,
    # Wilcox (Columbia Regional Inclusive Services) — not in our PPS-operated roster.
}

# Seismic retrofit status (2012/2017/2020 bonds + 2025 bond plans).
# Source: bond.pps.net/seismic-improvements.
# Values: "full" = full seismic retrofit/modernization complete,
#         "targeted" = targeted retrofit or roof-level completed,
#         "planned_full" / "planned_targeted" = design or construction scheduled,
#         None = no retrofit recorded.
SEISMIC_RETROFIT = {
    # Completed full retrofits / modernizations
    "Benson Polytechnic High School": "full",
    "Franklin High School": "full",
    "Grant High School": "full",
    "Lincoln High School": "full",
    "Leodis V. McDaniel High School": "full",
    "Roosevelt High School": "full",
    "Alameda Elementary School": "full",
    "Bridger Creative Science School": "full",
    "Faubion Elementary School": "full",
    "Hayhurst Elementary School": "full",
    "Kellogg Middle School": "full",
    "Lent Elementary School": "full",
    "Lewis Elementary School": "full",
    "Marysville Elementary School": "full",
    # Completed targeted retrofits / roof-level improvements
    "Abernethy Elementary School": "targeted",
    "Arleta Elementary School": "targeted",
    "Boise-Eliot Elementary School": "targeted",
    "Bridlemile Elementary School": "targeted",
    "Buckman Elementary School": "targeted",
    "César Chávez K-8 School": "targeted",
    "Chapman Elementary School": "targeted",
    "Chief Joseph Elementary School": "targeted",
    "Cleveland High School": "targeted",
    "Creston Elementary School": "targeted",
    "Duniway Elementary School": "targeted",
    "Glencoe Elementary School": "targeted",
    "Harriet Tubman Middle School": "targeted",
    "Harrison Park School": "targeted",
    "Hosford Middle School": "targeted",
    "Ida B. Wells-Barnett High School": "targeted",
    "Jackson Middle School": "targeted",
    "James John Elementary School": "targeted",
    "Laurelhurst Elementary School": "targeted",
    "Maplewood Elementary School": "targeted",
    "Markham Elementary School": "targeted",
    "Dr. Martin Luther King Jr. School": "targeted",
    "Mt Tabor Middle School": "targeted",
    "Ockley Green Middle School": "targeted",
    "Rigler Elementary School": "targeted",
    "Sabin Elementary School": "targeted",
    "Sellwood Middle School": "targeted",
    "Sitton Elementary School": "targeted",
    "Skyline Elementary School": "targeted",
    "Stephenson Elementary School": "targeted",
    "Woodlawn Elementary School": "targeted",
    # Planned (2025 bond, design starting / construction scheduled)
    "Vernon Elementary School": "planned_targeted",
    "Winterhaven School": "planned_targeted",
    "Ainsworth Elementary School": "planned_targeted",
    "Beach Elementary School": "planned_targeted",
    "Capitol Hill Elementary School": "planned_targeted",
    "Kelly Elementary School": "planned_targeted",
    "Richmond Elementary School": "planned_targeted",
    "Beverly Cleary School": "planned_full",
    "Rose City Park": "planned_full",
}

# Specialized programs hosted at each PPS school. Two categories:
#   1. Dual Language Immersion (DLI) — five languages, district-administered.
#   2. Focus options — distinctive curricular models (Arts, Environmental,
#      STEAM, accelerated/TAG, history-integrated, alternative).
# Each value is a list of program tags. Sources:
#   pps.net/departments/dual-language/current-dli-programs (DLI by language)
#   pps.net/departments/enrollment-transfer/enroll/explore-your-options/k-5-explore-your-options
# Captured 2026-04 — programs change between school years, re-verify annually.
PROGRAMS = {
    # Spanish DLI elementary
    "Ainsworth Elementary School": ["Spanish DLI"],
    "Atkinson Elementary School": ["Spanish DLI"],
    "Beach Elementary School": ["Spanish DLI"],
    "César Chávez K-8 School": ["Spanish DLI"],
    "James John Elementary School": ["Spanish DLI"],
    "Lent Elementary School": ["Spanish DLI"],
    "Rigler Elementary School": ["Spanish DLI"],
    "Scott Elementary School": ["Spanish DLI"],
    "Sitton Elementary School": ["Spanish DLI"],
    # Spanish DLI middle
    "Beaumont Middle School": ["Spanish DLI"],
    "George Middle School": ["Spanish DLI"],
    "Kellogg Middle School": ["Spanish DLI"],
    "Ockley Green Middle School": ["Spanish DLI"],
    "West Sylvan Middle School": ["Spanish DLI"],
    # Chinese DLI
    "Clark Elementary School": ["Chinese DLI"],
    "Dr. Martin Luther King Jr. School": ["Chinese DLI"],
    "Woodstock Elementary School": ["Chinese DLI"],
    "Harrison Park School": ["Chinese DLI"],
    "Harriet Tubman Middle School": ["Chinese DLI"],
    # Japanese DLI
    "Richmond Elementary School": ["Japanese DLI"],
    "Mt Tabor Middle School": ["Japanese DLI"],
    # Russian DLI
    "Kelly Elementary School": ["Russian DLI"],
    "Lane Middle School": ["Russian DLI"],
    # Vietnamese DLI
    "Rose City Park": ["Vietnamese DLI"],
    # Roseway Heights = Spanish + Vietnamese DLI middle feeder
    "Roseway Heights School": ["Spanish DLI", "Vietnamese DLI"],
    # Focus options
    "Buckman Elementary School": ["Arts focus"],
    "Bridger Creative Science School": ["Creative Science focus"],
    "Sunnyside Environmental School": ["Environmental focus"],
    "Winterhaven School": ["STEAM focus"],
    "Odyssey Program (K-8)": ["TAG / integrated focus"],
    "ACCESS Academy": ["TAG / accelerated focus"],
    "Metropolitan Learning Center": ["Alternative K-12 focus"],
}


# Title I schoolwide schools, 2025-26 (from pps.net/departments/funded-programs/title-ia).
# All PPS Title I designations are schoolwide model. Charter Title I schools
# (Arthur Academy, Kairos) are excluded from our master (not PPS-operated).
TITLE_I = {
    "Arleta Elementary School",
    "Atkinson Elementary School",
    "Boise-Eliot Elementary School",
    "Buckman Elementary School",
    "César Chávez K-8 School",
    "Chapman Elementary School",
    "Clark Elementary School",
    "Dr. Martin Luther King Jr. School",
    "Faubion Elementary School",
    "Grout Elementary School",
    "James John Elementary School",
    "Kelly Elementary School",
    "Lee Elementary School",
    "Lent Elementary School",
    "Markham Elementary School",
    "Marysville Elementary School",
    "Peninsula Elementary School",
    "Rigler Elementary School",
    "Rosa Parks Elementary School",
    "Scott Elementary School",
    "Sitton Elementary School",
    "Vestal Elementary School",
    "Whitman Elementary School",
    "Woodlawn Elementary School",
    "Woodmere Elementary School",
    "George Middle School",
    "Harriet Tubman Middle School",
    "Harrison Park School",
    "Kellogg Middle School",
    "Lane Middle School",
    "Ockley Green Middle School",
    "Roseway Heights School",
    "Jefferson High School",
    "Roosevelt High School",
}

# Name-matching: master_name -> CCD 2022 school_name. Only entries where names differ.
CCD_NAME_MAP = {
    "Beverly Cleary School ": "Beverly Cleary School",
    "Bridger Creative Science School": "Bridger Elementary School",
    "César Chávez K-8 School": "Cesar Chavez K-8 School",
}

# Master_name -> facility_2009 school_name_2009. Accounts for renamings,
# typos, and schools that reopened after 2009 (Kellogg, Rose City Park).
FACILITY_NAME_MAP = {
    "Abernethy Elementary School": "Abernathy Elementary",
    "Ainsworth Elementary School": "Ainsworth Elementary",
    "Alameda Elementary School": "Alameda Elementary",
    "Alliance High School": "Meek / Alliance",
    "Arleta Elementary School": "Arleta Elementary",
    "Astor Elementary School": "Astor Elementary",
    "Atkinson Elementary School": "Atkinson Elementary",
    "Beach Elementary School": "Beach Elementary",
    "Beaumont Middle School": "Beaumont Middle School",
    "Benson Polytechnic High School": "Benson High School",
    "Beverly Cleary School ": "Fernwood Middle School",
    "Boise-Eliot Elementary School": "Boise Elliot Elementary",
    "Bridger Creative Science School": "Creative Science Elementary",
    "Bridlemile Elementary School": "Bridlemile Elementary",
    "Buckman Elementary School": "Buckman Elementary",
    "Capitol Hill Elementary School": "Capitol Hill Elementary",
    "Chapman Elementary School": "Chapman Elementary",
    "Chief Joseph Elementary School": "Chief Joseph Elementary",
    "Cleveland High School": "Cleveland High School",
    "Creston Elementary School": "Creston Elementary",
    "César Chávez K-8 School": "Portsmouth / Clarendon",
    "Dr. Martin Luther King Jr. School": "King Elementary",
    "Duniway Elementary School": "Duniway Elementary",
    "Faubion Elementary School": "Faubion Elementary",
    "Forest Park Elementary School": "Forest Park",
    "Franklin High School": "Franklin High School",
    "George Middle School": "George Middle School",
    "Glencoe Elementary School": "Glencoe Elementary",
    "Grant High School": "Grant High School",
    "Gray Middle School": "Gray Middle School",
    "Grout Elementary School": "Grout Middle School",
    "Harriet Tubman Middle School": "Tubman Middle School",
    "Harrison Park School": "Harrison Park Middle School",
    "Hayhurst Elementary School": "Hayhurst Elementary",
    "Hosford Middle School": "Hosford Middle School",
    "Ida B. Wells-Barnett High School": "Wilson High School",
    "Irvington Elementary School": "Irvington Elementary",
    "Jackson Middle School": "Jackson Middle School",
    "James John Elementary School": "James John Elementary",
    "Jefferson High School": "Jefferson High School",
    "Kellogg Middle School": "Kellogg Middle School closed",
    "Kelly Elementary School": "Kelly Elementary",
    "Lane Middle School": "Lane Middle School",
    "Laurelhurst Elementary School": "Laurelhurst Elementary",
    "Lee Elementary School": "Lee Elementary",
    "Lent Elementary School": "Lent Elementary",
    "Leodis V. McDaniel High School": "Madison High School",
    "Lewis Elementary School": "Lewis Elementary",
    "Lincoln High School": "Lincoln High School",
    "Llewellyn Elementary School": "Llewelyn Elementary",
    "Maplewood Elementary School": "Maplewood Elementary",
    "Markham Elementary School": "Markham Elementary",
    "Marysville Elementary School": "Marysville Elementary",
    "Metropolitan Learning Center": "Metro Learning Center",
    "Mt Tabor Middle School": "Mt. Tabor Middle School",
    "Ockley Green Middle School": "Ockley Green",
    "Peninsula Elementary School": "Peninsula Elementary",
    "Richmond Elementary School": "Richmond Elementary",
    "Rieke Elementary School": "Rieke Elementary",
    "Rigler Elementary School": "Rigler Elementary",
    "Roosevelt High School": "Roosevelt High School",
    "Rosa Parks Elementary School": "Rosa Parks",
    "Rose City Park": "Rose City Park Elementary closed",
    "Roseway Heights School": "Gregory Heights Middle School",
    "Sabin Elementary School": "Sabin Elementary",
    "Scott Elementary School": "Scott Elementary",
    "Sellwood Middle School": "Sellwood Middle School",
    "Sitton Elementary School": "Sitton Elementary",
    "Skyline Elementary School": "Skyline Elementary",
    "Stephenson Elementary School": "Stephenson Elementary",
    "Sunnyside Environmental School": "Sunnyside Elementary",
    "Vernon Elementary School": "Vernon Elementary",
    "Vestal Elementary School": "Vestal Elementary",
    "West Sylvan Middle School": "West Sylvan Middle School",
    "Whitman Elementary School": "Whitman Elementary",
    "Winterhaven School": "Winterhaven @ Brooklyn",
    "Woodlawn Elementary School": "Woodlawn Elementary",
    "Woodmere Elementary School": "Woodmere Elementary",
    "Woodstock Elementary School": "Woodstock Elementary",
    "da Vinci Middle School": "DaVinci",
    # Clark Elementary (opened post-2009 at former ACCESS site) — no 2009 row.
    # Odyssey Program is embedded, no facility of its own.
}


# Manual patches for schools missing from CCD 2022 (Clark opened post-2022;
# Odyssey is an embedded program hosted at Hayhurst in 2024-25).
MANUAL_LOCATION = {
    "Clark Elementary School": {
        "street_address": "1231 SE 92nd Ave", "city": "Portland", "zip_code": "97216",
        "latitude": 45.5133327, "longitude": -122.5706442,
    },
    "Odyssey Program (K-8)": {
        "street_address": "5037 SW Iowa St", "city": "Portland", "zip_code": "97221",
        "latitude": 45.4804, "longitude": -122.729,
    },
}


# Schools absent from ODE (e.g., embedded programs without their own institution
# ID). Appended as stub rows after the main merge so PPS's 74-school in-scope
# count is preserved downstream.
SUPPLEMENTAL_SCHOOLS = [
    {
        "School Name": "ACCESS Academy",
        "level": "alternative",
        "school_type": "alternative",
        "street_address": "6318 S Corbett Ave",
        "city": "Portland",
        "zip_code": "97239",
        "latitude": 45.4783395,
        "longitude": -122.6753753,
        "is_closure_candidate": False,
        "programs": "TAG / accelerated focus",
        "has_dli": False,
        "has_focus_option": True,
        "dli_languages": None,
    },
]


LEVEL_OVERRIDES = {
    "Astor Elementary School": "k8",
    "Beach Elementary School": "k8",
    "Beverly Cleary School": "k8",
    "Bridger Creative Science School": "k8",
    "César Chávez K-8 School": "k8",
    "Faubion Elementary School": "k8",
    "Harrison Park School": "middle",         # Harrison Park Middle School
    "Laurelhurst Elementary School": "k8",
    "Marysville Elementary School": "k8",
    "Rose City Park": "elementary",
    "Roseway Heights School": "middle",       # Roseway Heights Middle School
    "Skyline Elementary School": "k8",        # actually a K-8
    "Sunnyside Environmental School": "k8",
    "Vernon Elementary School": "k8",
    "Winterhaven School": "k8",
    "da Vinci Middle School": "middle",
    "Gray Middle School": "middle",
    "Dr. Martin Luther King Jr. School": "elementary",
}


def infer_level(row):
    name = row["School Name"]
    if name in LEVEL_OVERRIDES:
        return LEVEL_OVERRIDES[name]
    if row["School Type"] == "Alternative School":
        return "alternative"
    n = name.lower()
    if "high school" in n:
        return "high"
    if "middle school" in n:
        return "middle"
    if "elementary" in n:
        return "elementary"
    return "other"


def main():
    df = pd.read_excel(ODE, sheet_name="School 20252026")
    pps = df[df["District Name"] == "Portland SD 1J"].copy()

    # Keep only PPS-operated schools: Regular + Alternative.
    # Excludes charters, private, LTC, community college, district-level row.
    pps = pps[pps["School Type"].isin(["Regular School", "Alternative School"])].copy()

    pps["level"] = pps.apply(infer_level, axis=1)
    pps["is_closure_candidate"] = pps["School Name"].isin(CANDIDATES)
    pps["closure_rank"] = pps["School Name"].map(lambda n: CANDIDATES.get(n, (None, None))[0])
    pps["is_title_i"] = pps["School Name"].isin(TITLE_I)
    pps["is_urm_building"] = pps["School Name"].isin(URM_BUILDINGS)
    pps["urm_retrofit_cost_usd"] = pps["School Name"].map(URM_BUILDINGS)
    pps["seismic_retrofit_status"] = pps["School Name"].map(SEISMIC_RETROFIT)
    pps["programs"] = pps["School Name"].map(lambda n: "; ".join(PROGRAMS.get(n, [])) or None)
    pps["has_dli"] = pps["School Name"].map(
        lambda n: any("DLI" in p for p in PROGRAMS.get(n, [])))
    pps["has_focus_option"] = pps["School Name"].map(
        lambda n: any("focus" in p for p in PROGRAMS.get(n, [])))
    pps["dli_languages"] = pps["School Name"].map(
        lambda n: "; ".join(p.replace(" DLI", "") for p in PROGRAMS.get(n, []) if "DLI" in p) or None)
    pps["enrollment_pct_change"] = (
        (pps["2025-26 Total Enrollment"] - pps["2024-25 Total Enrollment"])
        / pps["2024-25 Total Enrollment"]
    ).round(4)

    # Add Odyssey Program manually — it's embedded in another PPS school in ODE data.
    odyssey = {
        "District Name": "Portland SD 1J",
        "School Institution ID": None,
        "School Name": "Odyssey Program (K-8)",
        "School Type": "Program",
        "Virtual": None,
        "2024-25 Total Enrollment": None,
        "2025-26 Total Enrollment": 217,  # from WW article
        "level": "k8",
        "is_closure_candidate": True,
        "closure_rank": 5,
        "is_title_i": False,
        "is_urm_building": False,
        "urm_retrofit_cost_usd": None,
        "seismic_retrofit_status": None,
        "enrollment_pct_change": None,
        "programs": "; ".join(PROGRAMS.get("Odyssey Program (K-8)", [])) or None,
        "has_dli": False,
        "has_focus_option": True,
        "dli_languages": None,
    }
    pps = pd.concat([pps, pd.DataFrame([odyssey])], ignore_index=True)

    # Merge NCES CCD 2022: addresses, lat/lon, FRL counts.
    with open(CCD) as f:
        ccd_raw = json.load(f)
    ccd_rows = []
    for s in ccd_raw:
        ccd_rows.append({
            "ccd_match_name": s["school_name"],
            "nces_school_id": s["ncessch"],
            "street_address": s.get("street_location"),
            "city": s.get("city_location"),
            "zip_code": s.get("zip_location"),
            "latitude": s.get("latitude"),
            "longitude": s.get("longitude"),
            "ccd_enrollment_2022": s.get("enrollment"),
            "frl_free_lunch": s.get("free_lunch"),
            "frl_reduced_lunch": s.get("reduced_price_lunch"),
            "frl_direct_certification": s.get("direct_certification"),
        })
    ccd = pd.DataFrame(ccd_rows)
    pps["ccd_match_name"] = pps["School Name"].map(lambda n: CCD_NAME_MAP.get(n, n))
    pps = pps.merge(ccd, on="ccd_match_name", how="left")
    pps = pps.drop(columns=["ccd_match_name"])

    # Patch schools missing from CCD 2022 (Clark opened post-2022; Odyssey is
    # an embedded program hosted at Hayhurst).
    for name, vals in MANUAL_LOCATION.items():
        for col, v in vals.items():
            pps.loc[pps["School Name"] == name, col] = v

    # Merge 2009 KPFF facility: year built, sq ft, construction type.
    fac = pd.read_csv(FACILITY)[[
        "school_name_2009", "year_built", "square_feet_2009", "construction_type",
    ]].rename(columns={
        "square_feet_2009": "square_feet",
        "construction_type": "construction_type_2009",
    })
    pps["fac_match_name"] = pps["School Name"].map(FACILITY_NAME_MAP)
    pps = pps.merge(fac, left_on="fac_match_name", right_on="school_name_2009", how="left")
    pps = pps.drop(columns=["fac_match_name", "school_name_2009"])

    # Derived: students per sqft (2025-26 enrollment / 2009 sqft).
    pps["students_per_sqft"] = (
        pps["2025-26 Total Enrollment"] / pps["square_feet"]
    ).round(5)

    # Merge CRDC 2020: LEP, IDEA, chronic absenteeism counts.
    with open(CRDC) as f:
        crdc = json.load(f)
    pps["crdc_lep_2020"] = pps["nces_school_id"].map(crdc["lep"])
    pps["crdc_idea_2020"] = pps["nces_school_id"].map(crdc["idea"])
    pps["crdc_chronic_absent_2020"] = pps["nces_school_id"].map(crdc["absent"])
    # Derived CRDC rates (vs 2025-26 enrollment — rough proxy, 2020 enrollment differs).
    pps["pct_lep"] = (pps["crdc_lep_2020"] / pps["2025-26 Total Enrollment"]).round(4)
    pps["pct_idea"] = (pps["crdc_idea_2020"] / pps["2025-26 Total Enrollment"]).round(4)

    # CRDC 2021 (2020-21 SY): wraparound staff FTE + suspension instances +
    # post-COVID chronic absenteeism. Same NCES key as the 2020 file.
    with open(CRDC_2021) as f:
        c21 = json.load(f)
    staff = c21["teachers_staff"]
    pps["counselors_fte_2021"] = pps["nces_school_id"].map(
        lambda n: (staff.get(str(n)) or {}).get("counselors_fte"))
    pps["social_workers_fte_2021"] = pps["nces_school_id"].map(
        lambda n: (staff.get(str(n)) or {}).get("social_workers_fte"))
    pps["psychologists_fte_2021"] = pps["nces_school_id"].map(
        lambda n: (staff.get(str(n)) or {}).get("psychologists_fte"))
    pps["nurses_fte_2021"] = pps["nces_school_id"].map(
        lambda n: (staff.get(str(n)) or {}).get("nurses_fte"))
    pps["suspensions_2021"] = pps["nces_school_id"].map(c21["suspensions_instances"])
    pps["chronic_absent_2021"] = pps["nces_school_id"].map(c21["chronic_absent"])
    pps["enrollment_crdc_2021"] = pps["nces_school_id"].map(c21["enrollment"])

    # NCES CCD historical enrollment 2018-2023 (per-school totals from the
    # directory endpoint). Combined with ODE 2024-25/2025-26 this gives an
    # 8-year window for the long-term-sustainability analysis.
    with open(CCD_HISTORY) as f:
        hist = json.load(f)
    for year_str, year_map in hist.items():
        col = f"enrollment_{year_str}"
        pps[col] = pps["nces_school_id"].map(year_map)

    # Derived FRL rates. CCD 2022 enrollment is closer in time to FRL counts
    # than 2025-26 enrollment, so use it when available.
    base_enroll = pps["ccd_enrollment_2022"].fillna(pps["2025-26 Total Enrollment"])
    pps["pct_free_lunch"] = (pps["frl_free_lunch"] / base_enroll).round(4)
    pps["pct_frl"] = ((pps["frl_free_lunch"].fillna(0) +
                       pps["frl_reduced_lunch"].fillna(0)) / base_enroll).round(4)
    pps["pct_direct_cert"] = (pps["frl_direct_certification"] / base_enroll).round(4)

    # Merge OSAS academic performance (ELA/Math proficiency, total pop).
    def osas_total(path, subject_col, year):
        df = pd.read_excel(path, sheet_name=0)
        tot = df[(df["District"] == "Portland SD 1J") &
                 (df["Student Group"] == "Total Population (All Students)")]
        out = tot[["School ID", "Percent Proficient", "Number of Participants"]].copy()
        out.columns = ["ode_school_id", f"pct_{subject_col}_prof_{year}", f"n_{subject_col}_participants_{year}"]
        # Percent column may have "<5.0%", ">95.0%", "*", "--" — coerce to numeric.
        pct_col = f"pct_{subject_col}_prof_{year}"
        out[pct_col] = pd.to_numeric(out[pct_col], errors="coerce")
        out[f"n_{subject_col}_participants_{year}"] = pd.to_numeric(
            out[f"n_{subject_col}_participants_{year}"], errors="coerce")
        return out

    # Join by ODE institution id (rename hasn't happened yet — use raw column).
    pps["_ode_int"] = pd.to_numeric(pps["School Institution ID"], errors="coerce")
    for path, subj, yr in [
        (OSAS_ELA_25, "ela", "2425"), (OSAS_MATH_25, "math", "2425"),
        (OSAS_ELA_24, "ela", "2324"), (OSAS_MATH_24, "math", "2324"),
    ]:
        o = osas_total(path, subj, yr).rename(columns={"ode_school_id": "_ode_int"})
        o["_ode_int"] = pd.to_numeric(o["_ode_int"], errors="coerce")
        pps = pps.merge(o, on="_ode_int", how="left")
    pps = pps.drop(columns=["_ode_int"])

    # Rename columns to snake_case for easier downstream use.
    rename = {
        "School Institution ID": "ode_school_id",
        "School Name": "school_name",
        "School Type": "school_type",
        "Virtual": "virtual",
        "2024-25 Total Enrollment": "enrollment_2024_25",
        "2025-26 Total Enrollment": "enrollment_2025_26",
        "2025-26 American Indian/ Alaska Native": "pct_ai_an_count",
        "2025-26 % American Indian/ Alaska Native": "pct_ai_an",
        "2025-26 Asian": "asian_count",
        "2025-26 % Asian": "pct_asian",
        "2025-26 Native Hawaiian/ Pacific Islander": "nhpi_count",
        "2025-26 % Native Hawaiian/ Pacific Islander": "pct_nhpi",
        "2025-26 Black/ African American": "black_count",
        "2025-26 % Black/ African American": "pct_black",
        "2025-26 Hispanic/ Latino": "hispanic_count",
        "2025-26 % Hispanic/ Latino": "pct_hispanic",
        "2025-26 White": "white_count",
        "2025-26 % White": "pct_white",
        "2025-26 Multi-Racial": "multiracial_count",
        "2025-26 % Multi-Racial": "pct_multiracial",
    }
    pps = pps.rename(columns=rename)

    cols = [
        "school_name", "level", "school_type", "ode_school_id", "nces_school_id",
        "street_address", "city", "zip_code", "latitude", "longitude",
        "is_closure_candidate", "closure_rank", "is_title_i",
        "is_urm_building", "urm_retrofit_cost_usd", "seismic_retrofit_status",
        "programs", "has_dli", "dli_languages", "has_focus_option",
        "year_built", "square_feet", "construction_type_2009", "students_per_sqft",
        "enrollment_2024_25", "enrollment_2025_26", "enrollment_pct_change",
        "enrollment_2018", "enrollment_2019", "enrollment_2020",
        "enrollment_2021", "enrollment_2022", "enrollment_2023",
        "ccd_enrollment_2022",
        "pct_ela_prof_2425", "pct_math_prof_2425",
        "pct_ela_prof_2324", "pct_math_prof_2324",
        "n_ela_participants_2425", "n_math_participants_2425",
        "crdc_lep_2020", "crdc_idea_2020", "crdc_chronic_absent_2020",
        "pct_lep", "pct_idea",
        "counselors_fte_2021", "social_workers_fte_2021",
        "psychologists_fte_2021", "nurses_fte_2021",
        "suspensions_2021", "chronic_absent_2021", "enrollment_crdc_2021",
        "frl_free_lunch", "frl_reduced_lunch", "frl_direct_certification",
        "pct_free_lunch", "pct_frl", "pct_direct_cert",
        "pct_ai_an", "pct_asian", "pct_nhpi", "pct_black",
        "pct_hispanic", "pct_white", "pct_multiracial",
    ]
    pps = pps[cols].sort_values(["is_closure_candidate", "closure_rank", "school_name"],
                                ascending=[False, True, True])

    # Append schools that don't exist in ODE (embedded/alternative programs).
    rename_map = {"School Name": "school_name"}
    for stub in SUPPLEMENTAL_SCHOOLS:
        row = {rename_map.get(k, k): v for k, v in stub.items()}
        if row["school_name"] in pps["school_name"].values:
            continue
        pps = pd.concat([pps, pd.DataFrame([{c: row.get(c) for c in pps.columns}])],
                        ignore_index=True)

    pps.to_csv(OUT, index=False)
    print(f"Wrote {len(pps)} rows to {OUT}")
    print(f"  Closure candidates: {pps['is_closure_candidate'].sum()}")
    print(f"  Title I schools: {pps['is_title_i'].sum()}")
    candidates_ti = pps[pps['is_closure_candidate'] & pps['is_title_i']]['school_name'].tolist()
    print(f"  Candidates that are Title I ({len(candidates_ti)}): {candidates_ti}")
    print(f"  By level: {pps['level'].value_counts().to_dict()}")

    # Sanity check candidate enrollment matches.
    print("\nCandidate enrollment sanity check (ODE vs WW article):")
    for name, (rank, ww_n) in sorted(CANDIDATES.items(), key=lambda x: x[1][0]):
        row = pps[pps["school_name"] == name]
        if row.empty:
            print(f"  #{rank:>2} {name}: NOT FOUND")
            continue
        ode_n = row["enrollment_2025_26"].iloc[0]
        flag = "✓" if abs(ode_n - ww_n) <= 10 else "!"
        print(f"  #{rank:>2} {name}: ODE={ode_n} WW={ww_n} {flag}")


if __name__ == "__main__":
    main()
