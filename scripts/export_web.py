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
    "is_closure_candidate": {"label": "Low enrollment (WW)", "desc": "One of the 15 lowest-enrollment schools listed by Willamette Week (2026-03-18). Often discussed as a closure shortlist, but PPS has not published its own list; that is expected Nov 2026, with a board vote in Dec 2026.", "source": "Willamette Week", "fmt": "bool"},
    "closure_rank": {"label": "WW low-enroll. rank", "desc": "Willamette Week's enrollment-based rank within the bottom 15 (1 = smallest). Not a PPS rank or prediction.", "source": "Willamette Week", "fmt": "int"},
    "enrollment_2025_26": {"label": "Enrollment 25-26", "desc": "Total students, fall 2025.", "source": "Oregon ODE Fall Membership", "fmt": "int"},
    "enrollment_2024_25": {"label": "Enrollment 24-25", "desc": "Total students, fall 2024.", "source": "Oregon ODE Fall Membership", "fmt": "int"},
    "enrollment_pct_change": {"label": "Enrollment Δ% YoY", "desc": "Year-over-year enrollment change (2024-25 → 2025-26).", "source": "Derived from ODE", "fmt": "pct_0_1"},
    "functional_capacity_2021": {"label": "Functional capacity", "desc": "PPS's planning number for how many students a building can educate: gross capacity (classrooms × student-station size by area) minus set-asides (SPED focus, art/music, computer lab, DLI co-location) and reduced by a configuration-specific utilization rate (85% at middle/high, higher at K-5/K-8), with further reductions for Title I and TSI/CSI schools. From the 2021 Long-Range Facility Plan.", "source": "PPS Long-Range Facility Plan 2021 (Vol 1)", "fmt": "int"},
    "utilization_pct_2526": {"label": "Utilization 25-26", "desc": "2025-26 enrollment ÷ 2021 functional capacity. Below 50% is substantially underused; above 100% is overcrowded per PPS's own planning standard.", "source": "Derived: ODE 2025-26 ÷ LRFP 2021", "fmt": "pct_0_1"},
    "year_built": {"label": "Year built", "desc": "Year the main building was constructed.", "source": "KPFF Seismic Report 2009", "fmt": "year"},
    "square_feet": {"label": "Square feet", "desc": "Building square footage (2009 inventory; may predate recent bond expansions).", "source": "KPFF Seismic Report 2009", "fmt": "int"},
    "construction_type_2009": {"label": "Construction", "desc": "Structural type: URM (unreinforced masonry), LRCW (reinforced concrete wall), Wood, Concrete, Steel, Masonry.", "source": "KPFF 2009", "fmt": "text"},
    "is_urm_building": {"label": "URM?", "desc": "Unreinforced masonry: highest seismic risk category.", "source": "Holmes Engineering 2024, via WW 2025-05-10", "fmt": "bool"},
    "urm_retrofit_cost_usd": {"label": "URM retrofit $", "desc": "Estimated cost of a partial retrofit targeting just the unreinforced-masonry areas (URM schools only). Meant as a lower-cost option vs. full-building retrofit.", "source": "Holmes Engineering 2024", "fmt": "usd"},
    "retrofit_cost_remaining_usd": {"label": "Retrofit cost remaining", "desc": "Holmes 2024 rough-order-of-magnitude (ROM) cost to fully retrofit every building on this campus. Schools shown as $0 were excluded from Holmes's scope due to recent or near-complete modernization (Benson, Cleveland, Franklin, Grant, Jefferson, Kellogg, Lent, Lewis, Lincoln, McDaniel, Roosevelt, Rosa Parks, Faubion, Ida B Wells, Creative Science/Clark).", "source": "Holmes Engineering 2024", "fmt": "usd"},
    "seismic_retrofit_status": {"label": "Seismic retrofit", "desc": "'full' = full modernization, 'targeted' = roof/partial retrofit done, 'planned_*' = 2025 bond scheduled, blank = none.", "source": "PPS bond.pps.net/seismic-improvements", "fmt": "text"},
    "recent_boundary_change": {"label": "Recent boundary change", "desc": "Schools whose attendance boundary or grade configuration was meaningfully redrawn inside the 2018→2025 window. 'DBRAC 2018' = affected by the Districtwide Boundary Review (new middle-school feeders, K-8→K-5 conversions, NE/N redraws). 'SEGC 2023' = affected by the Southeast Guiding Coalition (Harrison Park and Bridger reconfigured, Lent/Marysville/Clark feeder redraws). Raw enrollment trends for these rows partly reflect catchment redraws, not pure demand change.", "source": "Curated from PPS DBRAC + SEGC implementation documentation", "fmt": "text"},
    "is_title_i": {"label": "Title I?", "desc": "Receives federal Title I-A schoolwide funding (high-poverty designation).", "source": "PPS Funded Programs 2025-26", "fmt": "bool"},
    "pct_regular_attenders_2425": {"label": "% Regular attenders 24-25", "desc": "Share of students who attended more than 90% of their enrolled days in 2024-25. Oregon's core attendance metric (students not chronically absent). Statewide average was 66.5% in 2024-25.", "source": "Oregon ODE At-A-Glance 2024-25", "fmt": "pct_0_100"},
    "pct_experienced_teachers_2425": {"label": "% Experienced teachers 24-25", "desc": "Share of teachers at this school with 3+ years of experience in education (any district). A higher value usually means a more stable, veteran faculty.", "source": "Oregon ODE At-A-Glance 2024-25", "fmt": "pct_0_100"},
    "pct_teacher_retention_2425": {"label": "% Teacher retention 24-25", "desc": "Share of teachers who returned from the previous school year; a turnover indicator. Low retention often reflects instability or chronic understaffing.", "source": "Oregon ODE At-A-Glance 2024-25", "fmt": "pct_0_100"},
    "class_size_2425": {"label": "Median class size 24-25", "desc": "ODE-reported median class size (all self-contained classes in 2024-25). Oregon statewide median is 24.", "source": "Oregon ODE At-A-Glance 2024-25", "fmt": "int"},
    "pct_ela_prof_2425": {"label": "% ELA prof 24-25", "desc": "Percent of students meeting/exceeding on Oregon ELA state test (all grades, all students).", "source": "Oregon OSAS 2024-25", "fmt": "pct_0_100"},
    "pct_math_prof_2425": {"label": "% Math prof 24-25", "desc": "Percent of students meeting/exceeding on Oregon Math state test (all grades, all students).", "source": "Oregon OSAS 2024-25", "fmt": "pct_0_100"},
    "pct_ela_prof_2324": {"label": "% ELA prof 23-24", "desc": "Prior-year ELA proficiency.", "source": "Oregon OSAS 2023-24", "fmt": "pct_0_100"},
    "pct_math_prof_2324": {"label": "% Math prof 23-24", "desc": "Prior-year Math proficiency.", "source": "Oregon OSAS 2023-24", "fmt": "pct_0_100"},
    "crdc_chronic_absent_2020": {"label": "Chronic absent (CRDC)", "desc": "Count of students chronically absent (missed ≥15 days). Based on 2020 COVID year; likely suppressed.", "source": "US Dept of Ed CRDC 2020", "fmt": "int"},
    "crdc_lep_2020": {"label": "LEP students", "desc": "English Learner (Limited English Proficient) count.", "source": "US Dept of Ed CRDC 2020", "fmt": "int"},
    "crdc_idea_2020": {"label": "IDEA (SPED) students", "desc": "Students on IEPs under IDEA (special education).", "source": "US Dept of Ed CRDC 2020", "fmt": "int"},
    "frl_free_lunch": {"label": "Free lunch (count)", "desc": "Students eligible for free meals (raw count).", "source": "NCES CCD 2022", "fmt": "int"},
    "frl_reduced_lunch": {"label": "Reduced lunch (count)", "desc": "Students eligible for reduced-price meals (raw count).", "source": "NCES CCD 2022", "fmt": "int"},
    "pct_free_lunch": {"label": "% Free lunch", "desc": "Share of students on free meals (free ÷ 2022 enrollment); proxy for poverty.", "source": "Derived: NCES CCD 2022", "fmt": "pct_0_1"},
    "pct_frl": {"label": "% FRL", "desc": "Share of students eligible for free OR reduced-price meals; standard poverty proxy.", "source": "Derived: NCES CCD 2022", "fmt": "pct_0_1"},
    "pct_direct_cert": {"label": "% Direct cert", "desc": "Share directly certified for free meals via SNAP/TANF/foster; proxy for deep poverty.", "source": "Derived: NCES CCD 2022", "fmt": "pct_0_1"},
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
    "support_staff_per_100": {"label": "Support staff per 100", "desc": "Counselor + social worker + psychologist FTE, per 100 students (denominator: 2020-21 CRDC enrollment). Reflects 2020-21 staffing; base allocations make smaller schools score higher per-pupil.", "source": "Derived: CRDC 2021 staff ÷ 2021 enrollment", "fmt": "ratio"},
    "pct_chronic_absent_2021": {"label": "% chronic absent (21)", "desc": "Share of students who missed ≥10% of enrolled days in the 2020-21 school year. Post-COVID but still pandemic-era; treat as floor on current rates. Clamped at 100%; a few alternative schools' raw CRDC counts exceed point-in-time enrollment due to mid-year mobility.", "source": "Derived: CRDC 2021 ÷ CRDC 2021 enrollment", "fmt": "pct_0_1"},
    "suspensions_per_100_2021": {"label": "OSS per 100", "desc": "Out-of-school suspension instances per 100 students (2020-21). Counts events not students; one student can be suspended multiple times.", "source": "Derived: CRDC 2021 ÷ CRDC 2021 enrollment", "fmt": "ratio"},
    "counselors_fte_2021": {"label": "Counselor FTE", "desc": "Full-time-equivalent school counselors as reported to CRDC for 2020-21.", "source": "CRDC 2021", "fmt": "ratio"},
    "social_workers_fte_2021": {"label": "Social worker FTE", "desc": "Full-time-equivalent school social workers as reported to CRDC for 2020-21.", "source": "CRDC 2021", "fmt": "ratio"},
    "teachers_fte_2023": {"label": "Teacher FTE (2023)", "desc": "Classroom teacher full-time-equivalents as reported to NCES for fall 2023. Excludes support staff (counselors, social workers, nurses) and non-instructional roles.", "source": "NCES CCD 2023", "fmt": "ratio"},
    "students_per_teacher_2023": {"label": "Students per teacher (2023)", "desc": "Same-year ratio: fall 2023 NCES enrollment ÷ fall 2023 NCES teacher FTE. PPS in-scope median is ~17; small alternative and K-8 programs skew low, DLI and large elementaries skew high.", "source": "Derived: NCES CCD 2023", "fmt": "ratio"},
    "teachers_fte_2021": {"label": "Teacher FTE (2021)", "desc": "Classroom teacher full-time-equivalents as reported to CRDC for 2020-21. COVID-era snapshot.", "source": "CRDC 2021", "fmt": "ratio"},
    "students_per_teacher_2021": {"label": "Students per teacher (2021)", "desc": "Same-year ratio: 2020-21 CRDC enrollment ÷ 2020-21 CRDC teacher FTE. COVID-era snapshot; shown alongside 2023 for trend context.", "source": "Derived: CRDC 2021", "fmt": "ratio"},
    "airflow_rooms_tested": {"label": "Rooms airflow-tested", "desc": "Count of classrooms and shared spaces airflow-tested by Amerseco/Neudorfer in the PPS 2021 NEBB-certified survey. Coverage is typically 20-40 rooms per elementary and 80-150 per high school.", "source": "PPS 2021 airflow survey", "fmt": "int"},
    "airflow_ach_e_median": {"label": "ACH_e median (2021)", "desc": "Median ASHRAE Total Effective Air Changes per Hour without portable filtration — the rate at which a room's air is fully cycled through HVAC filtration and outside air dilution. Lancet COVID-era guidance recommends ≥6 ACH for classrooms; ≥3 ACH is a bare-minimum floor. Measured 2021; includes MERV-13 upgrades done by then.", "source": "PPS 2021 airflow survey", "fmt": "ratio"},
    "airflow_pct_below_3_ach": {"label": "% rooms < 3 ACH_e", "desc": "Share of tested rooms with effective ACH below 3 — the lower-bound floor in Lancet COVID ventilation guidance. Districtwide median is near 60%; buildings without filter upgrades are much higher.", "source": "PPS 2021 airflow survey", "fmt": "pct_0_1"},
    "airflow_pct_below_6_ach": {"label": "% rooms < 6 ACH_e", "desc": "Share of tested rooms below the Lancet-recommended 6 ACH benchmark for classrooms. Most PPS rooms fall below this standard.", "source": "PPS 2021 airflow survey", "fmt": "pct_0_1"},
    "airflow_filter_upgraded": {"label": "MERV-13 upgrade?", "desc": "Whether a MERV-13 filter upgrade had been installed in this building's HVAC as of the 2021 airflow survey. Filter upgrades raise effective ACH even without added outside air.", "source": "PPS 2021 airflow survey", "fmt": "bool"},
    "prof_residual": {"label": "Proficiency residual", "desc": "Each school's avg proficiency minus what a linear fit on % BIPOC would predict. Positive = outperforms demographics; negative = underperforms.", "source": "Derived: OSAS 2024-25 + ODE", "fmt": "ratio"},
    "affordable_units_within_1mi": {"label": "Afford. units in catchment", "desc": "Total existing affordable housing units inside the school's PPS attendance area (or a 1-mile radius for schools without a published catchment).", "source": "OAHI + Metro RLIS", "fmt": "int"},
    "pipeline_affordable_units_within_1mi": {"label": "Pipeline afford. units", "desc": "Affordable units in projects currently in development inside the school's PPS attendance area (2023–2027).", "source": "OAHI", "fmt": "int"},
    "pipeline_family_units_within_1mi": {"label": "Pipeline family units", "desc": "2+BR pipeline units inside the school's PPS attendance area; proxy for future families with kids.", "source": "OAHI", "fmt": "int"},
    "n_pipeline_projects_within_1mi": {"label": "Pipeline projects", "desc": "Number of affordable housing projects in development inside the school's PPS attendance area.", "source": "OAHI", "fmt": "int"},
    "permits_units_within_1mi_since_2022": {"label": "Permitted units (2022+)", "desc": "New residential units on building permits issued since 2022-01-01 inside the school's PPS attendance area (single-family, ADUs, and multifamily) (all tenures). Schools without a published catchment fall back to a 1-mile radius.", "source": "Portland BDS via PortlandMaps", "fmt": "int"},
    "n_permits_within_1mi_since_2022": {"label": "Permits (2022+)", "desc": "Number of residential building permits issued since 2022-01-01 inside the school's PPS attendance area. Permits = approved to build; not all reach completion.", "source": "Portland BDS via PortlandMaps", "fmt": "int"},
    "bli_forecast_units_within_catchment": {"label": "BLI forecast units (~2035)", "desc": "Projected new residential units by ~2035 inside the school's PPS attendance area, from Metro's 2045 Building Land Inventory (BLI) Housing-Employment Allocation grid. Area-weighted sum over intersecting BLI grid cells; schools without a catchment fall back to a 1-mile buffer around the school's point location. A forward-looking counterpart to the (backwards-looking) permits metric.", "source": "Metro 2045 BLI (Portland open data)", "fmt": "int"},
    "nearest_alt_school_mi": {"label": "Miles to nearest alt.", "desc": "Great-circle distance from this low-enrollment school to the nearest larger same-grade-band school (elementary, k8, middle, or alternative). A rough proxy for transportation impact if this school were closed; actual PPS reassignment may differ.", "source": "Derived from NCES CCD coordinates", "fmt": "miles"},
    "nearest_alt_school_name": {"label": "Nearest alt. school", "desc": "Name of the closest larger school of the same grade band.", "source": "Derived from NCES CCD coordinates", "fmt": "text"},
    "enrollment_2018": {"label": "Enrollment 2018", "desc": "Fall 2018 enrollment (NCES CCD directory).", "source": "NCES CCD 2018", "fmt": "int"},
    "programs": {"label": "Programs", "desc": "Specialized programs hosted at this school. Captures Dual Language Immersion (DLI) language tracks plus K-8 focus options (Arts, Environmental, STEAM, Creative Science, TAG, Alternative). 'Access to programs' is one of the four PPS-announced closure-decision factors; closing a DLI host school would cut off that language pathway in its catchment.", "source": "PPS Dual Language + Enrollment & Transfer pages", "fmt": "text"},
    "has_dli": {"label": "Hosts DLI?", "desc": "School hosts a Dual Language Immersion program in any language.", "source": "Derived from PPS DLI program list", "fmt": "bool"},
    "dli_languages": {"label": "DLI languages", "desc": "Dual Language Immersion language(s) hosted at this school (Spanish, Chinese, Japanese, Russian, Vietnamese).", "source": "PPS Dual Language pages", "fmt": "text"},
    "has_focus_option": {"label": "Hosts focus option?", "desc": "School is one of the K-8 focus-option / specialized-curriculum programs (Buckman Arts, Bridger Creative Science, Sunnyside Environmental, Winterhaven STEAM, Odyssey TAG, ACCESS, MLC).", "source": "PPS Enrollment & Transfer", "fmt": "bool"},
    "dli_students_2526": {"label": "DLI students 25-26", "desc": "Students enrolled in this school's Dual Language Immersion strand(s), October 2025. Populated only for the ~32 schools that host a DLI program. A school's total enrollment is divided between neighborhood/mainstream students and DLI-strand students; ODE's Fall Membership cannot separate them.", "source": "PPS Language Immersion Enrollment Report 2025-26", "fmt": "int"},
    "neighborhood_students_2526": {"label": "Neighborhood students 25-26", "desc": "Total enrollment minus DLI-strand enrollment. At a DLI host school this is the mainstream/English-medium population; the cohort that would remain if the DLI program were relocated.", "source": "Derived: ODE − PPS Immersion Report", "fmt": "int"},
    "pct_dli_2526": {"label": "% DLI 25-26", "desc": "Share of this school's students in a DLI strand. Ranges from ~4% at schools where a single DLI cohort visits (Jefferson) to 100% at whole-school DLI programs (Lent, Rigler, Richmond).", "source": "Derived: PPS Immersion Report ÷ ODE", "fmt": "pct_0_1"},
    "enrollment_pct_change_7yr": {"label": "Enrollment Δ% 2018→2025", "desc": "7-year enrollment change from fall 2018 (NCES CCD) to fall 2025 (ODE). The in-scope set lost 20% over this window (median school -19%); schools well below that baseline are losing students faster than PPS as a whole.", "source": "Derived: NCES CCD 2018 + ODE 2025-26", "fmt": "pct_0_1"},
    "prc_baseline_2024_25": {"label": "PRC baseline 24-25", "desc": "PRC's own historic 2024-25 enrollment used as the base year for its 10-year forecast (Table 5.5, medium scenario). Usually matches ODE 2024-25, but differs at co-located programs — e.g. Hayhurst's PRC baseline is Hayhurst-proper only and excludes Odyssey K-8.", "source": "PSU Population Research Center 2025 (Table 5.5)", "fmt": "int"},
    "enrollment_forecast_2025_26": {"label": "Forecast 25-26 (med.)", "desc": "PRC medium-scenario enrollment forecast for 2025-26 (per-school, all programs combined).", "source": "PSU Population Research Center 2025 (Table 5.5)", "fmt": "int"},
    "enrollment_forecast_2026_27": {"label": "Forecast 26-27 (med.)", "desc": "PRC medium-scenario enrollment forecast for 2026-27.", "source": "PSU Population Research Center 2025 (Table 5.5)", "fmt": "int"},
    "enrollment_forecast_2027_28": {"label": "Forecast 27-28 (med.)", "desc": "PRC medium-scenario enrollment forecast for 2027-28.", "source": "PSU Population Research Center 2025 (Table 5.5)", "fmt": "int"},
    "enrollment_forecast_2028_29": {"label": "Forecast 28-29 (med.)", "desc": "PRC medium-scenario enrollment forecast for 2028-29.", "source": "PSU Population Research Center 2025 (Table 5.5)", "fmt": "int"},
    "enrollment_forecast_2029_30": {"label": "Forecast 29-30 (med.)", "desc": "PRC medium-scenario enrollment forecast for 2029-30.", "source": "PSU Population Research Center 2025 (Table 5.5)", "fmt": "int"},
    "enrollment_forecast_2030_31": {"label": "Forecast 30-31 (med.)", "desc": "PRC medium-scenario enrollment forecast for 2030-31 (5-year horizon).", "source": "PSU Population Research Center 2025 (Table 5.5)", "fmt": "int"},
    "enrollment_forecast_2031_32": {"label": "Forecast 31-32 (med.)", "desc": "PRC medium-scenario enrollment forecast for 2031-32.", "source": "PSU Population Research Center 2025 (Table 5.5)", "fmt": "int"},
    "enrollment_forecast_2032_33": {"label": "Forecast 32-33 (med.)", "desc": "PRC medium-scenario enrollment forecast for 2032-33.", "source": "PSU Population Research Center 2025 (Table 5.5)", "fmt": "int"},
    "enrollment_forecast_2033_34": {"label": "Forecast 33-34 (med.)", "desc": "PRC medium-scenario enrollment forecast for 2033-34.", "source": "PSU Population Research Center 2025 (Table 5.5)", "fmt": "int"},
    "enrollment_forecast_2034_35": {"label": "Forecast 34-35 (med.)", "desc": "PRC medium-scenario enrollment forecast for 2034-35 (10-year horizon). The closure argument hinges on future utilization, not current — this is the per-school number PRC published.", "source": "PSU Population Research Center 2025 (Table 5.5)", "fmt": "int"},
    "enrollment_forecast_2034_35_low": {"label": "Forecast 34-35 (low)", "desc": "Low-scenario 2034-35 forecast. PRC only publishes Low/High at the district-and-grade level, so each school's band is the medium forecast × the district-wide Low/Medium ratio at its grade band (K-5 elementaries: ×0.869; 6-8 middle: ×0.859).", "source": "Derived: PRC Tables 5.3/5.5 (low × med ratio by grade band)", "fmt": "int"},
    "enrollment_forecast_2034_35_high": {"label": "Forecast 34-35 (high)", "desc": "High-scenario 2034-35 forecast. PRC only publishes Low/High at the district-and-grade level, so each school's band is the medium forecast × the district-wide High/Medium ratio at its grade band (K-5 elementaries: ×1.221; 6-8 middle: ×1.075).", "source": "Derived: PRC Tables 5.4/5.5 (high × med ratio by grade band)", "fmt": "int"},
    "enrollment_forecast_pct_change_10yr": {"label": "Forecast Δ% 10yr (24-25→34-35)", "desc": "Projected 10-year enrollment change from PRC's 2024-25 baseline to its 2034-35 medium-scenario forecast. Uses PRC's own historic baseline (not ODE) so co-located programs like Odyssey/Hayhurst net out correctly.", "source": "Derived: PRC 2025 medium scenario", "fmt": "pct_0_1"},
    "utilization_pct_2034_35": {"label": "Forecast utilization 34-35", "desc": "Projected 2034-35 enrollment (PRC medium) ÷ 2021 functional capacity. A forward-looking counterpart to current utilization. Below 50% signals deeply underused by PPS's own planning standard — even if today's utilization is higher.", "source": "Derived: PRC 2025 ÷ LRFP 2021", "fmt": "pct_0_1"},
    "street_address": {"label": "Address", "desc": "Street address of the building.", "source": "NCES CCD + manual", "fmt": "text"},
    "latitude": {"label": "Latitude", "desc": "Geocoded latitude.", "source": "NCES CCD", "fmt": "text"},
    "longitude": {"label": "Longitude", "desc": "Geocoded longitude.", "source": "NCES CCD", "fmt": "text"},
}

# Columns to surface in the default table (order matters).
TABLE_COLS = [
    "school_name", "level",
    "enrollment_2025_26", "enrollment_pct_change",
    "functional_capacity_2021", "utilization_pct_2526",
    "year_built", "square_feet", "pct_ela_prof_2425", "pct_math_prof_2425",
    "is_urm_building", "retrofit_cost_remaining_usd", "seismic_retrofit_status",
    "is_title_i", "pct_bipoc",
    "enrollment_pct_change_7yr",
    "enrollment_forecast_2034_35", "enrollment_forecast_pct_change_10yr",
    "utilization_pct_2034_35",
    "recent_boundary_change", "programs",
    "has_dli", "pct_dli_2526", "pct_chronic_absent_2021", "support_staff_per_100",
    "pct_regular_attenders_2425", "pct_experienced_teachers_2425",
    "pct_teacher_retention_2425", "class_size_2425", "students_per_teacher_2023",
    "airflow_ach_e_median", "airflow_pct_below_3_ach",
    "pipeline_family_units_within_1mi", "affordable_units_within_1mi",
    "permits_units_within_1mi_since_2022",
]

# Pre-defined scatter plots.
SCATTERS = [
    {
        "id": "enrollment_vs_utilization",
        "title": "Enrollment vs. building utilization",
        "x": "enrollment_2025_26",
        "y": "utilization_pct_2526",
        "subtitle": "Utilization = current enrollment ÷ 2021 functional capacity. The closure shortlist (low-enrollment schools) clusters in the lower-left quadrant of underused buildings. Horizontal line at 100% marks PPS's own planning capacity.",
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
        "id": "enrollment_vs_bipoc",
        "title": "Enrollment vs. % BIPOC students",
        "x": "enrollment_2025_26",
        "y": "pct_bipoc",
        "subtitle": "Across PPS, smaller-enrollment schools do not serve a disproportionately BIPOC student body; the cloud is roughly flat.",
    },
    {
        "id": "chronic_absent_vs_prof",
        "title": "Chronic absenteeism vs. avg proficiency (2020-21)",
        "x": "pct_chronic_absent_2021",
        "y": "avg_prof_2425",
        "subtitle": "Strong inverse relationship: schools where 1 in 3+ students were chronically absent post-COVID rarely exceed 30% proficiency now.",
        "trendline": True,
    },
    {
        "id": "support_staff_vs_direct_cert",
        "title": "Student-support staff vs. deep poverty",
        "x": "pct_direct_cert",
        "y": "support_staff_per_100",
        "subtitle": "Counselor + social worker + psychologist FTE per 100 students (CRDC 2020-21). Base allocations mean small schools score higher per-pupil by default; read with enrollment in mind. A high-poverty school well below the cloud has less per-pupil support than its peers. Alliance High School excluded as a staffing outlier.",
        "trendline": True,
        "exclude": ["Alliance High School"],
    },
    {
        "id": "enrollment_2018_vs_change",
        "title": "7-year enrollment change vs. current size (2018 → 2025)",
        "x": "enrollment_2025_26",
        "y": "enrollment_pct_change_7yr",
        "subtitle": "Long-term sustainability check. The in-scope set as a whole shrank ~20% over seven years; schools well below that line are losing students faster than the district overall.",
    },
    {
        "id": "forecast_change_vs_size",
        "title": "Projected 10-year enrollment change vs. current size",
        "x": "enrollment_2025_26",
        "y": "enrollment_forecast_pct_change_10yr",
        "subtitle": "PSU's Population Research Center projects districtwide enrollment to fall from ~43k to ~37k by 2034-35 (medium scenario). Schools in the lower-left — already small today and projected to keep shrinking — are the ones where future-utilization pressure is most acute.",
    },
    {
        "id": "forecast_utilization_vs_current",
        "title": "Current vs. projected 2034-35 utilization",
        "x": "utilization_pct_2526",
        "y": "utilization_pct_2034_35",
        "subtitle": "How underused each building gets by the end of PRC's 10-year forecast. The 45° line means 'holding steady'; below it means the building empties further. Below 50% is PPS's own threshold for 'substantially underused'.",
    },
    {
        "id": "ratio_vs_enrollment",
        "title": "Student/teacher ratio vs. enrollment",
        "x": "enrollment_2025_26",
        "y": "students_per_teacher_2023",
        "subtitle": "Smaller schools sit lower because teacher FTE has a floor (a K-5 still needs one teacher per grade level regardless of section size). Schools well above the trend are stretched thin for their size; schools well below are staffed unusually richly. Ratio uses fall 2023 NCES; enrollment is fall 2025 ODE for consistency with other scatters.",
        "trendline": True,
    },
]

# Features used for k-means clustering + PCA projection.
CLUSTER_FEATURES = [
    "enrollment_2025_26", "enrollment_pct_change", "utilization_pct_2526",
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
    # Student-experience metrics derived from CRDC 2021 (2020-21 SY).
    enr21 = df["enrollment_crdc_2021"]
    valid_enr = enr21.notna() & (enr21 > 0)
    df["pct_chronic_absent_2021"] = pd.NA
    # Clamp at 100%: alternative schools have mid-year mobility so cumulative
    # CRDC counts can exceed point-in-time enrollment.
    df.loc[valid_enr, "pct_chronic_absent_2021"] = (
        (df.loc[valid_enr, "chronic_absent_2021"] / enr21[valid_enr]).clip(upper=1.0)
    ).round(4)
    df["suspensions_per_100_2021"] = pd.NA
    df.loc[valid_enr, "suspensions_per_100_2021"] = (
        df.loc[valid_enr, "suspensions_2021"] / enr21[valid_enr] * 100
    ).round(2)
    support_cols = ["counselors_fte_2021", "social_workers_fte_2021", "psychologists_fte_2021"]
    support_total = df[support_cols].sum(axis=1, min_count=1)
    df["support_staff_per_100"] = pd.NA
    df.loc[valid_enr, "support_staff_per_100"] = (
        support_total[valid_enr] / enr21[valid_enr] * 100
    ).round(3)

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

    # 7-year enrollment trend: 2018 (CCD) -> 2025-26 (ODE).
    base = df["enrollment_2018"]
    cur = df["enrollment_2025_26"]
    valid = base.notna() & cur.notna() & (base > 0)
    df["enrollment_pct_change_7yr"] = pd.NA
    df.loc[valid, "enrollment_pct_change_7yr"] = (
        (cur[valid] - base[valid]) / base[valid]
    ).round(4)

    # Forward-looking utilization: PRC 2034-35 medium ÷ 2021 functional capacity.
    if "enrollment_forecast_2034_35" in df.columns:
        fut = df["enrollment_forecast_2034_35"]
        cap = df["functional_capacity_2021"]
        uvalid = fut.notna() & cap.notna() & (cap > 0)
        df["utilization_pct_2034_35"] = pd.NA
        df.loc[uvalid, "utilization_pct_2034_35"] = (
            fut[uvalid] / cap[uvalid]
        ).round(4)

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
