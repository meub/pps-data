#!/usr/bin/env python3
"""Write web/budget.json for the budget explainer (web/budget.html).

District-level figures are typed in below from the 2026-27 Adopted Budget,
Volume 1 (data/raw/pps_budget_2026-27_adopted_vol1.pdf). Its tables do not
survive text extraction, so each constant cites the printed page it came
from. All dollar amounts are in thousands, as printed.

Per-school figures come from parse_budget_school_reports.py (Volume 2).
School names are joined to the master CSV so the page can mark schools in
PPS's rightsizing scenarios.

The script checks that the grouped General Fund uses add up to the printed
fund total, so a typo in a constant fails loudly.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCHOOLS_JSON = ROOT / "data" / "raw" / "pps_budget_school_reports.json"
MASTER = ROOT / "data" / "pps_schools.csv"
OUT = ROOT / "web" / "budget.json"

SOURCES = {
    "vol1": {
        "title": "2026-27 Adopted Budget, Volume 1",
        "publisher": "Portland Public Schools",
        "url": "https://www.pps.net/departments/budget",
    },
    "vol2": {
        "title": "2026-27 Proposed Budget, Volume 2: Individual School Reports (updated May 6, 2026)",
        "publisher": "Portland Public Schools",
        "url": "https://www.pps.net/departments/budget",
    },
}

GENERAL_FUND_TOTAL = 862_112  # Vol 1 p. 103, 2026-27 adopted

# General Fund resources, 2026-27 adopted. Vol 1 p. 103.
GF_RESOURCES = [
    ("Local property taxes", 358_223,
     "PPS's permanent tax rate of $5.28 per $1,000 of assessed value."),
    ("State School Fund and other state", 317_100,
     "Oregon's per-student school funding, weighted for students with higher needs."),
    ("Local option levy", 103_938,
     "A voter-approved extra property tax ($1.99 per $1,000), renewed in May 2024."),
    ("Carryover from last year", 41_000,
     "Money left at the end of 2025-26 (the beginning fund balance)."),
    ("Other local revenue", 28_701,
     "Fees, rentals, interest, and similar local income."),
    ("County education service district", 9_791,
     "Pass-through money from the Multnomah Education Service District."),
    ("Other", 3_359,
     "Federal money, transfers from other funds, and other small sources."),
]

# General Fund uses by function code, 2026-27 adopted. Vol 1 pp. 107-108.
# Two lines are negative (1100, 2690): district-wide adjustments PPS books
# against those functions. They fold into their groups here.
GF_USES = {
    "Instruction": {
        # Regular classroom programs by grade level (teachers, aides, subs,
        # their benefits, and classroom supplies). 1100's district-wide
        # adjustment is spread over these three in grouped_uses().
        "Elementary classrooms (K-5)": [("1111", 144_164), ("1113", 106), ("1140", 195)],
        "Middle school classrooms": [("1121", 75_828)],
        "High school classrooms": [("1131", 103_134)],
        "Athletics and activities (middle and high)": [("1122", 2_073), ("1132", 7_330)],
        "Special education": [("1220", 38_899), ("1250", 40_136), ("1260", 773)],
        "Alternative education": [("1280", 30_807)],
        "English language learners": [("1291", 15_450)],
        "Other programs (gifted, summer school)": [
            ("1210", 355), ("1271", 2_029), ("1292", 62), ("1299", 1_073), ("1400", 458)],
    },
    "Support services": {
        "Student support (counselors, social workers, etc.)": [
            ("2110", 19_576), ("2120", 27_356), ("2140", 7_195), ("2150", 16_136),
            ("2160", 5_445), ("2190", 11_907)],
        "Building operations and maintenance": [("2540", 72_437)],
        "Principals' offices": [("2410", 58_889), ("2490", 508)],
        "Student transportation": [("2550", 47_648)],
        "Teacher support and libraries": [
            ("2210", 6_396), ("2220", 11_663), ("2230", 771), ("2240", 11_404)],
        "Central services (HR, tech, research)": [
            ("2610", 563), ("2620", 2_600), ("2630", 4_183), ("2640", 8_892),
            ("2660", 13_689), ("2670", 455), ("2690", -3_982)],
        "Business and finance": [("2510", 483), ("2520", 8_117), ("2570", 2_180)],
        "School board and superintendent": [("2310", 950), ("2320", 9_560)],
    },
    "Other": {
        "Required reserve (contingency)": [("6000", 41_056)],
        "Transfers to other funds": [("5200", 14_755)],
        "Community services": [("3300", 2_391)],
    },
}

# "Instructional Other" (function 1100), a negative district-wide adjustment
# PPS books against instruction. Vol 1 p. 107. Spread over the grade-level
# classroom lines in proportion to their size, since PPS doesn't assign it.
INSTRUCTION_ADJUSTMENT = ("1100", -3_984)
ADJUSTED_LINES = ("Elementary classrooms (K-5)", "Middle school classrooms", "High school classrooms")

# Budgeted General Fund FTE by function, 2026-27. Vol 1 p. 107.
FUNCTION_FTE = {"1111": 870.20, "1121": 436.87, "1122": 11.60, "1131": 583.62, "1132": 3.45}

GROUP_NOTES = {
    "Instruction": "Teachers, classroom aides, and classroom costs.",
    "Support services": "Everything that keeps schools running outside the classroom.",
    "Other": "Money set aside or moved to other PPS funds.",
}

# Salaries plus payroll costs (benefits, PERS, taxes), General Fund 2026-27.
# Vol 1 p. 102 ("$680.5 million, or 78.9%") and p. 104 (4,481.34 budgeted FTE).
GF_PEOPLE_COST = 680_515
GF_FTE = 4_481.34

# General Fund salaries and budgeted FTE by employee group, 2026-27. Vol 1
# p. 104. Payroll costs (benefits, PERS, payroll taxes) are $245,251K on
# $435,264K of salaries, a 56% load applied to every group below.
GF_SALARIES_TOTAL = 435_264
GF_PAYROLL_COSTS = 245_251
SALARY_FTE = {
    "licensed": (276_175, 2_718.10),       # teachers and other licensed staff
    "classified": (65_416, 1_255.72),      # office staff, custodians, aides
    "administrator": (33_957, 199.90),     # principals and assistant principals
}

# Building operations and maintenance (function 2540), 2026-27. Vol 1 p. 108.
# Includes custodians, utilities, and upkeep for every PPS building.
GF_BUILDING_OPS = 72_437

# Every PPS fund, 2026-27 adopted. Vol 1 p. 320.
ALL_FUNDS = [
    ("General Fund (day-to-day operations)", 862_112),
    ("Construction and capital projects (mostly voter-approved bonds)", 1_382_508),
    ("Debt payments (bonds and pension debt)", 278_876),
    ("Grants and other restricted money", 223_936),
    ("Insurance and internal services", 21_159),
]

# General Fund history. Vol 1 pp. 103-104 (2022-23 to 2026-27) and p. 113
# (forecast). "carryover" is the beginning fund balance; "reserve" is the
# contingency plus any budgeted ending balance. Revenue and spending on the
# page exclude both, so each year shows money in vs. money out.
GF_YEARS = [
    # year, status, total resources, carryover, spending, reserve
    ("2022-23", "actual",   830_038,  98_804, 724_541, 105_497),
    ("2023-24", "actual",   873_375, 105_497, 787_298,  86_078),
    ("2024-25", "actual",   855_834,  86_078, 811_511,  44_323),
    ("2025-26", "budget",   868_580,  45_000, 827_401,  41_179),
    ("2026-27", "budget",   862_112,  41_000, 821_055,  41_056),
    ("2027-28", "forecast", 876_860,  41_056, 900_269,  41_790),
    ("2028-29", "forecast", 902_250,  41_790, 901_577,  43_023),
    ("2029-30", "forecast", 916_481,  43_023, 933_802,  43_673),
]
# PPS's own projected shortfalls (negative budgeted ending balance). p. 113.
GF_SHORTFALL = {"2027-28": 65_198, "2028-29": 42_349, "2029-30": 60_991}

# General Fund spending by object, for the cost-drivers panel. Vol 1 p. 104
# (2022-23 to 2026-27) and p. 113 (forecast). "salaries" is licensed +
# classified + non-represented + administrators + other salaries; "benefits"
# is object 200, associated payroll costs (health insurance, PERS, payroll
# taxes). "other" is purchased services, supplies, capital outlay, other
# objects, and transfers. Each year must add to GF_YEARS spending.
GF_OBJECTS = {
    #           salaries  benefits  other
    "2022-23": (401_758, 188_552, 134_231),
    "2023-24": (431_978, 206_210, 149_110),
    "2024-25": (453_047, 212_696, 145_768),
    "2025-26": (442_948, 248_507, 135_946),
    "2026-27": (435_264, 245_251, 140_542),
    "2027-28": (448_322, 309_868, 142_079),
    "2028-29": (461_772, 295_055, 144_750),
    "2029-30": (475_626, 310_702, 147_474),
}

# Average daily enrollment (ADMr). Vol 1 p. 21 (2025-26 and 2026-27 are
# estimates).
ENROLLMENT = {"2022-23": 44_518, "2023-24": 43_882, "2024-25": 43_201,
              "2025-26": 42_467, "2026-27": 41_661}

# General Fund purchased services (object 300) by line, for the contracts
# panel. Vol 1 pp. 109-110; years are 2022-23 to 2024-25 actual, 2025-26
# adopted, 2026-27 adopted. Object definitions are on pp. 81-82. Grouped and
# renamed in plain language.
# Union contracts are not here: those costs are salaries and benefits.
PS_YEARS = ["2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]
# Printed "Subtotal - 300 - PURCHASED SERVICES", p. 110.
PS_TOTALS = [106_798, 119_699, 117_095, 104_835, 110_672]
GF_PURCHASED_SERVICES = {
    "Student transportation": (
        "Contracted school buses and taxis, TriMet passes, and field trips.", [
        ("533110", "Reimb - School Bus", (23_696, 28_710, 29_716, 25_495, 28_829)),
        ("533120", "Reimb - Taxi Cab", (3_513, 2_647, 2_405, 3_290, 2_885)),
        ("533130", "Reimb - In-Lieu", (22, 12, 14, 14, 14)),
        ("533140", "Reimb - Tri-Met", (2_020, 2_025, 2_034, 2_066, 2_108)),
        ("533150", "Reimb - Field Trips", (1_241, 1_578, 1_923, 1_258, 1_388)),
        ("533200", "Non-Reimb Student Transport", (1_101, 1_255, 1_333, 763, 953)),
        ("534901", "Student Academic Transport", (0, 0, 7, 7, 7)),
    ]),
    "Payments to charter schools": (
        "Money PPS passes to the charter schools it sponsors for the students they serve.", [
        ("536000", "Charter Schools", (16_670, 18_504, 18_177, 19_449, 18_639)),
    ]),
    "Professional services": (
        "Consultants, legal, audit, security, and other specialized outside services.", [
        ("538100", "Audit Services", (220, 353, 265, 227, 227)),
        ("538200", "Legal Services", (1_068, 1_198, 1_063, 1_370, 1_350)),
        ("538300", "Architect and Engineering Svcs", (139, 326, 118, 0, 0)),
        ("538400", "Negotiation Services", (116, 433, 5, 140, 100)),
        ("538500", "Management Services", (65, 382, 735, 78, 78)),
        ("538600", "Data Processing Services", (85, 0, 0, 0, 0)),
        ("538800", "Election Services", (249, 74, 331, 350, 350)),
        ("538910", "Security Services", (130, 494, 298, 148, 83)),
        ("538920", "Staff Services", (26, 27, 33, 0, 0)),
        ("538930", "Secretarial/Clerical Services", (0, 16, 36, 0, 0)),
        ("538940", "Professional Moving Services", (162, 214, 253, 163, 164)),
        ("538950", "Professional Health Care Svcs", (14, 83, 30, 2, 2)),
        ("538960", "Professional Child Care Svcs", (116, 35, 2, 356, 194)),
        ("538970", "Graphic Arts Services", (63, 1, 0, 7, 7)),
        ("538980", "Laundering Services", (27, 50, 78, 21, 17)),
        ("538990", "Non-Instr Pers/Professional Sv", (10_023, 13_862, 9_412, 4_636, 5_855)),
        ("538995", "Meal Services", (0, 0, 189, 187, 187)),
    ]),
    "Utilities": (
        "Electricity, heating fuel, water and sewer, and garbage.", [
        ("532500", "Electricity", (4_402, 4_839, 5_815, 5_373, 6_450)),
        ("532600", "Fuel", (4_200, 3_520, 3_499, 4_763, 3_285)),
        ("532700", "Water and Sewage", (3_201, 3_390, 3_992, 3_634, 3_634)),
        ("532800", "Garbage", (1_146, 1_034, 1_076, 1_185, 1_185)),
    ]),
    "Building repairs and property services": (
        "Outside repair and maintenance work, rentals, and other property services.", [
        ("532100", "Cleaning Services", (0, 40, 3, 0, 0)),
        ("532200", "Repairs and Maintenance Svcs", (997, 1_384, 880, 1_729, 778)),
        ("532400", "Rentals", (304, 301, 235, 362, 383)),
        ("532900", "Other Property Services", (9_947, 9_408, 10_781, 3_245, 8_790)),
    ]),
    "Tuition to other schools": (
        "Tuition PPS pays to private schools and other agencies that serve its students.", [
        ("537300", "Tuition to Private Schools", (7_731, 9_514, 9_936, 9_898, 9_410)),
        ("537410", "Tuition - Fees College Credit", (10, 48, 77, 0, 0)),
    ]),
    "Instruction and training services": (
        "Outside providers for teaching, program improvement, and staff training.", [
        ("531100", "Instructional Services", (2_802, 3_925, 4_298, 5_324, 5_000)),
        ("531200", "Instr Program Improvement Svcs", (1_782, 1_626, 1_241, 1_105, 1_035)),
        ("531300", "Student Services", (28, 11, 13, 246, 126)),
        ("531800", "Local Mtgs/Non-Instr Staff Dev", (1_429, 914, 448, 905, 760)),
        ("531810", "Non-Instr Dev Profess Dev Fds", (122, 98, 125, 259, 104)),
        ("531900", "Other Instr Prof/Tech Svcs", (3_271, 3_750, 2_603, 1_837, 2_363)),
    ]),
    "Travel, phone, postage, and printing": (
        "Staff and student travel, communications, and printing.", [
        ("534100", "Travel, Local in District", (395, 148, 143, 195, 208)),
        ("534200", "Travel, Out of District", (726, 503, 604, 497, 380)),
        ("534210", "Trav Out Dist Profess Dev Fds", (42, 54, 58, 4, 4)),
        ("534300", "Travel, Student Activities", (412, 449, 408, 767, 705)),
        ("534900", "Other Travel", (0, 0, 1, 0, 0)),
        ("535100", "Telephone", (597, 873, 683, 414, 721)),
        ("535300", "Postage", (278, 291, 91, 353, 371)),
        ("535400", "Advertising", (119, 88, 49, 67, 65)),
        ("535500", "Printing and Binding", (1_091, 954, 1_349, 1_643, 1_427)),
        ("535920", "Internet Fees", (0, 9, 37, 0, 0)),
        ("535990", "Wide Area Network/Misc", (999, 248, 190, 1_005, 52)),
    ]),
}

# Volume 2 short names that don't prefix-match a master-CSV name.
MASTER_NAME = {
    "Access": "ACCESS Academy",
    "Boise-Eliot/Humboldt": "Boise-Eliot Elementary School",
    "MLK Jr": "Dr. Martin Luther King Jr. School",
    "Metro. Learning Center": "Metropolitan Learning Center",
    "McDaniel": "Leodis V. McDaniel High School",
}


def grouped_uses() -> list[dict]:
    groups = []
    total = 0
    for group, items in GF_USES.items():
        children = []
        adjust_base = sum(v for label in ADJUSTED_LINES if label in items for _, v in items[label])
        for label, rows in items.items():
            amount = sum(v for _, v in rows)
            codes = [c for c, _ in rows]
            if label in ADJUSTED_LINES:
                amount += round(INSTRUCTION_ADJUSTMENT[1] * amount / adjust_base)
                codes.append(INSTRUCTION_ADJUSTMENT[0])
            total += amount
            child = {"label": label, "amount": amount, "codes": codes}
            fte = sum(FUNCTION_FTE.get(c, 0) for c in codes)
            if fte:
                child["fte"] = round(fte, 2)
            children.append(child)
        groups.append({"label": group, "note": GROUP_NOTES[group],
                       "amount": sum(c["amount"] for c in children),
                       "children": children})
    # Printed rows are rounded to the thousand, so allow a little slack.
    if abs(total - GENERAL_FUND_TOTAL) > 3:
        sys.exit(f"General Fund uses add to {total:,}, expected {GENERAL_FUND_TOTAL:,}")
    resources = sum(v for _, v, _ in GF_RESOURCES)
    if abs(resources - GENERAL_FUND_TOTAL) > 3:
        sys.exit(f"General Fund resources add to {resources:,}, expected {GENERAL_FUND_TOTAL:,}")
    return groups


def history() -> list[dict]:
    rows = []
    for year, status, total, carry, spend, reserve in GF_YEARS:
        # Each year balances: resources = spending + reserve - shortfall.
        short = GF_SHORTFALL.get(year, 0)
        if abs(total - (spend + reserve - short)) > 3:
            sys.exit(f"{year}: resources {total:,} != spending + reserve - shortfall")
        salaries, benefits, other = GF_OBJECTS[year]
        if abs(salaries + benefits + other - spend) > 3:
            sys.exit(f"{year}: objects add to {salaries + benefits + other:,}, spending is {spend:,}")
        rows.append({"year": year, "status": status,
                     "salaries": salaries, "benefits": benefits, "other_costs": other,
                     "revenue": total - carry, "spending": spend,
                     "carryover": carry, "reserve": reserve,
                     "shortfall": GF_SHORTFALL.get(year),
                     "enrollment": ENROLLMENT.get(year)})
    return rows


def purchased_services() -> dict:
    groups = []
    for label, (note, lines) in GF_PURCHASED_SERVICES.items():
        groups.append({"label": label, "note": note,
                       "by_year": [sum(y[i] for _, _, y in lines) for i in range(len(PS_YEARS))],
                       "lines": [{"code": c, "name": n, "by_year": list(y)} for c, n, y in lines]})
    for i, year in enumerate(PS_YEARS):
        total = sum(g["by_year"][i] for g in groups)
        if abs(total - PS_TOTALS[i]) > 3:
            sys.exit(f"{year}: purchased services add to {total:,}, PPS prints {PS_TOTALS[i]:,}")
    return {"years": PS_YEARS, "totals": PS_TOTALS,
            "groups": sorted(groups, key=lambda g: -g["by_year"][-1])}


def schools() -> list[dict]:
    data = json.loads(SCHOOLS_JSON.read_text())["schools"]
    master = pd.read_csv(MASTER)
    by_name = master.set_index(master["school_name"].str.strip())
    out = []
    for s in data:
        name = MASTER_NAME.get(s["name"])
        if name is None:
            hits = [m for m in by_name.index
                    if m.lower().startswith(s["name"].lower() + " ") or m.lower() == s["name"].lower()]
            if len(hits) != 1:
                sys.exit(f"{s['name']!r}: {len(hits)} master matches {hits}")
            name = hits[0]
        row = by_name.loc[name]

        def cat(col: str) -> str | None:
            v = row.get(col)
            return None if v is None or (isinstance(v, float) and math.isnan(v)) else v

        out.append({**s, "master_name": name,
                    "rs_a": cat("rs_a_category"), "rs_b": cat("rs_b_category")})
    return out


def cost_per_fte() -> dict[str, float]:
    """Average salary plus benefits per FTE, in dollars."""
    load = 1 + GF_PAYROLL_COSTS / GF_SALARIES_TOTAL
    return {k: 1000 * pay / fte * load for k, (pay, fte) in SALARY_FTE.items()}


def closure_savings(rows: list[dict]) -> dict:
    """Annual General Fund savings estimate for each scenario's closures.

    Per closing school, in 2026-27 dollars:
      leadership  principal/AP and office staff (go away with the school)
      custodians  the building's custodians
      building    the building's share of all operations and maintenance by
                  square footage, which includes custodians (high case only:
                  assumes PPS sells or leases the building)
      teachers    teachers beyond what the school's students need at the
                  district's average students per teacher for its grade band
                  (high case only: assumes receiving schools absorb students
                  at that average)
    Low = leadership + custodians. High = leadership + max(custodians,
    building) + teachers. Teachers and other staff that follow students
    (special education, ESL, counselors, nutrition) are not savings.
    """
    cost = cost_per_fte()
    master = pd.read_csv(MASTER)
    pps = master[master["school_type"] != "Charter School"]
    sqft = pps.set_index(pps["school_name"].str.strip())["square_feet"]
    # Upper bound per square foot: school buildings only, so offices and
    # other district sites don't dilute the rate.
    per_sqft = GF_BUILDING_OPS * 1000 / sqft.sum()

    band_ratio = {}
    for band in ("K-5", "K-8", "Middle", "High"):
        # MLC is listed with middle schools but is K-12.
        b = [r for r in rows if r["band"] == band and r["name"] != "Metro. Learning Center"]
        band_ratio[band] = sum(r["enrollment"]["2026_27"] for r in b) / sum(r["fte_2026_27"]["teachers"] for r in b)

    def one(r: dict) -> dict:
        f = r["fte_2026_27"]
        enroll = r["enrollment"]["2026_27"]
        leadership = f["admin"] * cost["administrator"] + f["clerical"] * cost["classified"]
        custodians = f["custodial"] * cost["classified"]
        building = sqft.get(r["master_name"], float("nan")) * per_sqft
        extra_teachers = max(0.0, f["teachers"] - enroll / band_ratio[r["band"]])
        teachers = extra_teachers * cost["licensed"]
        high_building = max(custodians, building) if building == building else custodians
        return {
            "name": r["name"], "master_name": r["master_name"], "band": r["band"],
            "enrollment": enroll, "school_dollars": r["school_dollars_2026_27"],
            "leadership": round(leadership), "custodians": round(custodians),
            "building_extra": round(high_building - custodians),
            "teachers": round(teachers), "extra_teacher_fte": round(extra_teachers, 2),
            "low": round(leadership + custodians),
            "high": round(leadership + high_building + teachers),
        }

    out = {"cost_per_fte": {k: round(v) for k, v in cost.items()},
           "building_ops_per_sqft": round(per_sqft, 2),
           "students_per_teacher": {k: round(v, 1) for k, v in band_ratio.items()},
           "scenarios": {}}
    for key in ("rs_a", "rs_b"):
        schools = [one(r) for r in rows if r[key] == "closes"]
        out["scenarios"][key] = {
            "schools": sorted(schools, key=lambda s: -s["high"]),
            "low": sum(s["low"] for s in schools),
            "high": sum(s["high"] for s in schools),
            "students": sum(s["enrollment"] for s in schools),
        }
    return out


def main() -> None:
    payload = {
        "sources": SOURCES,
        "general_fund_total": GENERAL_FUND_TOTAL,
        "resources": [{"label": l, "amount": a, "note": n} for l, a, n in GF_RESOURCES],
        "uses": grouped_uses(),
        "people_cost": GF_PEOPLE_COST,
        "fte": GF_FTE,
        "all_funds": [{"label": l, "amount": a} for l, a in ALL_FUNDS],
        "history": history(),
        "purchased_services": purchased_services(),
        "schools": schools(),
    }
    payload["savings"] = closure_savings(payload["schools"])
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(payload['schools'])} schools")


if __name__ == "__main__":
    main()
