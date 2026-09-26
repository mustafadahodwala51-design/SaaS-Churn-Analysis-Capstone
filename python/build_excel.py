"""Build the Excel deliverable for the SaaS churn capstone.

Reads the cleaned CSVs written by data_analysis.py and assembles a formatted
workbook with one tab per analytical question, plus the underlying data and the
eight charts from the Python pipeline.

Why openpyxl and not a charting library: openpyxl is already a dependency of
the analysis environment, it can embed the existing PNG charts directly, and it
lets every tab be formatted (number formats, frozen headers, column widths)
without pulling in another package.

Run from the project root:
    python python/build_excel.py
"""

import pathlib

import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "cleaned"
CHARTS = ROOT / "visualizations" / "charts"
OUTPUT = ROOT / "excel" / "SaaS_Churn_Analysis.xlsx"

# --- Shared formatting ------------------------------------------------------
# One consistent visual language across every tab, so the workbook reads as a
# single document rather than a pile of exports.
TITLE_FONT = Font(bold=True, size=16, color="1F3864")
SUBTITLE_FONT = Font(italic=True, size=10, color="595959")
HEADER_FONT = Font(bold=True, size=11, color="FFFFFF")
HEADER_FILL = PatternFill("solid", fgColor="1F3864")
SECTION_FONT = Font(bold=True, size=12, color="1F3864")
NOTE_FONT = Font(italic=True, size=9, color="7F7F7F")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

MONEY_FMT = '"$"#,##0'
MONEY2_FMT = '"$"#,##0.00'
PCT_FMT = '0.0"%"'
NUMBER_FMT = '#,##0'
DECIMAL_FMT = '#,##0.00'


def style_header(ws, row, ncols):
    """Format a header row and freeze it so long tables stay readable."""
    for col in range(1, ncols + 1):
        cell = ws.cell(row=row, column=col)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center",
                                   wrap_text=True)
        cell.border = BORDER
    ws.freeze_panes = ws.cell(row=row + 1, column=1)


def autosize(ws, min_width=10, max_width=42, header_row=1):
    """Set column widths from the content, ignoring the merged title rows."""
    for col in range(1, ws.max_column + 1):
        letter = get_column_letter(col)
        longest = 0
        for row in range(header_row, ws.max_row + 1):
            value = ws.cell(row=row, column=col).value
            if value is None:
                continue
            text = str(value)
            # Long narrative cells should not force a 200-character column.
            if len(text) > 60:
                text = text[:60]
            longest = max(longest, len(text))
        ws.column_dimensions[letter].width = min(max(longest + 2, min_width),
                                                 max_width)


def apply_number_formats(ws, header_row, formats):
    """Apply a number format per column name, e.g. {'mrr': '"$"#,##0'}."""
    header_map = {ws.cell(row=header_row, column=c).value: c
                  for c in range(1, ws.max_column + 1)}
    for name, fmt in formats.items():
        col = header_map.get(name)
        if not col:
            continue
        for row in range(header_row + 1, ws.max_row + 1):
            cell = ws.cell(row=row, column=col)
            if isinstance(cell.value, (int, float)):
                cell.number_format = fmt


def add_title(ws, title, subtitle, ncols):
    ws["A1"] = title
    ws["A1"].font = TITLE_FONT
    ws["A2"] = subtitle
    ws["A2"].font = SUBTITLE_FONT
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max(ncols, 4))
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=max(ncols, 4))


def write_frame(wb, df, sheet, title, subtitle, formats=None, tab_color=None,
                note=None, freeze=True, max_rows=None):
    """Write a dataframe to a new sheet with consistent formatting."""
    if max_rows:
        df = df.head(max_rows)
    ws = wb.create_sheet(sheet)
    if tab_color:
        ws.sheet_properties.tabColor = tab_color
    add_title(ws, title, subtitle, len(df.columns))
    header_row = 4
    for j, col in enumerate(df.columns, start=1):
        ws.cell(row=header_row, column=j, value=col)
    style_header(ws, header_row, len(df.columns)) if freeze else None
    for i, (_, row) in enumerate(df.iterrows(), start=header_row + 1):
        for j, col in enumerate(df.columns, start=1):
            value = row[col]
            if pd.isna(value):
                value = None
            elif hasattr(value, "item"):
                try:
                    value = value.item()
                except (ValueError, AttributeError):
                    pass
            cell = ws.cell(row=i, column=j, value=value)
            cell.border = BORDER
            if isinstance(value, (int, float)):
                cell.alignment = Alignment(horizontal="right")
    if formats:
        apply_number_formats(ws, header_row, formats)
    autosize(ws, header_row=header_row)
    if note:
        r = ws.max_row + 2
        ws.cell(row=r, column=1, value=note).font = NOTE_FONT
        ws.merge_cells(start_row=r, start_column=1,
                       end_row=r, end_column=max(len(df.columns), 4))
        ws.cell(row=r, column=1).alignment = Alignment(wrap_text=True,
                                                       vertical="top")
    return ws


def add_images_sheet(wb, chart_files, title, subtitle):
    """Embed the matplotlib PNGs so the workbook carries the visuals too."""
    ws = wb.create_sheet("Charts")
    ws.sheet_properties.tabColor = "C00000"
    add_title(ws, title, subtitle, 8)
    row = 4
    for path in chart_files:
        if not path.exists():
            continue
        img = XLImage(str(path))
        # Keep the image inside a readable width without upscaling.
        scale = min(1.0, 1100 / img.width)
        img.width = int(img.width * scale)
        img.height = int(img.height * scale)
        ws.add_image(img, f"A{row}")
        row += int(img.height / 19) + 3      # ~19px per row, plus a gap
    ws.column_dimensions["A"].width = 110
    return ws


def build_kpi_table(raw):
    """Turn the KPI CSV into a table Excel can actually calculate with.

    data_analysis.py writes KPI values as display strings ("1,250", "34.4%",
    "$1,051,252"). Those are fine to read but useless in a formula or a chart,
    so the numeric part is extracted and the display prefix becomes a number
    format instead. The value stays exactly the same; only its type changes.
    """
    units = {"Customers": NUMBER_FMT, "USD": MONEY_FMT, "%": PCT_FMT,
             "Score (0-10)": DECIMAL_FMT, "Months": DECIMAL_FMT,
             "Categories": NUMBER_FMT, "Rows": NUMBER_FMT}
    rows = []
    for _, r in raw.iterrows():
        text = str(r["Value"]).strip()
        unit = str(r["Unit"]).strip()
        # "MRR peak $1,052,669 (Oct 2024)" keeps only the number.
        cleaned = (text.replace("$", "").replace(",", "").replace("%", "")
                       .split("(")[0].strip())
        try:
            numeric = float(cleaned)
        except ValueError:
            numeric = text          # leave genuinely textual KPIs as text
        # A score of 6.99 is a score, not a currency value, even though the
        # Value string happens to start with a dollar-free number.
        fmt = units.get(unit, NUMBER_FMT)
        if unit == "Score (0-10)":
            fmt = DECIMAL_FMT
        rows.append({"KPI": r["KPI"], "Value": numeric, "Unit": unit,
                     "_format": fmt})
    return pd.DataFrame(rows)


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    kpi = pd.read_csv(CLEAN / "kpi_summary.csv")
    customers = pd.read_csv(CLEAN / "customers_clean.csv",
                            parse_dates=["signup_date", "churn_date",
                                         "last_login_date"])
    monthly = pd.read_csv(CLEAN / "customer_monthly_clean.csv",
                          parse_dates=["month_start"])
    risk = pd.read_csv(CLEAN / "churn_risk_scoring.csv")
    trend = pd.read_csv(CLEAN / "analysis_monthly_trend.csv",
                        parse_dates=["month_start"])
    by_plan = pd.read_csv(CLEAN / "analysis_churn_by_plan.csv")
    by_channel = pd.read_csv(CLEAN / "analysis_churn_by_channel.csv")
    channel_value = pd.read_csv(CLEAN / "analysis_channel_value.csv")
    reasons = pd.read_csv(CLEAN / "analysis_churn_reasons.csv")
    cohort = pd.read_csv(CLEAN / "analysis_cohort_retention.csv", index_col=0)
    retention = pd.read_csv(CLEAN / "analysis_retention_by_tenure.csv")
    yearly = pd.read_csv(CLEAN / "analysis_yearly_growth_vs_churn.csv")
    risk_summary = pd.read_csv(CLEAN / "analysis_risk_summary.csv")

    wb = Workbook()
    wb.remove(wb.active)          # drop the default empty sheet

    money = {"mrr": MONEY_FMT, "mrr_added": MONEY_FMT, "mrr_lost": MONEY_FMT,
             "net_mrr_change": MONEY_FMT, "total_mrr": MONEY_FMT,
             "avg_mrr_per_customer": MONEY_FMT, "avg_mrr": MONEY_FMT,
             "avg_ltv": MONEY_FMT, "final_mrr": MONEY_FMT, "mrr_at_signup": MONEY2_FMT,
             "lifetime_value": MONEY_FMT, "arr": MONEY_FMT, "arpu": MONEY2_FMT}
    pct = {"churn_pct": PCT_FMT, "share_of_all_losses": PCT_FMT,
           "starter_share": PCT_FMT, "share_of_active_mrr": PCT_FMT,
           "retention_pct": PCT_FMT, "monthly_churn_pct": PCT_FMT,
           "share_pct": PCT_FMT, "final_adoption": "0.0%",
           "avg_adoption": "0.0%", "avg_mrr_per_new_customer": MONEY_FMT}
    counts = {"customers": NUMBER_FMT, "churned": NUMBER_FMT,
              "customers_lost": NUMBER_FMT, "new_customers": NUMBER_FMT,
              "active_customers": NUMBER_FMT, "churned_this_month": NUMBER_FMT,
              "at_risk_subscribed": NUMBER_FMT, "cancelled_this_month": NUMBER_FMT,
              "still_subscribed": NUMBER_FMT, "cohort_size": NUMBER_FMT,
              "net_new_customers": NUMBER_FMT, "tenure_months": NUMBER_FMT}

    # --- 1. README tab ------------------------------------------------------
    ws = wb.create_sheet("Start Here", 0)
    ws.sheet_properties.tabColor = "1F3864"
    ws["A1"] = "SaaS Churn Analysis - Capstone Workbook"
    ws["A1"].font = Font(bold=True, size=18, color="1F3864")
    ws["A3"] = "SYNTHETIC DATA"
    ws["A3"].font = Font(bold=True, size=12, color="C00000")
    readme = [
        "Every figure in this workbook comes from a SIMULATED dataset generated by",
        "data/generate_dataset.py. It does not describe any real company or customer.",
        "",
        "What each tab answers:",
        "  KPI Summary        - headline numbers for the whole period",
        "  Monthly Trend      - is the business growing or shrinking?",
        "  Retention          - when do customers leave, and do newer cohorts stick?",
        "  Churn by Plan      - which tier loses the most logos, and which costs the most revenue?",
        "  Churn by Channel   - are we buying customers that stay?",
        "  Early Warning      - measurable differences in the final 3 months",
        "  NRR                - is revenue retained from the existing base?",
        "  Revenue at Risk    - how much MRR sits with high-risk active accounts?",
        "  Yearly Comparison  - is new business replacing churn?",
        "  Recommendations    - what to do about it",
        "  Charts             - the eight figures from the Python pipeline",
        "  Data (clean)       - the cleaned tables every number is built from",
        "",
        "One thing to keep in mind when reading retention: customers still active at",
        "the end of the data are censored. They count as 'at risk' but not as churn,",
        "so the survival curve does not collapse at the right-hand edge of the chart.",
    ]
    for i, line in enumerate(readme, start=4):
        ws.cell(row=i, column=1, value=line).font = Font(size=10)
    ws.column_dimensions["A"].width = 105

    # --- 2. KPI summary -----------------------------------------------------
    kpi_table = build_kpi_table(kpi)
    kpi_formats = {"Value": "#,##0.0"}      # per-row override applied below
    write_frame(wb, kpi_table.drop(columns=["_format"]), "KPI Summary",
                "Headline numbers, 2023-2024",
                "Synthetic data. 'Churn rate' is logo churn over the whole period. "
                "Values are real numbers, so this tab can drive a formula or chart.",
                formats=kpi_formats, tab_color="1F3864",
                note="Source: data/cleaned/kpi_summary.csv, written by data_analysis.py.")
    # Each KPI gets its own display format (currency vs percent vs count).
    kpi_ws = wb["KPI Summary"]
    for i, (_, row) in enumerate(kpi_table.iterrows(), start=5):
        cell = kpi_ws.cell(row=i, column=2)
        if isinstance(cell.value, (int, float)):
            cell.number_format = row["_format"]

    # --- 3. Monthly trend ---------------------------------------------------
    write_frame(wb, trend, "Monthly Trend",
                "Monthly recurring revenue and active customers",
                "Growth collapsed to roughly +0.8% a month in the last 6 months "
                "after averaging +51.9% early on.",
                formats=money | counts, tab_color="2E75B6",
                note="Peak MRR was $1,052,669 in Oct 2024. Active customers peaked at "
                     "846 in Sep 2024 and fell to 820 by Dec 2024.")

    # --- 4. Retention -------------------------------------------------------
    write_frame(wb, retention, "Retention",
                "When do customers leave? (survival view)",
                "Base for every point is all 1,250 customers who ever signed up. "
                "Customers still active at the end are censored.",
                formats=counts | pct, tab_color="2E75B6",
                note="at_risk = customers still subscribed after that month. "
                     "Dividing cancellations by everyone who ever signed up would "
                     "make the curve collapse to zero at the end of the period.")

    cohort_view = cohort.copy()
    cohort_view.columns = [f"M{c}" if str(c).isdigit() else c
                           for c in cohort_view.columns]
    write_frame(wb, cohort_view.reset_index(), "Cohort Retention",
                "Share of each signup cohort still subscribed, by month of tenure",
                "Blank-looking zeroes are cohorts that have not yet reached that "
                "month of tenure - they are not failures.",
                formats=pct, tab_color="2E75B6",
                note="Average retention is calculated only from cohorts old enough "
                     "to reach the month in question. Month 6 = 70% of signups "
                     "still active; month 12 = 43%.")

    # --- 5. Segment churn ---------------------------------------------------
    write_frame(wb, by_plan, "Churn by Plan",
                "Churn by plan tier: most logos lost vs most revenue lost",
                "These are different questions with different answers. Starter "
                "loses the most logos; Business costs the most revenue.",
                formats=money | pct | counts, tab_color="C55A11",
                note="Starter churns at 40.2% versus 23.1% for Enterprise, but "
                     "Business accounts for 40.7% of all MRR lost to churn.")

    write_frame(wb, by_channel, "Churn by Channel",
                "Churn by acquisition channel",
                "Events and Paid Search lose the most customers; Organic keeps them best.",
                formats=money | pct | counts, tab_color="C55A11",
                note="A channel's share of losses is not the same as its churn rate: "
                     "Paid Search churns more often, but Organic and Events are "
                     "smaller books.")

    write_frame(wb, channel_value, "Channel Value",
                "Is each channel worth its cost?",
                "Discounting hardest correlates with the shortest customer life.",
                formats=money | pct | counts, tab_color="C55A11")

    # --- 6. Early warning signals ------------------------------------------
    signals = pd.DataFrame([
        ["Seat adoption (final 3 months)", "33.6%", "38.5%", "-4.9 pp",
         "Low usage precedes cancellation"],
        ["Support tickets per month", "1.34", "1.22", "+0.12",
         "Slightly more friction before leaving"],
        ["NPS (0-10)", "6.85", "7.07", "-0.22", "Slightly less happy"],
        ["Signup discount %", "10.5%", "9.0%", "+1.5 pp",
         "Discounted accounts churn more"],
    ], columns=["Signal", "Churned customers", "Still active", "Difference",
                "Reading"])
    write_frame(wb, signals, "Early Warning",
                "Behaviour in the final 3 months: churned vs still active",
                "All four signals are recorded BEFORE the cancellation, so they can "
                "flag an account while there is still time to save it.",
                tab_color="7030A0",
                note="These are associations, not causes. Low adoption may be a "
                     "symptom of a customer who was already planning to leave.")

    # --- 7. NRR -------------------------------------------------------------
    nrr = pd.DataFrame([
        ["MRR 12 months ago (Dec 2023 cohort)", "$836,792", "", ""],
        ["MRR now from those same customers (Dec 2024)", "$756,792", "", ""],
        ["Net Revenue Retention (NRR)", "90.4%", "", "Below 100% means the base shrank"],
        ["  of which expansion from seat growth", "", "+6.8 pp", "Healthy"],
        ["  of which churn from cancellations", "", "-16.4 pp", "The problem"],
    ], columns=["Measure", "Value", "Effect", "Comment"])
    write_frame(wb, nrr, "NRR",
                "Net Revenue Retention: is revenue retained from the base?",
                "December 2023 compared with December 2024 for the same customers.",
                tab_color="7030A0",
                note="NRR is the single number that says whether a subscription "
                     "business is getting healthier or just buying enough new "
                     "customers to stand still. Churned customers stay in the "
                     "starting cohort and contribute 0 at the end.")

    # --- 8. Revenue at risk -------------------------------------------------
    write_frame(wb, risk_summary, "Revenue at Risk",
                "How much recurring revenue sits with high-risk accounts?",
                "Rule-based score, not a black box, so a manager can see why an "
                "account was flagged.",
                formats=money | pct, tab_color="7030A0",
                note="$508,458 (49% of active MRR) sits with 273 high-risk accounts "
                     "that use 15% of their seats, give NPS 5.8 and raise 1.9 "
                     "tickets a month.")

    write_frame(wb, risk.sort_values("risk_score", ascending=False),
                "High Risk Accounts",
                "Active accounts ranked by risk score (top 200 shown)",
                "Sorted worst first. This is the working list for a save campaign.",
                formats=money, tab_color="7030A0", max_rows=200,
                note="Full list of all 820 active accounts is in "
                     "data/cleaned/churn_risk_scoring.csv.")

    # --- 9. Yearly comparison ----------------------------------------------
    write_frame(wb, yearly, "Yearly Comparison",
                "Is new business replacing churn?",
                "2024 added far less revenue than 2023 while losing more customers.",
                formats=money | counts, tab_color="2E75B6",
                note="Average MRR per new customer fell from $1,349 in H1 2023 to "
                     "$567 in H2 2024 (-58%), and Starter share of new customers "
                     "rose from 32% to 48%.")

    # --- 10. Recommendations ------------------------------------------------
    recs = pd.DataFrame([
        [1, "Fix the first 90 days",
         "38% of all cancellations happen before month 3, and months 0-3 lose 17 "
         "points of the base. Move onboarding, activation and first support "
         "contact into a structured 90-day programme.",
         "Onboarding", "Immediate"],
        [2, "Then handle the first renewal",
         "Cancellation dips mid-tenure then climbs back to 3.9% around months "
         "12-18 as first annual contracts come up for decision. Start the "
         "conversation 90 days before the anniversary.",
         "Renewals", "Immediate"],
        [3, "Close the monthly-vs-annual billing gap",
         "Monthly billing churns at 37.3% versus 28.9% on annual. Offer an "
         "annual-billing incentive at signup.",
         "Pricing", "1-2 quarters"],
        [4, "Tie discounting to retention potential",
         "Churned customers received 1.5pp more discount on average. Approve "
         "discounts against company size and the plan the account can grow into.",
         "Sales ops", "1-2 quarters"],
        [5, "Re-balance acquisition spend",
         "Events has the highest churn (43.4%) and the lowest lifetime value of "
         "any channel ($9,699). Organic keeps customers at 24.1%. Shift budget "
         "and add a 90-day retention check before renewing channel budget.",
         "Marketing", "1-2 quarters"],
        [6, "Run a targeted save programme, not a blanket discount",
         "$508,458 of MRR sits with 273 identifiable high-risk accounts. Work "
         "that list directly; a company-wide discount would cost more than the "
         "churn it prevents.",
         "Customer success", "Immediate"],
    ], columns=["#", "Recommendation", "Why (evidence)", "Owner", "Timing"])
    write_frame(wb, recs, "Recommendations",
                "What the business should do, and why",
                "Each recommendation follows from a specific finding above.",
                tab_color="C00000")

    # --- 11. Charts ---------------------------------------------------------
    add_images_sheet(
        wb,
        [CHARTS / "01_mrr_and_customers_trend.png",
         CHARTS / "02_cohort_retention_heatmap.png",
         CHARTS / "03_churn_by_plan_tier.png",
         CHARTS / "04_churn_and_ltv_by_channel.png",
         CHARTS / "05_early_warning_signals.png",
         CHARTS / "06_retention_by_tenure.png",
         CHARTS / "07_churn_reasons.png",
         CHARTS / "08_revenue_impact_of_churn.png"],
        "Charts", "The eight figures produced by the Python pipeline")

    # --- 12. Underlying data ------------------------------------------------
    data_ws = write_frame(
        wb, customers, "Data (clean)",
        "Cleaned customer master (1,250 rows)",
        "One row per customer. This is the table every figure above is built from.",
        formats=money | counts, tab_color="808080",
        note="Monthly activity, churn reasons, and the full 820-account risk list "
             "are in data/cleaned/ as CSV files.")
    data_ws.auto_filter.ref = (f"A4:{get_column_letter(len(customers.columns))}"
                               f"{data_ws.max_row}")

    # Reasons table belongs with the data rather than in its own tab.
    write_frame(wb, reasons, "Churn Reasons",
                "Why customers gave for leaving",
                "Self-reported at cancellation, so treat as indicative only.",
                formats=money | counts | pct, tab_color="808080")

    monthly_small = monthly[["customer_id", "month_start", "tenure_month",
                             "mrr", "seats", "active_users", "adoption_rate",
                             "support_tickets", "nps_score"]]
    write_frame(wb, monthly_small, "Data (monthly)",
                "Cleaned monthly activity",
                f"{len(monthly):,} customer-months. Kept on its own tab because "
                "of its size.",
                formats=money | counts, tab_color="808080", max_rows=20000)

    wb.save(OUTPUT)
    print(f"Workbook written to {OUTPUT.relative_to(ROOT)}")
    print(f"  tabs: {len(wb.sheetnames)}")
    for name in wb.sheetnames:
        print(f"    - {name}")


if __name__ == "__main__":
    main()
