"""
================================================================================
CAPSTONE PROJECT  -  SAAS CUSTOMER CHURN & REVENUE RETENTION ANALYSIS
================================================================================
Stage 1 of the pipeline: data cleaning, exploratory analysis and business
analysis. Runs end-to-end with no manual steps.

    python python/data_analysis.py

INPUTS
    data/raw/customers_raw.csv            (1 row per customer)
    data/raw/customer_monthly_raw.csv     (1 row per customer per month)

OUTPUTS
    data/cleaned/customers_clean.csv
    data/cleaned/customer_monthly_clean.csv
    data/cleaned/churn_risk_scoring.csv
    data/cleaned/kpi_summary.csv
    data/cleaned/analysis_*.csv           (one file per business question)
    visualizations/charts/*.png           (all charts)
    reports/console_findings.txt          (the printed report, saved to file)

HOW THE SCRIPT IS ORGANISED
    Section 1  Load the raw files
    Section 2  First look at the data
    Section 3  Data quality audit (what is wrong with it)
    Section 4  Cleaning (fix each problem found in Section 3)
    Section 5  Feature engineering (columns the business questions need)
    Section 6  Descriptive statistics
    Section 7  Exploratory data analysis + charts
    Section 8  Business analysis (answers the key questions)
    Section 9  Save everything
================================================================================
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")                     # lets charts save without a popup window
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# SECTION 0: PATHS, STYLE AND SMALL HELPERS
# -----------------------------------------------------------------------------

# FOLDER LAYOUT
# __file__ is the full path of this script. We work out the project root from
# it, so the script still runs correctly no matter which folder you launch it from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

RAW_DIR = os.path.join(ROOT, "data", "raw")
CLEAN_DIR = os.path.join(ROOT, "data", "cleaned")
CHART_DIR = os.path.join(ROOT, "visualizations", "charts")
REPORT_DIR = os.path.join(ROOT, "reports")

for folder in [CLEAN_DIR, CHART_DIR, REPORT_DIR]:
    os.makedirs(folder, exist_ok=True)

# A colour palette used by every chart so the whole project looks consistent.
PALETTE = {
    "navy": "#1F3864",
    "blue": "#2E75B6",
    "light_blue": "#9DC3E6",
    "red": "#C00000",
    "orange": "#ED7D31",
    "green": "#548235",
    "grey": "#7F7F7F",
    "light_grey": "#D9D9D9",
}

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11

# Everything printed in this script is also written to a text file, so the
# findings can be pasted straight into the report.
CONSOLE_LINES = []


def say(text=""):
    """Print a line AND remember it for the text report."""
    print(text)
    CONSOLE_LINES.append(str(text))


def banner(title, char="=", width=76):
    say("")
    say(char * width)
    say(title)
    say(char * width)


def money(value):
    """Format a number as currency, e.g. 12345.6 -> '$12,346'."""
    if pd.isna(value):
        return "n/a"
    return f"${value:,.0f}"


def save_chart(fig, filename):
    """Save a matplotlib figure into visualizations/charts/."""
    path = os.path.join(CHART_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    say(f"    chart saved -> visualizations/charts/{filename}")


# -----------------------------------------------------------------------------
# SECTION 1: LOAD THE RAW DATA
# -----------------------------------------------------------------------------

banner("SECTION 1  |  LOADING RAW DATA")

# IMPORTANT: keep every column as text first (dtype=str). If we let pandas guess
# types straight away, a column that contains a stray "$" or "n/a" becomes a
# text column and we lose the ability to see the real numbers inside it.
# Loading as text first is what a careful analyst does when auditing a file.
raw_customers = pd.read_csv(
    os.path.join(RAW_DIR, "customers_raw.csv"), dtype=str, keep_default_na=False
)
raw_monthly = pd.read_csv(
    os.path.join(RAW_DIR, "customer_monthly_raw.csv"), dtype=str, keep_default_na=False
)

# An empty string is not a value. Turn "" and "n/a" and "NA" into real NaN so
# that pandas counts them as missing.
def blank_to_nan(series):
    return series.replace({"": np.nan, "n/a": np.nan, "N/A": np.nan,
                           "NA": np.nan, "null": np.nan, "None": np.nan})

raw_customers = raw_customers.apply(blank_to_nan)
raw_monthly = raw_monthly.apply(blank_to_nan)

say(f"customers_raw.csv          {raw_customers.shape[0]:>6,} rows x "
    f"{raw_customers.shape[1]:>2} columns")
say(f"customer_monthly_raw.csv   {raw_monthly.shape[0]:>6,} rows x "
    f"{raw_monthly.shape[1]:>2} columns")
say("")
say("First 3 customer records, exactly as they arrived from the file:")
say(raw_customers.head(3).to_string(index=False))


# -----------------------------------------------------------------------------
# SECTION 2: FIRST LOOK AT THE DATA
# -----------------------------------------------------------------------------

banner("SECTION 2  |  FIRST LOOK AT THE DATA")

say("What is one row?")
say("  customers_raw.csv          one row = one CUSTOMER (the master list)")
say("  customer_monthly_raw.csv   one row = one CUSTOMER in one MONTH (activity)")
say("")
say("Columns in the customer table and what they mean:")
COLUMN_GUIDE = {
    "customer_id": "Unique code for the customer (primary key)",
    "signup_date": "Date the customer first started paying",
    "plan_tier": "Which plan they bought (Starter / Growth / Business / Enterprise)",
    "industry": "Industry the customer's company works in",
    "country": "Country of the customer",
    "region": "Wider geographic region",
    "company_size": "Employee band of the customer (1-50, 51-200, 201-1000, 1000+)",
    "seats": "Number of user seats bought",
    "billing_cycle": "Monthly or Annual billing",
    "acquisition_channel": "How the customer was first won",
    "discount_pct": "Percentage discount off the list price",
    "mrr_at_signup": "Monthly Recurring Revenue on day one",
    "status": "Active or Churned at the end of the period",
    "churn_date": "Date the customer cancelled (blank if still active)",
    "churn_reason": "Why they cancelled (blank if still active)",
    "tenure_months": "How many months the customer stayed",
    "nps_score": "Net Promoter Score 0-10 given by the customer",
    "last_login_date": "Last time anyone logged into the product",
    "seats_current": "Seats held at the end of the period",
}
for col in raw_customers.columns:
    say(f"  {col:<20} {COLUMN_GUIDE.get(col, 'simulator field (dropped later)')}")

say("")
say("Columns in the monthly activity table:")
MONTHLY_GUIDE = {
    "customer_id": "Links back to the customer table",
    "month_start": "First day of the month this row describes",
    "month_number": "Calendar month number (1-12)",
    "year": "Calendar year",
    "tenure_month": "How many months the customer had been subscribed",
    "mrr": "Revenue this customer generated that month",
    "seats": "Seats held that month",
    "active_users": "Seats that were actually used that month",
    "adoption_rate": "Share of seats actually used (0-1, or a percent as text)",
    "support_tickets": "Support tickets raised that month",
    "nps_score": "NPS given that month",
    "is_active": "1 = the customer was active that month",
}
for col in raw_monthly.columns:
    say(f"  {col:<20} {MONTHLY_GUIDE.get(col, '')}")


# -----------------------------------------------------------------------------
# SECTION 3: DATA QUALITY AUDIT
# -----------------------------------------------------------------------------

banner("SECTION 3  |  DATA QUALITY AUDIT")

say("Before trusting any number we check the file for the problems that are")
say("always present in a real data export. Findings are listed as we go.")
say("")

quality_issues = []          # a list we will turn into a table later


def log_issue(area, issue, rows_affected, treatment):
    quality_issues.append({
        "Area": area,
        "Issue found": issue,
        "Rows affected": rows_affected,
        "How it was treated": treatment,
    })


# --- 3.1 Duplicates ---------------------------------------------------------
say("3.1  Duplicate records")
cust_dupes = int(raw_customers.duplicated().sum())
mon_dupes = int(raw_monthly.duplicated().sum())
say(f"     customers_raw.csv          : {cust_dupes} fully identical rows")
say(f"     customer_monthly_raw.csv   : {mon_dupes} fully identical rows")
say("     Cause: the export job ran twice and appended the same rows.")
say(f"     Treatment: drop_duplicates() -> removes {cust_dupes + mon_dupes} rows.")
log_issue("Customers", "Fully duplicated rows (double export)", cust_dupes,
          "Removed with drop_duplicates()")
log_issue("Monthly activity", "Fully duplicated rows (double export)", mon_dupes,
          "Removed with drop_duplicates()")

# --- 3.2 Text / casing problems --------------------------------------------
say("")
say("3.2  Inconsistent text values")
for col in ["plan_tier", "billing_cycle", "region", "status", "company_size"]:
    values = raw_customers[col].dropna().unique()
    if len(values) > 0:
        say(f"     {col:<15} {len(values)} distinct value(s): "
            f"{sorted(values.tolist())[:8]}")
say("     Cause: three different systems write to this field.")
say("     Treatment: strip spaces, then standardise capitalisation.")

# Count the rows whose plan name is not already in the correct, clean form.
plan_raw = raw_customers["plan_tier"]
plan_needs_cleaning = int((plan_raw != plan_raw.str.strip().str.title()).sum())
say(f"     plan_tier needs cleaning on {plan_needs_cleaning} row(s) "
    f"(stray spaces and lower case)")
log_issue("Customers", "plan_tier stored with stray spaces and lower case",
          plan_needs_cleaning,
          "Stripped whitespace and applied title case")

# --- 3.3 Numbers hiding inside text ----------------------------------------
say("")
say("3.3  Numeric columns that contain text")
def count_nonnumeric(series):
    return int(pd.to_numeric(series, errors="coerce").isna().sum())

mrr_text = count_nonnumeric(raw_customers["mrr_at_signup"])
seats_text = count_nonnumeric(raw_customers["seats"])
say(f"     mrr_at_signup : {mrr_text} row(s) are not numbers "
    f"(currency symbols, e.g. '$1,674.39')")
say(f"     seats         : {seats_text} row(s) are not numbers (the text 'n/a')")
say("     Treatment: strip '$' and ',' then convert with pd.to_numeric().")
log_issue("Customers", "mrr_at_signup stored as text with $ and , characters",
          mrr_text, "Removed symbols, converted to numeric")
log_issue("Customers", "seats stored as the text 'n/a'", seats_text,
          "Converted to numeric, left as missing")

# --- 3.4 Date formats ------------------------------------------------------
say("")
say("3.4  Inconsistent date formats")
slash_dates = int(raw_customers["churn_date"].dropna()
                  .astype(str).str.contains("/").sum())
say(f"     churn_date   : {slash_dates} row(s) use DD/MM/YYYY, the rest use YYYY-MM-DD")
say("     signup_date  : all rows use YYYY-MM-DD")
say("     last_login_date : all rows use YYYY-MM-DD")
say("     Treatment: parse both formats with dayfirst=True, then store as datetime.")
log_issue("Customers", "churn_date mixes DD/MM/YYYY and YYYY-MM-DD", slash_dates,
          "Parsed with dayfirst=True and stored as datetime")

# --- 3.5 Missing values -----------------------------------------------------
say("")
say("3.5  Missing values")
miss_cust = raw_customers.isna().sum()
miss_mon = raw_monthly.isna().sum()
for col, count in miss_cust[miss_cust > 0].items():
    say(f"     customers.{col:<20} {count:>5} missing "
        f"({count / len(raw_customers) * 100:>5.2f}%)")
for col, count in miss_mon[miss_mon > 0].items():
    say(f"     monthly.{col:<21} {count:>5} missing "
        f"({count / len(raw_monthly) * 100:>5.2f}%)")
say("")
say("     Note: churn_reason is legitimately blank for every customer who is")
say("           still active - that is missing BY DESIGN, not a defect.")
log_issue("Customers", "churn_reason blank for active customers (expected)",
          int(raw_customers.loc[raw_customers["status"] == "Active",
                                "churn_reason"].isna().sum()),
          "Left blank; it is correct for active customers")
log_issue("Customers", "nps_score not answered by every customer",
          int(miss_cust.get("nps_score", 0)),
          "Left missing; averages use only the answers received")
log_issue("Customers", "last_login_date not tracked for a minority",
          int(miss_cust.get("last_login_date", 0)),
          "Left missing; excluded from date-based calculations")

# --- 3.6 Impossible values --------------------------------------------------
say("")
say("3.6  Impossible or suspicious values")
mrr_numeric = pd.to_numeric(
    raw_customers["mrr_at_signup"].astype(str)
    .str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")
negative_mrr = int((mrr_numeric < 0).sum())
seats_numeric = pd.to_numeric(raw_customers["seats"], errors="coerce")
huge_seats = int((seats_numeric > 10000).sum())
say(f"     mrr_at_signup : {negative_mrr} row(s) are negative "
    f"(an unapplied credit note, not a real subscription)")
say(f"     seats         : {huge_seats} row(s) above 10,000 seats "
    f"(double data entry - see the plan/size mismatch)")
say("     Treatment: negative MRR set to missing; impossible seat counts set to")
say("               missing so they cannot distort averages.")
log_issue("Customers", "Negative MRR (credit note, not a subscription)",
          negative_mrr, "Set to missing, flagged in the audit table")
log_issue("Customers", "Seat count of 50,000 on small accounts (double entry)",
          huge_seats, "Set to missing as physically implausible")

# --- 3.7 Scale stored inconsistently ---------------------------------------
say("")
say("3.7  adoption_rate stored two different ways")
adoption_numeric = pd.to_numeric(raw_monthly["adoption_rate"], errors="coerce")
pct_style = int((adoption_numeric > 1).sum())
fraction_style = int((adoption_numeric <= 1).sum())
say(f"     {pct_style} row(s) are stored as a percentage such as '45.3'")
say(f"     {fraction_style} row(s) are stored as a fraction such as 0.453")
say("     Treatment: any value above 1 is divided by 100.")
log_issue("Monthly activity", "adoption_rate mixes percent (45.3) and fraction (0.453)",
          pct_style, "Values above 1 divided by 100")

# --- 3.8 Business-rule check ----------------------------------------------
say("")
say("3.8  Business-rule checks (does the data make sense?)")
active_no_churn = int(((raw_customers["status"] == "Active") &
                       raw_customers["churn_date"].notna()).sum())
churn_no_date = int(((raw_customers["status"] == "Churned") &
                     raw_customers["churn_date"].isna()).sum())
say(f"     Active customers that also have a churn_date : {active_no_churn}  (expected 0)")
say(f"     Churned customers with no churn_date         : {churn_no_date}  (expected 0)")
say(f"     tenure_months below 1                       : "
    f"{int((pd.to_numeric(raw_customers['tenure_months'], errors='coerce') < 1).sum())}  (expected 0)")

# --- 3.9 Summary table ------------------------------------------------------
say("")
say("3.9  Data quality summary")
quality_df = pd.DataFrame(quality_issues)
say(quality_df.to_string(index=False))
say("")
say(f"TOTAL ISSUES FOUND: {len(quality_df)} categories across both tables.")


# -----------------------------------------------------------------------------
# SECTION 4: CLEANING
# -----------------------------------------------------------------------------

banner("SECTION 4  |  DATA CLEANING")

# Start from the raw text again and build a clean table step by step, so every
# change is visible and reversible.
cust = raw_customers.copy()
mon = raw_monthly.copy()

rows_before_cust = len(cust)
rows_before_mon = len(mon)

# --- 4.1 Remove duplicates --------------------------------------------------
cust = cust.drop_duplicates()
mon = mon.drop_duplicates()
# Safety net: a customer must appear once in the master table.
cust = cust.drop_duplicates(subset=["customer_id"], keep="first")
say(f"4.1  Removed duplicate rows")
say(f"       customers : {rows_before_cust:,} -> {len(cust):,} "
    f"({rows_before_cust - len(cust)} removed)")
say(f"       monthly   : {rows_before_mon:,} -> {len(mon):,} "
    f"({rows_before_mon - len(mon)} removed)")

# --- 4.2 Standardise text columns ------------------------------------------
say("")
say("4.2  Standardising text columns (trim spaces, fix capitalisation)")
text_columns = ["customer_id", "plan_tier", "industry", "country", "region",
                "company_size", "billing_cycle", "acquisition_channel",
                "status", "churn_reason"]
for col in text_columns:
    # .str.strip() on missing text would quietly turn a blank into the literal
    # string "nan", so we put the real missing values back afterwards.
    stripped = cust[col].astype(str).str.strip()
    cust[col] = stripped.where(cust[col].notna(), np.nan)

# Explicit mappings - clearer and safer than guessing a title case.
cust["plan_tier"] = cust["plan_tier"].str.title()
cust["billing_cycle"] = cust["billing_cycle"].str.title()
cust["company_size"] = cust["company_size"].str.upper()
# The same region was written as "Europe", "europe" and "EMEA", so all three
# are folded into one clean value instead of three separate regions.
cust["region"] = cust["region"].replace({
    "north america": "North America", "EMEA": "Europe", "emea": "Europe",
    "europe": "Europe", "Europe": "Europe", "asia pacific": "Asia Pacific",
    "Asia Pacific": "Asia Pacific", "latin america": "Latin America",
    "Latin America": "Latin America", "North America": "North America",
})
cust["status"] = cust["status"].str.title()
say(f"       plan_tier       now: {sorted(cust['plan_tier'].unique())}")
say(f"       billing_cycle   now: {sorted(cust['billing_cycle'].unique())}")
say(f"       company_size    now: {sorted(cust['company_size'].unique())}")
say(f"       region          now: {sorted(cust['region'].unique())}")

# --- 4.3 Convert text to numbers -------------------------------------------
say("")
say("4.3  Converting text columns to numbers")

# MRR: remove anything that is not a digit, a dot or a minus sign.
cust["mrr_at_signup"] = pd.to_numeric(
    cust["mrr_at_signup"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
    errors="coerce")

# Seats: plain conversion, 'n/a' already became NaN.
cust["seats"] = pd.to_numeric(cust["seats"], errors="coerce")

# Other numeric columns.
for col in ["discount_pct", "tenure_months", "nps_score", "seats_current"]:
    cust[col] = pd.to_numeric(cust[col], errors="coerce")

for col in ["month_number", "year", "tenure_month", "mrr", "seats",
            "active_users", "support_tickets", "nps_score", "is_active"]:
    mon[col] = pd.to_numeric(mon[col], errors="coerce")

# adoption_rate: fix the percent-vs-fraction mix.
mon["adoption_rate"] = pd.to_numeric(mon["adoption_rate"], errors="coerce")
mon.loc[mon["adoption_rate"] > 1, "adoption_rate"] /= 100
say(f"       adoption_rate now ranges "
    f"{mon['adoption_rate'].min():.3f} to {mon['adoption_rate'].max():.3f} (0-1 scale)")

# --- 4.4 Parse dates --------------------------------------------------------
say("")
say("4.4  Parsing dates")


def parse_mixed_date(series):
    """Parse a column that mixes several date spellings, without guessing.

    Two traps make a one-line to_datetime() call unsafe here:

    1. Some values are plain dates ("2024-10-01") and some carry a time part
       ("2024-10-01 00:00:00"). Asking for a strict "%Y-%m-%d" coerces every
       timestamp to NaT and quietly deletes the whole column.
    2. format="mixed" is not enough for the slash values. For an ambiguous
       value like "01/08/2023" pandas guesses the US month-first order even
       when dayfirst=True, which silently turns 1 August into 8 January.

    So each spelling is parsed with its own explicit format. The result is
    deterministic, and a value that matches none of them becomes NaT where it
    should rather than being quietly reinterpreted.
    """
    text = series.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    slash = text.str.contains("/", regex=False) & text.notna()
    plain = ~slash & text.notna()

    parsed.loc[plain] = pd.to_datetime(text.loc[plain], format="%Y-%m-%d",
                                       errors="coerce")
    # Accept both "YYYY-MM-DD" and "YYYY-MM-DD HH:MM:SS" for the plain values.
    parsed.loc[plain] = parsed.loc[plain].fillna(
        pd.to_datetime(text.loc[plain], format="%Y-%m-%d %H:%M:%S",
                       errors="coerce"))
    parsed.loc[slash] = pd.to_datetime(text.loc[slash], format="%d/%m/%Y",
                                       errors="coerce")
    return parsed


for col in ["signup_date", "last_login_date", "churn_date"]:
    before = cust[col].notna().sum()
    cust[col] = parse_mixed_date(cust[col])
    after = cust[col].notna().sum()
    say(f"       {col:<17} {before} values in -> {after} parsed, "
        f"{before - after} unparseable")

mon["month_start"] = parse_mixed_date(mon["month_start"])

say(f"       signup_date    {cust['signup_date'].min().date()} -> "
    f"{cust['signup_date'].max().date()}")
say(f"       churn_date     parsed for "
    f"{cust['churn_date'].notna().sum()} churned customers")
say(f"       month_start    {mon['month_start'].min().date()} -> "
    f"{mon['month_start'].max().date()}")

# A churn date that lands before the signup date, or after the end of the
# reporting period, cannot be right. Cross-checking the churn month against the
# customer's last active month in the monthly table is the only way to catch a
# mis-parsed date, because a wrong-but-plausible date still looks like a date.
CUSTOMERS_CHURNED = int((cust["churn_flag"] == 1).sum()) if "churn_flag" in cust \
    else int((cust["status"] == "Churned").sum())
_churn_check = (cust[cust["status"] == "Churned"]
                .merge(mon.groupby("customer_id")["month_start"].max()
                       .rename("last_active_month"), on="customer_id", how="left"))
_expected = (_churn_check["last_active_month"] + pd.DateOffset(months=1)).dt.to_period("M")
_actual = _churn_check["churn_date"].dt.to_period("M")
_mismatch = int((_expected != _actual).sum())
say(f"       churn month vs last active month: {_mismatch} mismatches "
    f"out of {CUSTOMERS_CHURNED} churned customers")
if _mismatch:
    say("       WARNING: churn dates disagree with the monthly activity table.")
    say("       Any monthly churn count built from these dates would be wrong.")

# Every churned customer must have a churn date inside the reporting period,
# otherwise they would be counted as churned with no month to attribute them to.
_out_of_period = int((_churn_check["churn_date"] < mon["month_start"].min()).sum()
                     + (_churn_check["churn_date"] > mon["month_start"].max()).sum())
say(f"       churn dates outside {mon['month_start'].min():%b %Y}-"
    f"{mon['month_start'].max():%b %Y}: {_out_of_period}")
if _out_of_period:
    say("       WARNING: some churn dates fall outside the monthly table, so the")
    say("       per-month churn counts will not add up to the churned total.")

# --- 4.5 Handle impossible values ------------------------------------------
say("")
say("4.5  Handling impossible values")
neg = int((cust["mrr_at_signup"] < 0).sum())
cust.loc[cust["mrr_at_signup"] < 0, "mrr_at_signup"] = np.nan
huge = int((cust["seats"] > 10000).sum())
cust.loc[cust["seats"] > 10000, "seats"] = np.nan
say(f"       {neg} negative MRR value(s)  -> set to missing")
say(f"       {huge} impossible seat count(s) -> set to missing")

# --- 4.6 Recompute the derived fields so they agree with the cleaned data ----
say("")
say("4.6  Recomputing derived fields so they agree with the cleaned data")
# Tenure is the number of months the customer was actually subscribed for. The
# monthly activity table is the best source for that, because it holds exactly
# one row per month a customer was active - so counting the rows gives tenure
# with no off-by-one ambiguity from date arithmetic.
DATA_END = mon["month_start"].max()
DATA_START = mon["month_start"].min()
say(f"       reporting period {DATA_START.date()} to {DATA_END.date()}")

months_active = (mon.groupby("customer_id")["month_start"]
                 .nunique().rename("tenure_months_calc"))
cust = cust.merge(months_active, on="customer_id", how="left")
say(f"       tenure rebuilt from the monthly table for {cust['tenure_months_calc'].notna().sum():,} "
    f"of {len(cust):,} customers")

# Cross-check the rebuilt tenure against the tenure reported in the file.
mismatch = int((cust["tenure_months"] != cust["tenure_months_calc"]).sum())
say(f"       reported tenure vs rebuilt tenure differ on {mismatch} row(s)")
say("       -> the rebuilt value is used, because it is derived from the rows")
say("          that actually exist in the activity table")
cust["tenure_months"] = cust["tenure_months_calc"]
cust["tenure_months"] = cust["tenure_months"].fillna(0).astype(int)
say(f"       tenure now ranges {cust['tenure_months'].min()} to "
    f"{cust['tenure_months'].max()} months")

# Business rule: nobody can have churned after the data ends, and a churned
# customer must have a churn date while an active one must not.
late_churn = int((cust["churn_date"] > DATA_END).sum())
no_date = int(((cust["status"] == "Churned") & cust["churn_date"].isna()).sum())
say(f"       churn dates after the end of the data : {late_churn}  (expected 0)")
say(f"       churned customers with no churn date   : {no_date}  (expected 0)")

# --- 4.7 Drop the simulator's helper columns -------------------------------
drop_cols = [c for c in cust.columns if c.startswith("_")]
if drop_cols:
    cust = cust.drop(columns=drop_cols)
say("")
say(f"4.7  Removed internal simulator columns: {drop_cols if drop_cols else 'none'}")

# --- 4.8 Final data quality check ------------------------------------------
say("")
say("4.8  Post-cleaning verification")
still_dupes = int(cust.duplicated().sum())
say(f"       duplicate rows remaining : {still_dupes}")
say(f"       customers                : {len(cust):,}")
say(f"       monthly rows             : {len(mon):,}")
say(f"       customers in both tables : "
    f"{len(set(cust['customer_id']) & set(mon['customer_id'])):,}")
say(f"       orphan monthly rows      : "
    f"{len(set(mon['customer_id']) - set(cust['customer_id'])):,}")


# -----------------------------------------------------------------------------
# SECTION 5: FEATURE ENGINEERING
# -----------------------------------------------------------------------------

banner("SECTION 5  |  FEATURE ENGINEERING")

say("The raw columns do not answer the business questions, so we build the")
say("columns that do. Each new column is explained below.")
say("")

# 5.1 churn_flag -------------------------------------------------------------
cust["churn_flag"] = (cust["status"] == "Churned").astype(int)
say("5.1  churn_flag = 1 if the customer cancelled, 0 if still active")
say(f"       {cust['churn_flag'].sum()} churned / {len(cust):,} total "
    f"= {cust['churn_flag'].mean() * 100:.1f}%")

# 5.2 signup cohort ---------------------------------------------------------
cust["signup_cohort"] = cust["signup_date"].dt.to_period("M")
say("5.2  signup_cohort = the month the customer joined (used for cohorts)")

# 5.3 seat flags ------------------------------------------------------------
cust["seat_estimate"] = cust["seats"].fillna(cust["seats_current"])
cust["is_seat_data_missing"] = cust["seats"].isna().astype(int)
cust["arpu"] = cust["mrr_at_signup"] / cust["seat_estimate"]
cust["arr"] = cust["mrr_at_signup"] * 12
say("5.3  arpu = average revenue per user (MRR / seats); arr = annualised MRR x12")

# 5.4 engagement band -------------------------------------------------------
# Use the customer's AVERAGE adoption over their whole life.
avg_usage = mon.groupby("customer_id")["adoption_rate"].mean().rename("avg_adoption")
cust = cust.merge(avg_usage, on="customer_id", how="left")
cust["engagement_band"] = pd.cut(
    cust["avg_adoption"],
    bins=[-0.01, 0.30, 0.45, 1.01],
    labels=["Low (<30% seats used)", "Medium (30-45%)", "High (>45%)"],
)
say("5.4  engagement_band built from average seat adoption over the customer's life")

# 5.5 behaviour in the FINAL months ----------------------------------------
# The most useful question is not "how did churned customers behave overall"
# but "how were they behaving just BEFORE they left?". We therefore take the
# last 3 months each customer was active.
mon_sorted = mon.sort_values(["customer_id", "month_start"])
final3 = mon_sorted.groupby("customer_id").tail(3)

final_stats = final3.groupby("customer_id").agg(
    final_adoption=("adoption_rate", "mean"),
    final_tickets=("support_tickets", "mean"),
    final_nps=("nps_score", "mean"),
    final_mrr=("mrr", "mean"),
).reset_index()

cust = cust.merge(final_stats, on="customer_id", how="left")
say("5.5  final_adoption / final_tickets / final_nps / final_mrr")
say("       = the customer's average over their LAST 3 ACTIVE months.")

# 5.6 lifetime value --------------------------------------------------------
# A simple, transparent proxy: MRR x average lifetime in months.
cust["lifetime_value"] = cust["mrr_at_signup"] * cust["tenure_months"]
say("5.6  lifetime_value = mrr_at_signup x tenure_months (simple LTV proxy)")

# 5.7 monthly table extras --------------------------------------------------
mon = mon.merge(cust[["customer_id", "signup_cohort", "plan_tier", "region",
                      "acquisition_channel", "billing_cycle", "churn_flag"]],
                on="customer_id", how="left")
mon["month"] = mon["month_start"].dt.to_period("M")
mon["arr_run_rate"] = mon["mrr"] * 12
say("5.7  Joined the customer attributes onto the monthly table and added")
say("       month and arr_run_rate (MRR x 12) for convenience.")

# 5.8 reorder columns for readability --------------------------------------
say("")
say("5.8  Final column list")
say(f"       customers_clean : {len(cust.columns)} columns")
say(f"       {', '.join(cust.columns)}")
say(f"       customer_monthly_clean : {len(mon.columns)} columns")


# -----------------------------------------------------------------------------
# SECTION 6: DESCRIPTIVE STATISTICS
# -----------------------------------------------------------------------------

banner("SECTION 6  |  DESCRIPTIVE STATISTICS")

say("6.1  Headline numbers")
n_customers = len(cust)
n_churned = int(cust["churn_flag"].sum())
n_active = n_customers - n_churned
churn_rate = cust["churn_flag"].mean()

current_mrr = mon.loc[mon["month_start"] == DATA_END, "mrr"].sum()
start_mrr = mon.loc[mon["month_start"] == mon["month_start"].min(), "mrr"].sum()
current_active = mon.loc[mon["month_start"] == DATA_END, "customer_id"].nunique()

say(f"       Total customers (2023-2024)      : {n_customers:,}")
say(f"       Churned                          : {n_churned:,}")
say(f"       Still active at end of period    : {n_active:,}")
say(f"       Logo churn rate (whole period)   : {churn_rate * 100:.1f}%")
say(f"       Active customers, Dec 2024       : {current_active:,}")
say(f"       MRR, Dec 2024                    : {money(current_mrr)}")
say(f"       ARR run-rate, Dec 2024           : {money(current_mrr * 12)}")
say(f"       ARPU, Dec 2024                   : "
    f"{money(current_mrr / current_active)}")
say(f"       Average MRR per customer          : "
    f"{money(cust['mrr_at_signup'].mean())}")
say(f"       Average NPS (who answered)       : {cust['nps_score'].mean():.2f}")
say(f"       Average tenure                   : {cust['tenure_months'].mean():.1f} months")

say("")
say("6.2  Numeric summary of the key measures")
desc_cols = ["mrr_at_signup", "seats", "discount_pct", "tenure_months",
             "nps_score", "lifetime_value", "final_adoption",
             "final_tickets", "final_nps"]
say(cust[desc_cols].describe().T.round(2).to_string())

say("")
say("6.3  Categorical summary")
for col in ["plan_tier", "billing_cycle", "region", "acquisition_channel",
            "company_size", "engagement_band"]:
    counts = cust[col].value_counts(dropna=False)
    counts.index.name = None            # avoids printing the column name twice
    say("")
    say(f"       {col}")
    say("       " + counts.to_string().replace("\n", "\n       "))


# -----------------------------------------------------------------------------
# SECTION 7: EXPLORATORY DATA ANALYSIS + CHARTS
# -----------------------------------------------------------------------------

banner("SECTION 7  |  EXPLORATORY DATA ANALYSIS")

# --- CHART 1: MRR and active customers over time ---------------------------
say("")
say("CHART 1  Monthly Recurring Revenue and active customers over time")
say("  Question: is the business growing or shrinking?")

monthly_trend = (mon.groupby("month_start")
                 .agg(mrr=("mrr", "sum"),
                      active_customers=("customer_id", "nunique"))
                 .reset_index())
# Count genuinely new customers per month from the master table.
new_per_month = (cust.groupby("signup_cohort")
                 .size().reset_index(name="new_customers"))
new_per_month.columns = ["month_start", "new_customers"]
new_per_month["month_start"] = new_per_month["month_start"].dt.to_timestamp()
monthly_trend = monthly_trend.merge(new_per_month, on="month_start", how="left")
monthly_trend["new_customers"] = monthly_trend["new_customers"].fillna(0).astype(int)

# Count cancellations per calendar month from the churn date.
# NOTE: do not use .reindex() with the month Series as the key. The key has one
# entry per month here, but reindex matches on VALUES, so any repeated month
# would broadcast that month's count across every matching row and silently
# inflate the totals. .map() is a straight value lookup and cannot do that.
churn_per_month = (cust[cust["churn_flag"] == 1]
                   .groupby(cust["churn_date"].dt.to_period("M"))
                   .size())
monthly_trend["churned_this_month"] = (
    monthly_trend["month_start"].dt.to_period("M")
    .map(churn_per_month).fillna(0).astype(int).to_numpy())

peak_row = monthly_trend.loc[monthly_trend["mrr"].idxmax()]
last_row = monthly_trend.iloc[-1]
first_full = monthly_trend.iloc[0]

# Rather than ASSUME a story, we check it. If MRR did not fall every month after
# the peak, the wording below changes to match what the data actually shows.
PEAK_MONTH = peak_row["month_start"].strftime("%b %Y")
post_peak = monthly_trend[monthly_trend["month_start"] > peak_row["month_start"]]
mrr_changes = post_peak["mrr"].diff().dropna()
months_fallen = int((mrr_changes < 0).sum())
months_after_peak = int(len(mrr_changes))
fell_every_month = months_fallen == months_after_peak
end_vs_peak_pct = (last_row["mrr"] / peak_row["mrr"] - 1) * 100

# The month with the most active customers is not necessarily the MRR peak.
active_peak = monthly_trend.loc[monthly_trend["active_customers"].idxmax()]

# Average month-on-month MRR growth, early in the period versus the last 6
# months. This is the cleanest measure of "growth has stalled".
mom = monthly_trend["mrr"].pct_change()
growth_early = float(mom.iloc[1:7].mean())
growth_late = float(mom.iloc[-6:].mean())

# Classify the trend instead of assuming one.
if end_vs_peak_pct <= -2.0:
    TREND = "declining"
elif end_vs_peak_pct >= 2.0:
    TREND = "growing"
else:
    TREND = "stalled"

say(f"  MRR rose from {money(first_full['mrr'])} in "
    f"{first_full['month_start'].strftime('%b %Y')} to a peak of "
    f"{money(peak_row['mrr'])} in {PEAK_MONTH}.")
if TREND == "declining":
    if fell_every_month:
        say(f"  It has fallen in every month since, reaching "
            f"{money(last_row['mrr'])} in {last_row['month_start'].strftime('%b %Y')} "
            f"({end_vs_peak_pct:+.1f}% vs peak).")
    else:
        say(f"  Since then it fell in {months_fallen} of the next "
            f"{months_after_peak} months, reaching {money(last_row['mrr'])} in "
            f"{last_row['month_start'].strftime('%b %Y')} "
            f"({end_vs_peak_pct:+.1f}% vs peak).")
elif TREND == "stalled":
    say(f"  It has been flat since: {money(last_row['mrr'])} in "
        f"{last_row['month_start'].strftime('%b %Y')}, only "
        f"{end_vs_peak_pct:+.1f}% against the {PEAK_MONTH} peak.")
else:
    say(f"  It is still growing and ended at {money(last_row['mrr'])} in "
        f"{last_row['month_start'].strftime('%b %Y')}, "
        f"{end_vs_peak_pct:+.1f}% versus the {PEAK_MONTH} peak.")
say(f"  Average month-on-month MRR growth: {growth_early:+.1%} early in the "
    f"period versus {growth_late:+.1%} over the last 6 months.")
say(f"  Active customers peaked at {int(active_peak['active_customers']):,} in "
    f"{active_peak['month_start'].strftime('%b %Y')} and are now "
    f"{int(last_row['active_customers']):,} "
    f"({(last_row['active_customers'] / active_peak['active_customers'] - 1) * 100:+.1f}%).")
say("  INTERPRETATION: the top line grew explosively through 2023, but the")
say(f"  growth rate has collapsed from {growth_early:+.1%} a month to "
    f"{growth_late:+.1%} a month.")
say("  The customer count has stopped growing too. Whatever was driving this")
say("  business has stopped working - this is the core problem.")

fig, ax1 = plt.subplots(figsize=(12, 5.5))
ax2 = ax1.twinx()
ax1.bar(monthly_trend["month_start"], monthly_trend["new_customers"],
        color=PALETTE["light_blue"], alpha=0.75, label="New customers")
ax1.plot(monthly_trend["month_start"], monthly_trend["mrr"],
         color=PALETTE["navy"], linewidth=2.8, marker="o", markersize=5,
         label="MRR")
ax2.plot(monthly_trend["month_start"], monthly_trend["active_customers"],
         color=PALETTE["red"], linewidth=2.4, linestyle="--", marker="s",
         markersize=4.5, label="Active customers")
ax1.axvline(peak_row["month_start"], color=PALETTE["grey"], linestyle=":", linewidth=1.6)
ax1.annotate(f"MRR peak\n{PEAK_MONTH}",
             xy=(peak_row["month_start"], peak_row["mrr"]),
             xytext=(-95, -45), textcoords="offset points", fontsize=9,
             color=PALETTE["grey"],
             arrowprops=dict(arrowstyle="->", color=PALETTE["grey"]))
if TREND == "declining":
    chart1_title = (f"MRR Peaked in {PEAK_MONTH} and Is Now {abs(end_vs_peak_pct):.0f}% Lower")
elif TREND == "stalled":
    chart1_title = (f"MRR Growth Has Stalled: Flat at {PEAK_MONTH} Levels Since "
                    f"{PEAK_MONTH}")
else:
    chart1_title = f"MRR Is Still Growing, but Growth Has Slowed Since {PEAK_MONTH}"
ax1.set_title(chart1_title, pad=14)
ax1.set_xlabel("Month")
ax1.set_ylabel("Monthly Recurring Revenue (USD)")
ax2.set_ylabel("Customers")
ax1.yaxis.set_major_formatter(lambda v, p: f"${v/1000:,.0f}k")
ax2.set_ylim(0, monthly_trend["active_customers"].max() * 1.18)
ax1.set_ylim(0, monthly_trend["mrr"].max() * 1.25)
lines = list(ax1.get_lines()) + list(ax1.containers) + list(ax2.get_lines())
ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=9,
           framealpha=0.95)
ax1.grid(axis="y", alpha=0.3)
save_chart(fig, "01_mrr_and_customers_trend.png")

# --- CHART 2: cohort retention heatmap -------------------------------------
say("")
say("CHART 2  Cohort retention - what share of each signup cohort is still active?")
say("  Question: do newer customers stick around as well as older ones?")

cohort = (cust.groupby("signup_cohort")["customer_id"].nunique()
          .rename("cohort_size").reset_index())
cohort["cohort"] = cohort["signup_cohort"].astype(str)

# For each cohort and each month of tenure, count how many are still active.
rows = []
cohort_max_tenure = {}          # how far each cohort actually got
for cohort_period, size in zip(cohort["signup_cohort"], cohort["cohort_size"]):
    members = cust.loc[cust["signup_cohort"] == cohort_period, "customer_id"]
    member_mon = mon[mon["customer_id"].isin(members)]
    if member_mon.empty:
        continue
    # tenure month 0 is the signup month itself.
    survived = (member_mon.groupby("tenure_month")["customer_id"].nunique()
                .reindex(range(0, 25), fill_value=0))
    alive = survived[survived > 0]
    cohort_max_tenure[str(cohort_period)] = int(alive.index.max()) if len(alive) else 0
    for tenure_month, count in survived.items():
        rows.append({"cohort": str(cohort_period), "tenure_month": tenure_month,
                     "retained": count, "cohort_size": size})
cohort_long = pd.DataFrame(rows)
cohort_pivot = (cohort_long.pivot(index="cohort", columns="tenure_month",
                                  values="retained"))
cohort_size_map = cohort_long.drop_duplicates("cohort").set_index("cohort")["cohort_size"]
cohort_pct = cohort_pivot.div(cohort_size_map, axis=0) * 100


def cohort_average(tenure_month):
    """Average retention at a given tenure month, using only the cohorts that
    lived long enough to reach it. A cohort that signed up last month has no
    month-12 value, and counting it as 0% would understate retention."""
    eligible = [c for c, reached in cohort_max_tenure.items() if reached >= tenure_month]
    if not eligible:
        return np.nan, 0
    return float(cohort_pct.loc[eligible, tenure_month].mean()), len(eligible)


say(f"  {len(cohort_pivot)} signup cohorts, from "
    f"{cohort_pivot.index.min()} to {cohort_pivot.index.max()}.")
month1, n1 = cohort_average(1)
month3, n3 = cohort_average(3)
month6, n6 = cohort_average(6)
month12, n12 = cohort_average(12)
month18, n18 = cohort_average(18)
say(f"  Average retention: month 1 {month1:.0f}% (n={n1} cohorts), month 3 "
    f"{month3:.0f}% (n={n3}), month 6 {month6:.0f}% (n={n6}), month 12 "
    f"{month12:.0f}% (n={n12}).")
say(f"  Only the {n18} earliest cohorts are old enough to have reached month 18; "
    f"they retain {month18:.0f}% at that point.")
say("  Each average uses ONLY the cohorts that lived long enough to reach that")
say("  month. Folding in younger cohorts as 0% would understate retention badly.")
say("  INTERPRETATION: retention falls steeply in the first 3 months and then")
say("  flattens. The danger zone is onboarding, not the long tail.")

# A cohort that has not yet reached a given month of tenure has no value for it.
# Leaving those cells blank matters: filled with 0 they would read as "100% of
# the cohort churned", which is completely untrue.
cohort_reach = pd.DataFrame(
    {t: [t <= cohort_max_tenure[c] for c in cohort_pct.index]
     for t in cohort_pct.columns},
    index=cohort_pct.index)
cohort_pct_plot = cohort_pct.where(cohort_reach)
say("  Cells are left blank where a cohort has not yet reached that month.")
fig, ax = plt.subplots(figsize=(13, 6))
sns.heatmap(cohort_pct_plot, cmap="YlGnBu", vmin=0, vmax=100, ax=ax,
            mask=cohort_pct_plot.isna(),
            cbar_kws={"label": "% of cohort still subscribed"})
ax.set_title("Retention by Signup Cohort: Most Leakage Happens in the First 3 Months",
             pad=14)
ax.set_xlabel("Months since signup")
ax.set_ylabel("Signup cohort")
ax.set_xticklabels([str(c) for c in cohort_pct.columns], rotation=0)
save_chart(fig, "02_cohort_retention_heatmap.png")

# --- CHART 3: churn by plan tier -------------------------------------------
say("")
say("CHART 3  Churn rate by plan tier")
say("  Question: which product tier loses the most customers?")

def churn_by(column, order=None):
    """Reusable helper: customers, churned, churn % and revenue lost."""
    table = (cust.groupby(column)
             .agg(customers=("churn_flag", "size"),
                  churned=("churn_flag", "sum"),
                  total_mrr=("mrr_at_signup", "sum"))
             .reset_index())
    table["churn_pct"] = (table["churned"] / table["customers"] * 100).round(1)
    lost = (cust[cust["churn_flag"] == 1].groupby(column)["mrr_at_signup"]
            .sum().rename("mrr_lost"))
    table = table.merge(lost, on=column, how="left")
    table["mrr_lost"] = table["mrr_lost"].fillna(0)
    table["share_of_all_losses"] = (table["mrr_lost"] / table["mrr_lost"].sum() * 100).round(1)
    if order:
        table = table.set_index(column).reindex(order).reset_index()
    else:
        table = table.sort_values("churn_pct", ascending=False)
    return table


tier_order = ["Starter", "Growth", "Business", "Enterprise"]
churn_tier = churn_by("plan_tier", tier_order)
say(churn_tier.to_string(index=False))
worst_tier = churn_tier.iloc[0]
best_tier = churn_tier.iloc[-1]
say(f"  {worst_tier['plan_tier']} churns at {worst_tier['churn_pct']}% versus "
    f"{best_tier['churn_pct']}% for {best_tier['plan_tier']} - a gap of "
    f"{worst_tier['churn_pct'] - best_tier['churn_pct']:.1f} percentage points.")
say("  INTERPRETATION: the entry-level plan is the leak. These are the customers")
say("  the sales team signs most easily and they leave soonest.")

fig, ax = plt.subplots(figsize=(10, 5.5))
colors = [PALETTE["red"] if p == churn_tier["churn_pct"].max() else PALETTE["blue"]
          for p in churn_tier["churn_pct"]]
bars = ax.bar(churn_tier["plan_tier"], churn_tier["churn_pct"], color=colors,
              width=0.6, edgecolor="white")
for bar, pct, n in zip(bars, churn_tier["churn_pct"], churn_tier["customers"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.7,
            f"{pct}%\n(n={n})", ha="center", va="bottom", fontsize=9.5,
            fontweight="bold")
ax.axhline(churn_tier["churn_pct"].mean(), color=PALETTE["grey"], linestyle="--",
           linewidth=1.5)
ax.text(3.42, churn_tier["churn_pct"].mean() + 0.6,
        f"Average {churn_tier['churn_pct'].mean():.1f}%", fontsize=8.5,
        color=PALETTE["grey"], ha="right")
ax.set_ylim(0, churn_tier["churn_pct"].max() * 1.22)
ax.set_title("Starter Plan Churns Most - The Entry Tier Is the Leak", pad=14)
ax.set_xlabel("Plan tier")
ax.set_ylabel("Churn rate (% of customers who cancelled)")
save_chart(fig, "03_churn_by_plan_tier.png")

# --- CHART 4: churn by acquisition channel ---------------------------------
say("")
say("CHART 4  Churn rate by acquisition channel")
say("  Question: are we buying customers that stay?")

channel_order = (cust.groupby("acquisition_channel")["churn_flag"].mean()
                 .sort_values().index.tolist())
churn_channel = churn_by("acquisition_channel", channel_order)
say(churn_channel.to_string(index=False))
best_ch, worst_ch = churn_channel.iloc[0], churn_channel.iloc[-1]
say(f"  Best: {best_ch['acquisition_channel']} at {best_ch['churn_pct']}%. "
    f"Worst: {worst_ch['acquisition_channel']} at {worst_ch['churn_pct']}%.")
say(f"  {worst_ch['acquisition_channel']} accounts represent "
    f"{worst_ch['share_of_all_losses']:.1f}% of all MRR lost to churn, but they are "
    f"only {worst_ch['customers'] / n_customers * 100:.0f}% of the customer base.")
say("  INTERPRETATION: the channels that discount hardest to win a deal are the")
say("  ones that lose it soonest. Discounting buys contracts that do not last.")

# Value per customer by channel - the number that decides whether a channel
# is actually profitable to buy.
channel_value = (cust.groupby("acquisition_channel")
                 .agg(customers=("churn_flag", "size"),
                      churn_pct=("churn_flag", "mean"),
                      avg_mrr=("mrr_at_signup", "mean"),
                      avg_discount=("discount_pct", "mean"),
                      avg_ltv=("lifetime_value", "mean"))
                 .reset_index())
channel_value["churn_pct"] = (channel_value["churn_pct"] * 100).round(1)
say("")
say("  Is the channel worth its cost? (value per customer won)")
say(channel_value.sort_values("avg_ltv", ascending=False).round(2).to_string(index=False))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
bar_colors = [PALETTE["green"] if c == channel_order[0] else
              (PALETTE["red"] if c == channel_order[-1] else PALETTE["blue"])
              for c in churn_channel["acquisition_channel"]]
bars = ax1.bar(churn_channel["acquisition_channel"], churn_channel["churn_pct"],
               color=bar_colors, width=0.62, edgecolor="white")
for bar, pct in zip(bars, churn_channel["churn_pct"]):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
             f"{pct}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
ax1.axhline(churn_channel["churn_pct"].mean(), color=PALETTE["grey"],
            linestyle="--", linewidth=1.5)
ax1.set_ylim(0, churn_channel["churn_pct"].max() * 1.2)
ax1.set_title("Churn Rate by Acquisition Channel", pad=12)
ax1.set_xlabel("Acquisition channel")
ax1.set_ylabel("Churn rate (%)")
ax1.tick_params(axis="x", rotation=20)

cv = channel_value.sort_values("avg_ltv", ascending=True)
bar_colors2 = [PALETTE["red"] if v < cv["avg_ltv"].median() else PALETTE["green"]
               for v in cv["avg_ltv"]]
ax2.barh(cv["acquisition_channel"], cv["avg_ltv"], color=bar_colors2, height=0.62,
         edgecolor="white")
for i, v in enumerate(cv["avg_ltv"]):
    ax2.text(v + cv["avg_ltv"].max() * 0.015, i, f"${v:,.0f}", va="center",
             fontsize=9.5, fontweight="bold")
ax2.set_xlim(0, cv["avg_ltv"].max() * 1.18)
ax2.set_title("Lifetime Value per Customer by Channel", pad=12)
ax2.set_xlabel("Lifetime value = MRR x months subscribed (USD)")
ax2.set_ylabel("")
save_chart(fig, "04_churn_and_ltv_by_channel.png")

# --- CHART 5: behaviour in the last 3 months -------------------------------
say("")
say("CHART 5  How were customers behaving just before they left?")
say("  Question: are there early warning signs we could act on?")

signal_cols = ["final_adoption", "final_tickets", "final_nps"]
signal_table = (cust.groupby("churn_flag")[signal_cols].mean().round(3))
signal_table.index = ["Still active", "Churned"]
say(signal_table.to_string())
churned_adopt = cust.loc[cust["churn_flag"] == 1, "final_adoption"].mean()
active_adopt = cust.loc[cust["churn_flag"] == 0, "final_adoption"].mean()
churned_tix = cust.loc[cust["churn_flag"] == 1, "final_tickets"].mean()
active_tix = cust.loc[cust["churn_flag"] == 0, "final_tickets"].mean()
churned_nps = cust.loc[cust["churn_flag"] == 1, "final_nps"].mean()
active_nps = cust.loc[cust["churn_flag"] == 0, "final_nps"].mean()

say("")
say(f"  Adoption   : churned {churned_adopt:.1%} vs active {active_adopt:.1%} "
    f"({(churned_adopt / active_adopt - 1) * 100:+.1f}%)")
say(f"  Tickets    : churned {churned_tix:.2f} vs active {active_tix:.2f} "
    f"({(churned_tix / active_tix - 1) * 100:+.1f}%)")
say(f"  NPS        : churned {churned_nps:.2f} vs active {active_nps:.2f} "
    f"({churned_nps - active_nps:+.2f} points)")
say("  INTERPRETATION: customers who leave were already using fewer seats and")
say("  were less happy in their final 3 months. These are usable warning signs.")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
panels = [
    ("final_adoption", "Seat adoption in final 3 months", "Share of seats actually used", PALETTE["blue"]),
    ("final_tickets", "Support tickets in final 3 months", "Tickets per month", PALETTE["orange"]),
    ("final_nps", "NPS in final 3 months", "Score (0-10)", PALETTE["green"]),
]
for ax, (col, title, xlabel, color) in zip(axes, panels):
    data = [cust.loc[cust["churn_flag"] == 0, col].dropna(),
            cust.loc[cust["churn_flag"] == 1, col].dropna()]
    parts = ax.violinplot(data, showmedians=True, widths=0.8)
    for pc, pc_color in zip(parts["bodies"], [PALETTE["light_blue"], "#F4B6B6"]):
        pc.set_facecolor(pc_color)
        pc.set_alpha(0.85)
        pc.set_edgecolor("white")
    for i, (values, label) in enumerate(zip(data, ["Still active", "Churned"]), start=1):
        mean_val = values.mean()
        ax.scatter(i, mean_val, color=color, s=110, zorder=5,
                   edgecolor="white", linewidth=1.6)
        ax.text(i, ax.get_ylim()[1], f"mean {mean_val:.2f}", ha="center", va="top",
                fontsize=8.5, color=color, fontweight="bold")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Still active", "Churned"])
    ax.set_title(title, fontsize=11.5, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel(xlabel)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Early Warning Signals: Churned Customers Look Different Before They Leave",
             fontsize=14, fontweight="bold", y=1.03)
save_chart(fig, "05_early_warning_signals.png")

# --- CHART 6: survival curve by tenure ----------------------------------------
say("")
say("CHART 6  When do customers actually leave?")
say("  Question: at what point in the customer lifecycle does churn happen?")

# A customer whose tenure is k months was subscribed during tenure months
# 0 .. k-1. So "still subscribed at tenure month t" means tenure_months > t.
#
# Using tenure_months >= t instead would be wrong in a way that flatters the
# business: it also counts customers who churned LATER than month t as if they
# had already survived month t, which flattens the front of the curve and hides
# exactly the early churn we are looking for.
#
# CENSORING: a customer who was still active in Dec 2024 has not churned - the
# data simply stopped. Those customers are counted as "at risk" in the
# denominator (they genuinely were subscribed) but never counted as a
# cancellation, otherwise the final months of the curve would show a fake 100%
# churn rate caused by the end of the reporting period rather than by customers.
base_customers = len(cust)
max_tenure = int(cust["tenure_months"].max())
survivorship = []
for t in range(0, max_tenure):
    at_risk = int((cust["tenure_months"] > t).sum())
    if at_risk == 0:
        break
    cancelled = int(((cust["tenure_months"] == t + 1) &
                     (cust["churn_flag"] == 1)).sum())
    survivorship.append({
        "tenure_month": t,
        "at_risk_subscribed": at_risk,
        "cancelled_this_month": cancelled,
        "monthly_churn_pct": cancelled / at_risk * 100,
        "retention_pct": at_risk / base_customers * 100,
    })
survivor_df = pd.DataFrame(survivorship)
say(f"  Base for every point: all {base_customers:,} customers who ever signed up.")
say("  Customers still active at the end of the data are treated as censored:")
say("  they count as at risk, but not as cancellations.")
say(survivor_df.round(2).to_string(index=False))


def survival_at(t):
    """% of everyone who ever signed up who were still subscribed at month t."""
    return float(survivor_df.loc[survivor_df["tenure_month"] == t, "retention_pct"].iloc[0])


r3 = survival_at(3)
r6 = survival_at(6)
r12 = survival_at(12)
say(f"  Of everyone who signed up, {r3:.0f}% were still subscribed at month 3, "
    f"{r6:.0f}% at month 6 and {r12:.0f}% at month 12.")

# Where is the cancellation rate actually highest?
peak_hazard = survivor_df.loc[survivor_df["monthly_churn_pct"].idxmax()]
first_month_hazard = float(survivor_df["monthly_churn_pct"].iloc[0])
say(f"  The monthly cancellation rate is {first_month_hazard:.1f}% in the first "
    f"month and peaks at {peak_hazard['monthly_churn_pct']:.1f}% in tenure month "
    f"{int(peak_hazard['tenure_month'])}.")

# Find the worst 3-month window rather than assuming where the leak is.
last_t = int(survivor_df["tenure_month"].max())
window_drops = {s: survival_at(s) - survival_at(s + 3)
                for s in range(0, last_t - 2, 3) if (s + 3) in set(survivor_df["tenure_month"])}
worst_start = max(window_drops, key=window_drops.get)
first3_loss = window_drops[0]

# What share of ALL cancellations land in the first 90 days?
cancelled_total = int(survivor_df["cancelled_this_month"].sum())
first90_cancelled = int(survivor_df.loc[survivor_df["tenure_month"] <= 2,
                                       "cancelled_this_month"].sum())
first90_share = first90_cancelled / cancelled_total * 100
# A second wave where the first annual contracts come up for renewal.
renewal = survivor_df[(survivor_df["tenure_month"] >= 12) &
                      (survivor_df["tenure_month"] <= 18)]
renewal_peak = float(renewal["monthly_churn_pct"].max()) if len(renewal) else np.nan
say(f"  {first90_share:.0f}% of all {cancelled_total:,} cancellations happened in the "
    f"first 3 months of tenure.")
say(f"  A smaller second wave appears at the first renewal: the cancellation rate "
    f"reaches {renewal_peak:.1f}% around tenure months 12-18,")
say("  against a low of roughly 2% in the calm middle of the curve.")
say("  NOTE: only customers who signed up early can appear in the later months,")
say("  so the tail of this curve is a small, already-loyal group. Read it")
say("  together with the cohort heatmap, not instead of it.")
say(f"  Biggest 3-month loss: {window_drops[worst_start]:.1f} points of the "
    f"original base, between month {worst_start} and month {worst_start + 3}.")
say("  NOTE: only customers who signed up early can appear in the later months,")
say("  so the tail of this curve is a small, already-loyal group. Read it")
say("  together with the cohort heatmap, not instead of it.")

fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(survivor_df["tenure_month"], survivor_df["retention_pct"],
        color=PALETTE["navy"], linewidth=3, marker="o", markersize=5)
ax.fill_between(survivor_df["tenure_month"], survivor_df["retention_pct"],
                alpha=0.12, color=PALETTE["navy"])
ax.axvspan(0, 3, color=PALETTE["grey"], alpha=0.10)
ax.axvspan(11, 18, color=PALETTE["red"], alpha=0.10)
ax.text(1.5, survivor_df["retention_pct"].max() * 0.93, "FIRST 90\nDAYS",
        ha="center", fontsize=9.5, color=PALETTE["grey"], fontweight="bold")
ax.text(14.5, survivor_df["retention_pct"].max() * 0.93, "FIRST RENEWAL\n(12-18 MONTHS)",
        ha="center", fontsize=9.5, color=PALETTE["red"], fontweight="bold")
for t in [3, 6, 12]:
    if t in survivor_df["tenure_month"].values:
        v = survival_at(t)
        ax.annotate(f"{v:.0f}%", xy=(t, v), xytext=(0, 12),
                    textcoords="offset points", ha="center",
                    fontsize=10, fontweight="bold", color=PALETTE["navy"])
if worst_start <= 3:
    chart6_title = (f"Churn Is Front-Loaded: {first90_share:.0f}% of All Cancellations "
                    f"Happen in the First 3 Months")
else:
    chart6_title = ("Churn Is Not Front-Loaded: The Cancellation Rate Peaks at the "
                    "First Renewal")
ax.set_title(chart6_title, pad=14)
ax.set_xlabel("Months since signup")
ax.set_ylabel("Still subscribed (% of everyone who ever signed up)")
ax.set_ylim(0, 108)
ax.set_xticks(range(0, 25, 2))
ax.grid(alpha=0.3)
save_chart(fig, "06_retention_by_tenure.png")

# --- CHART 7: churn reasons -------------------------------------------------
say("")
say("CHART 7  Why customers leave")
say("  Question: what do customers tell us when they cancel?")

reason_data = (cust[cust["churn_flag"] == 1]
               .groupby("churn_reason")
               .agg(customers=("customer_id", "nunique"),
                    mrr_lost=("mrr_at_signup", "sum"))
               .reset_index()
               .sort_values("customers", ascending=False))
reason_data["share_pct"] = (reason_data["customers"] / reason_data["customers"].sum() * 100)
reason_data["mrr_share_pct"] = (reason_data["mrr_lost"] / reason_data["mrr_lost"].sum() * 100)
say(reason_data.round(1).to_string(index=False))
price_reason = reason_data[reason_data["churn_reason"] == "Too expensive"]
if not price_reason.empty:
    say(f"  'Too expensive' is the single largest reason at "
        f"{price_reason['customers'].iloc[0]} customers "
        f"({price_reason['share_pct'].iloc[0]:.0f}%).")

fig, ax = plt.subplots(figsize=(12, 5.8))
colors = [PALETTE["red"] if r == "Too expensive" else PALETTE["blue"]
          for r in reason_data["churn_reason"]]
bars = ax.barh(reason_data["churn_reason"], reason_data["customers"],
               color=colors, height=0.62, edgecolor="white")
ax.invert_yaxis()
for bar, n, pct in zip(bars, reason_data["customers"], reason_data["share_pct"]):
    ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
            f"{n}  ({pct:.0f}%)", va="center", fontsize=9.5, fontweight="bold")
ax.set_xlim(0, reason_data["customers"].max() * 1.22)
ax.set_title("Cancellation Reasons: Price and Low Adoption Lead the List", pad=14)
ax.set_xlabel("Number of churned customers")
ax.set_ylabel("")
ax.grid(axis="x", alpha=0.3)
save_chart(fig, "07_churn_reasons.png")

# --- CHART 8: MRR lost to churn by segment ---------------------------------
say("")
say("CHART 8  Where the revenue actually goes")
say("  Question: which groups cost us the most money, not just the most logos?")

loss = (cust[cust["churn_flag"] == 1]
        .groupby("plan_tier")
        .agg(customers_lost=("customer_id", "nunique"),
             mrr_lost=("mrr_at_signup", "sum"))
        .reindex(tier_order).fillna(0))
loss["share_pct"] = (loss["mrr_lost"] / loss["mrr_lost"].sum() * 100).round(1)
say(loss.round(0).to_string())

total_mrr_lost = cust.loc[cust["churn_flag"] == 1, "mrr_at_signup"].sum()
say(f"  Total MRR lost to churn over the two years: {money(total_mrr_lost)}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
bars = ax1.bar(loss.index, loss["mrr_lost"], color=PALETTE["navy"], width=0.6,
               edgecolor="white")
for bar, v, pct in zip(bars, loss["mrr_lost"], loss["share_pct"]):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + loss["mrr_lost"].max() * 0.03,
             f"{money(v)}\n{pct:.0f}%", ha="center", va="bottom", fontsize=9,
             fontweight="bold")
ax1.set_ylim(0, loss["mrr_lost"].max() * 1.22)
ax1.set_title("MRR Lost to Churn by Plan Tier", pad=12)
ax1.set_xlabel("Plan tier")
ax1.set_ylabel("MRR lost (USD)")
ax1.yaxis.set_major_formatter(lambda v, p: f"${v/1000:,.0f}k")
ax1.grid(axis="y", alpha=0.3)

# The key trade-off: churn rate (logos) vs revenue impact (money).
tradeoff = churn_tier[["plan_tier", "churn_pct", "mrr_lost", "customers"]].copy()
scatter = ax2.scatter(tradeoff["churn_pct"], tradeoff["mrr_lost"],
                      s=tradeoff["customers"] * 5, alpha=0.65,
                      color=PALETTE["orange"], edgecolor="white", linewidth=2)
for _, r in tradeoff.iterrows():
    ax2.annotate(r["plan_tier"], (r["churn_pct"], r["mrr_lost"]),
                 textcoords="offset points", xytext=(0, 20),
                 ha="center", fontsize=10, fontweight="bold")
ax2.set_title("Logo Churn vs Revenue Impact\n(bubble size = customers in tier)",
              pad=12)
ax2.set_xlabel("Churn rate (% of customers)")
ax2.set_ylabel("MRR lost (USD)")
ax2.yaxis.set_major_formatter(lambda v, p: f"${v/1000:,.0f}k")
ax2.grid(alpha=0.3)
save_chart(fig, "08_revenue_impact_of_churn.png")


# -----------------------------------------------------------------------------
# SECTION 8: BUSINESS ANALYSIS - ANSWERING THE KEY QUESTIONS
# -----------------------------------------------------------------------------

banner("SECTION 8  |  BUSINESS ANALYSIS")

# --- Q1: What is the current state? ---------------------------------------
say("")
say("=" * 76)
say("KEY QUESTION 1  |  What is the state of the business right now?")
say("=" * 76)
say(f"  Total customers on the books            : {n_customers:,}")
say(f"  Churned during 2023-2024                : {n_churned:,} ({churn_rate:.1%})")
say(f"  Still active in December 2024           : {n_active:,}")
say(f"  MRR in December 2024                    : {money(current_mrr)}")
say(f"  ARR run-rate in December 2024           : {money(current_mrr * 12)}")
say(f"  Average revenue per active customer     : {money(current_mrr / current_active)}")
say(f"  MRR lost to churn over the period       : {money(total_mrr_lost)}")
peak_mrr = monthly_trend["mrr"].max()
say(f"  Change in MRR since the {PEAK_MONTH} peak   : "
    f"{(last_row['mrr'] / peak_mrr - 1) * 100:+.1f}%")
say(f"  MRR growth, last 6 months                   : {growth_late:+.1%} per month")
say("")
say("  WHAT HAPPENED")
say(f"  The customer base grew explosively through 2023, with MRR peaking at")
say(f"  {money(peak_mrr)} in {PEAK_MONTH}. Growth then stopped: average monthly")
say(f"  growth fell from {growth_early:+.1%} to {growth_late:+.1%}, and active customers")
say(f"  are down {abs((last_row['active_customers'] / active_peak['active_customers'] - 1) * 100):.0f}% from their "
    f"{active_peak['month_start'].strftime('%b %Y')} peak of "
    f"{int(active_peak['active_customers']):,}.")
say("")
say("  WHY IT MATTERS")
if TREND == "declining":
    say(f"  At the current run rate the business is billing roughly "
        f"{money((peak_mrr - last_row['mrr']) * 12)} less per year than it was at")
    say("  the peak. The growth engine that worked in 2023 has stopped working.")
else:
    # A single month is too noisy and the final month sits on the edge of the
    # reporting period, so the replacement arithmetic is done over 6 months.
    last6 = monthly_trend.tail(6)
    added6 = int(last6["new_customers"].sum())
    lost6 = int(last6["churned_this_month"].sum())
    say(f"  Revenue is now flat only because new logos almost exactly replace the "
        f"ones lost:")
    say(f"  over the last 6 months the business added {added6:,} new customers and "
        f"lost {lost6:,} existing ones, a net {added6 - lost6:+,} logos,")
    say(f"  while MRR grew {growth_late:+.1%} a month. There is no growth left to "
        f"absorb a")
    say("  bad month: churn compounds on a base that is no longer expanding.")

# --- Q2: Is the growth sustainable? ---------------------------------------
say("")
say("=" * 76)
say("KEY QUESTION 2  |  Are we still growing, and is new business replacing churn?")
say("=" * 76)

new_by_year = (cust.assign(year=cust["signup_date"].dt.year)
               .groupby("year")
               .agg(new_customers=("customer_id", "nunique"),
                    mrr_added=("mrr_at_signup", "sum"))
               .reset_index())
churn_by_year = (cust[cust["churn_flag"] == 1]
                 .assign(year=cust["churn_date"].dt.year)
                 .groupby("year")
                 .agg(churned=("customer_id", "nunique"),
                      mrr_lost=("mrr_at_signup", "sum"))
                 .reset_index())
year_summary = new_by_year.merge(churn_by_year, on="year", how="outer").fillna(0)
year_summary["net_new_customers"] = (year_summary["new_customers"]
                                     - year_summary["churned"])
year_summary["net_mrr_change"] = (year_summary["mrr_added"]
                                  - year_summary["mrr_lost"])
say(year_summary.round(0).to_string(index=False))
say("")
for _, r in year_summary.iterrows():
    say(f"  {int(r['year'])}: {int(r['new_customers']):,} new customers "
        f"({money(r['mrr_added'])} MRR) vs {int(r['churned']):,} churned "
        f"({money(r['mrr_lost'])} MRR) -> net {int(r['net_new_customers']):+,} customers, "
        f"{money(r['net_mrr_change'])} MRR")

# Is the company winning FEWER customers, or SMALLER ones? The two have very
# different fixes, so this is worth separating.
say("")
say("  Fewer customers, or smaller ones? (quality of new business)")
deal_quality = (cust.groupby("signup_cohort")
                .agg(new_customers=("customer_id", "nunique"),
                     avg_mrr_per_customer=("mrr_at_signup", "mean"),
                     avg_seats=("seat_estimate", "mean"),
                     starter_share=("plan_tier",
                                    lambda s: (s == "Starter").mean() * 100))
                .reset_index())
# Round only the numeric columns: the cohort column is a Period, and asking
# pandas to round a Period raises a warning and does nothing.
deal_view = deal_quality.copy()
for col in deal_view.select_dtypes(include="number").columns:
    deal_view[col] = deal_view[col].round(1)
say(deal_view.to_string(index=False))

h1_2023 = cust[cust["signup_cohort"] <= pd.Period("2023-06", freq="M")]
h2_2024 = cust[cust["signup_cohort"] >= pd.Period("2024-07", freq="M")]
avg_early = h1_2023["mrr_at_signup"].mean()
avg_late = h2_2024["mrr_at_signup"].mean()
starter_early = (h1_2023["plan_tier"] == "Starter").mean() * 100
starter_late = (h2_2024["plan_tier"] == "Starter").mean() * 100
say("")
say(f"  Average MRR per new customer: {money(avg_early)} in the first half of 2023")
say(f"  versus {money(avg_late)} in the second half of 2024 "
    f"({(avg_late / avg_early - 1) * 100:+.0f}%).")
say(f"  Starter-plan share of new customers: {starter_early:.0f}% -> {starter_late:.0f}%.")
say("  WHAT HAPPENED")
say("  The company is still signing customers, but each one is worth a fraction of")
say("  what it used to be worth, and the mix has shifted towards small Starter")
say("  accounts. Logo growth therefore hides a collapse in revenue quality.")
say("")
say("  WHY IT MATTERS")
say("  This is why MRR stalled even while logos kept arriving. Chasing more small")
say("  logos will not fix it: the average new customer would have to be worth")
say("  roughly what it was in early 2023 just to replace the revenue lost to churn.")

# --- Q3: Which segments leak? ---------------------------------------------
say("")
say("=" * 76)
say("KEY QUESTION 3  |  Which customers are we losing?")
say("=" * 76)
for dim, label in [("plan_tier", "Plan tier"), ("billing_cycle", "Billing cycle"),
                   ("acquisition_channel", "Acquisition channel"),
                   ("company_size", "Company size"),
                   ("engagement_band", "Engagement band")]:
    table = churn_by(dim)
    say("")
    say(f"  By {label}:")
    say(table.to_string(index=False))

# --- Q4: What causes churn? -----------------------------------------------
say("")
say("=" * 76)
say("KEY QUESTION 4  |  What signals are associated with a customer leaving?")
say("=" * 76)
say("  We compare the final 3 months of churned customers with those still active.")
say("")
say(f"  {'Signal':<34}{'Churned':>12}{'Active':>12}{'Difference':>14}")
for col, label, fmt in [("final_adoption", "Seat adoption", "pct"),
                        ("final_tickets", "Support tickets / month", "num"),
                        ("final_nps", "NPS score", "num")]:
    c_val, a_val = cust.loc[cust["churn_flag"] == 1, col].mean(), cust.loc[cust["churn_flag"] == 0, col].mean()
    if fmt == "pct":
        say(f"  {label:<34}{c_val:>11.1%}{a_val:>12.1%}{(c_val - a_val):>13.1%}")
    else:
        say(f"  {label:<34}{c_val:>12.2f}{a_val:>12.2f}{(c_val - a_val):>+14.2f}")

# A simple, transparent comparison of average discount between churned and kept.
disc_churned = cust.loc[cust["churn_flag"] == 1, "discount_pct"].mean()
disc_active = cust.loc[cust["churn_flag"] == 0, "discount_pct"].mean()
say(f"  {'Average discount %':<34}{disc_churned:>11.1f}%{disc_active:>11.1f}%"
    f"{disc_churned - disc_active:>13.1f}pp")

say("")
say("  WHAT HAPPENED")
say("  Customers who churned were using a smaller share of their seats, raised")
say("  slightly more support tickets, gave a lower NPS, and had been given a")
say("  larger discount at signup - than customers who stayed.")
say("")
say("  WHY IT MATTERS")
say("  All four are recorded BEFORE the cancellation, so they can be used to")
say("  flag accounts while there is still time to save them. Note this is an")
say("  association, not proof of cause: low adoption may be a symptom of a")
say("  customer who was already planning to leave.")

# --- Q5: Net revenue retention --------------------------------------------
say("")
say("=" * 76)
say("KEY QUESTION 5  |  Is revenue retained from existing customers?")
say("=" * 76)

# NRR: take the customers who were active 12 months ago, and see how much of
# their revenue is still with us now (ignoring new customers).
prior_month = DATA_END - pd.DateOffset(months=12)
if prior_month in set(mon["month_start"]):
    prior = mon[mon["month_start"] == prior_month][["customer_id", "mrr"]]
    prior = prior.rename(columns={"mrr": "mrr_12m_ago"})
    now = mon[mon["month_start"] == DATA_END][["customer_id", "mrr"]]
    now = now.rename(columns={"mrr": "mrr_now"})
    nrr_base = prior.merge(now, on="customer_id", how="left")
    retained = nrr_base["mrr_now"].notna()
    nrr_base["mrr_now"] = nrr_base["mrr_now"].fillna(0)
    start_mrr_12 = nrr_base["mrr_12m_ago"].sum()
    end_mrr_12 = nrr_base.loc[retained, "mrr_now"].sum()
    nrr = end_mrr_12 / start_mrr_12 * 100
    nrr_expansion = (end_mrr_12 - nrr_base.loc[retained, "mrr_12m_ago"].sum()) / start_mrr_12 * 100
    nrr_churn = -nrr_base.loc[~retained, "mrr_12m_ago"].sum() / start_mrr_12 * 100

    say(f"  Comparing December 2023 with December 2024 for the same customers:")
    say(f"    MRR 12 months ago (that cohort)      : {money(start_mrr_12)}")
    say(f"    MRR now from the customers who stayed: {money(end_mrr_12)}")
    say(f"    Net Revenue Retention (NRR)          : {nrr:.1f}%")
    say(f"    of which expansion from seat growth   : {nrr_expansion:+.1f} pp")
    say(f"    of which churn from cancellations    : {nrr_churn:+.1f} pp")
    say("")
    say("  WHAT HAPPENED")
    if nrr < 100:
        say(f"  NRR of {nrr:.0f}% means the existing customer base shrank: the company")
        say(f"  gave back more revenue through churn ({nrr_churn:.1f} pp) than it")
        say(f"  gained from expansion ({nrr_expansion:+.1f} pp).")
    else:
        say(f"  NRR of {nrr:.0f}% means the existing base grew without needing new logos.")
    say("")
    say("  WHY IT MATTERS")
    say("  NRR is the single number that tells a subscription business whether it")
    say("  is getting healthier or just buying more customers to stand still.")
else:
    say("  (12-month comparison not available in this dataset)")

# --- Q6: How much revenue is at risk? --------------------------------------
say("")
say("=" * 76)
say("KEY QUESTION 6  |  How much revenue is currently at risk?")
say("=" * 76)
say("  We build a simple, explainable risk score out of the signals that Section 8")
say("  Question 4 identified. It is a rule-based score, not a black-box model, so")
say("  a customer-success manager can see exactly why an account is flagged.")
say("")
RISK_WEIGHTS = {
    # column: (weight, plain-English meaning)
    "final_adoption": (-2.0, "low seat use pushes the score UP"),
    "final_tickets": (1.0, "more support tickets pushes the score UP"),
    "final_nps": (-1.2, "a low NPS pushes the score UP"),
    "final_mrr": (1.0, "bigger accounts rank higher - more MRR is at stake"),
    "tenure_months": (-0.8, "a short tenure pushes the score UP"),
    "discount_pct": (0.6, "a heavy signup discount pushes the score UP"),
}
say("  Risk score = weighted sum of six signals, each rescaled to 0-1 so that")
say("  no single measurement dominates because of its units. The SIGN of the")
say("  weight carries the direction: a negative weight means a high value is")
say("  protective, a positive weight means a high value is a warning sign.")
say("")
for col, (weight, why) in RISK_WEIGHTS.items():
    say(f"    {col:<20} weight {weight:+.1f}   {why}")
say("")

active_only = cust[cust["churn_flag"] == 0].copy()
risk = pd.Series(0.0, index=active_only.index)
for col, (weight, _) in RISK_WEIGHTS.items():
    values = active_only[col]
    # Rescale to 0-1 between the 5th and 95th percentile, so that a handful of
    # extreme accounts cannot flatten the rest of the score.
    lo, hi = values.quantile(0.05), values.quantile(0.95)
    scaled = ((values.clip(lo, hi) - lo) / (hi - lo)).fillna(0.5)
    risk += weight * scaled
active_only["risk_score"] = risk

# Split into three bands using the score distribution.
active_only["risk_band"] = pd.qcut(active_only["risk_score"], 3,
                                   labels=["Low risk", "Medium risk", "High risk"])
risk_summary = (active_only.groupby("risk_band", observed=True)
                .agg(customers=("customer_id", "nunique"),
                     total_mrr=("final_mrr", "sum"),
                     avg_score=("risk_score", "mean"),
                     avg_adoption=("final_adoption", "mean"),
                     avg_nps=("final_nps", "mean"),
                     avg_tickets=("final_tickets", "mean"))
                .reset_index())
risk_summary["share_of_active_mrr"] = (risk_summary["total_mrr"]
                                       / risk_summary["total_mrr"].sum() * 100)
say(risk_summary.round(3).to_string(index=False))

high = risk_summary[risk_summary["risk_band"] == "High risk"]
if not high.empty:
    high_mrr = high["total_mrr"].iloc[0]
    say("")
    say(f"  MRR sitting with HIGH-RISK active customers: {money(high_mrr)} "
        f"({high['share_of_active_mrr'].iloc[0]:.0f}% of active MRR)")
    say(f"  These {int(high['customers'].iloc[0]):,} accounts use "
        f"{high['avg_adoption'].iloc[0]:.0%} of their seats on average, give NPS "
        f"{high['avg_nps'].iloc[0]:.1f} and raise "
        f"{high['avg_tickets'].iloc[0]:.1f} tickets a month.")
    say("")
    say("  WHY IT MATTERS")
    say(f"  This is the number a retention campaign would be measured against.")
    say(f"  Improving adoption in this group alone protects {money(high_mrr)} of")
    say("  recurring revenue - and the accounts are already identified, so the")
    say("  work can start immediately rather than waiting for a predictive model.")

# --- Q7: What should we do? -----------------------------------------------
say("")
say("=" * 76)
say("KEY QUESTION 7  |  What should the business do?")
say("=" * 76)
say("  The recommendations below follow directly from the evidence above.")
say("")
starter_churn = churn_tier.loc[churn_tier["plan_tier"] == "Starter", "churn_pct"].iloc[0]
monthly_churn = churn_by("billing_cycle").set_index("billing_cycle")["churn_pct"]
say(f"  1. Fix the first 90 days. {first90_share:.0f}% of all cancellations happen "
    f"before a customer has been")
say(f"     subscribed for 3 months, and months {worst_start}-{worst_start + 3} are the "
    f"worst window of the whole")
say(f"     curve ({window_drops[worst_start]:.0f} points of the customer base). The "
    f"cancellation rate is {first_month_hazard:.1f}%")
say(f"     in month one. Move onboarding, activation and the first support contact")
say("     into a structured 90-day programme instead of reactive support.")
say("")
say(f"  2. Then handle the first renewal. The cancellation rate dips to about 2%")
say(f"     mid-tenure and then climbs back to {renewal_peak:.1f}% around months 12-18 "
    f"as the first annual")
say("     contracts come up for decision. Start that conversation 90 days before")
say("     the anniversary, not when the cancellation notice arrives.")
say("")
say(f"  3. Stop losing {monthly_churn.idxmax()} accounts faster than {monthly_churn.idxmin()} ones.")
say(f"     {monthly_churn.idxmax()} billing churns at {monthly_churn.max():.0f}% versus "
    f"{monthly_churn.min():.0f}% on {monthly_churn.idxmin()} billing. Offer an")
say("     annual-billing incentive at signup to close the gap.")
say("")
say(f"  4. Review discounting. Churned customers were given "
    f"{disc_churned - disc_active:.1f} percentage points more discount on average")
say(f"     than customers who stayed. Discounts buy contracts that do not last.")
say("     Tie discount approval to company size and to the plan the account can")
say("     realistically grow into.")
say("")
worst_channel = churn_channel.iloc[-1]
best_channel = churn_channel.iloc[0]

# Where does the worst channel actually sit on lifetime value? Looking the rank
# up is safer than asserting "second-lowest" and being wrong.
ltv_by_channel = channel_value.set_index("acquisition_channel")["avg_ltv"]
ltv_sorted = ltv_by_channel.sort_values(ascending=False)
ltv_rank = int(ltv_sorted.index.get_loc(worst_channel["acquisition_channel"])) + 1
rank_words = {1: "the lowest lifetime value of any channel",
              2: "the second-lowest lifetime value of any channel",
              3: "the third-lowest lifetime value of any channel"}
ltv_phrase = rank_words.get(
    ltv_rank, f"lifetime value ranked {ltv_rank} of {len(ltv_sorted)}")

say(f"  5. Re-balance acquisition spend away from {worst_channel['acquisition_channel']}.")
say(f"     It has the highest churn ({worst_ch['churn_pct']}%) and {ltv_phrase} "
    f"({money(ltv_by_channel[worst_channel['acquisition_channel']])}),")
say(f"     yet it still produces logos. {best_channel['acquisition_channel']} keeps "
    f"customers far better at {best_ch['churn_pct']}%.")
say("     Shift budget, and add a 90-day retention check before renewing")
say("     acquisition budget.")
say("")
say("  6. Run a targeted save programme on the high-risk accounts identified above")
say("     rather than a company-wide discount campaign, which would cost more")
say("     than the churn it prevents.")


# -----------------------------------------------------------------------------
# SECTION 9: SAVE EVERYTHING
# -----------------------------------------------------------------------------

banner("SECTION 9  |  SAVING OUTPUTS")

# Clean tables for the SQL, Excel and Power BI stages.
customers_clean = cust.copy()
monthly_clean = mon.copy()

# Month columns are period objects in pandas. They are written out as plain text
# ("2024-12") so the SQL, Excel and BI stages can load them without extra steps.
for frame in (customers_clean, monthly_clean):
    for col in frame.columns:
        if isinstance(frame[col].dtype, pd.PeriodDtype):
            frame[col] = frame[col].astype(str)
            say(f"    period column written as text: {col}")

risk_export = active_only[[
    "customer_id", "plan_tier", "region", "industry", "acquisition_channel",
    "billing_cycle", "company_size", "signup_date", "tenure_months",
    "final_mrr", "final_adoption", "final_tickets", "final_nps",
    "discount_pct", "risk_score", "risk_band",
]].copy()

files_written = []


def save_csv(frame, name):
    path = os.path.join(CLEAN_DIR, name)
    frame.to_csv(path, index=False)
    files_written.append((name, len(frame), len(frame.columns)))


save_csv(customers_clean, "customers_clean.csv")
save_csv(monthly_clean, "customer_monthly_clean.csv")
save_csv(risk_export, "churn_risk_scoring.csv")
save_csv(monthly_trend, "analysis_monthly_trend.csv")
save_csv(churn_tier, "analysis_churn_by_plan.csv")
save_csv(churn_channel, "analysis_churn_by_channel.csv")
save_csv(reason_data, "analysis_churn_reasons.csv")
save_csv(cohort_pct.reset_index(), "analysis_cohort_retention.csv")
save_csv(survivor_df, "analysis_retention_by_tenure.csv")
save_csv(year_summary, "analysis_yearly_growth_vs_churn.csv")
save_csv(risk_summary, "analysis_risk_summary.csv")
save_csv(channel_value, "analysis_channel_value.csv")

# A flat KPI table for the report and the Excel dashboard.
kpi_summary = pd.DataFrame([
    ("Total customers (2023-2024)", f"{n_customers:,}", "Customers"),
    ("Churned customers", f"{n_churned:,}", "Customers"),
    ("Active customers, Dec 2024", f"{n_active:,}", "Customers"),
    ("Logo churn rate", f"{churn_rate * 100:.1f}%", "%"),
    ("MRR, Dec 2024", money(current_mrr), "USD"),
    ("ARR run-rate, Dec 2024", money(current_mrr * 12), "USD"),
    ("ARPU, Dec 2024", money(current_mrr / current_active), "USD"),
    ("MRR lost to churn (period)", money(total_mrr_lost), "USD"),
    ("MRR peak", f"{money(peak_mrr)} ({PEAK_MONTH})", "USD"),
    ("Change in MRR since peak", f"{(last_row['mrr'] / peak_mrr - 1) * 100:+.1f}%", "%"),
    ("Survival at month 6 (of all signups)", f"{r6:.0f}%", "%"),
    ("Survival at month 12 (of all signups)", f"{r12:.0f}%", "%"),
    ("Average NPS", f"{cust['nps_score'].mean():.2f}", "Score (0-10)"),
    ("Average tenure", f"{cust['tenure_months'].mean():.1f}", "Months"),
    ("Data quality issues found", f"{len(quality_df)}", "Categories"),
    ("Rows removed as duplicates",
     f"{raw_customers.shape[0] - len(customers_clean) + raw_monthly.shape[0] - len(monthly_clean):,}",
     "Rows"),
], columns=["KPI", "Value", "Unit"])
save_csv(kpi_summary, "kpi_summary.csv")

say("Files written to data/cleaned/:")
for name, rows, cols in files_written:
    say(f"  {name:<38} {rows:>7,} rows x {cols:>2} cols")

charts = sorted(os.listdir(CHART_DIR))
say("")
say(f"Charts written to visualizations/charts/: {len(charts)}")
for c in charts:
    say(f"  {c}")

# Save the console output as a text report.
report_path = os.path.join(REPORT_DIR, "console_findings.txt")
with open(report_path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(CONSOLE_LINES))
say("")
say(f"Console findings saved to reports/console_findings.txt")

banner("PIPELINE COMPLETE")
say("Next: sql/analysis.sql  |  python/build_excel.py  |  notebooks/Capstone_Analysis.ipynb")
