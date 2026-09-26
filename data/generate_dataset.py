"""
================================================================================
SYNTHETIC DATASET GENERATOR  -  B2B SaaS Customer Churn & Revenue Retention
================================================================================

PURPOSE
-------
This script creates a *synthetic* (simulated) dataset for the capstone project
"SaaS Customer Churn & Revenue Retention Analysis".

The data is NOT taken from any real company. It is generated here, from a fixed
random seed, using a documented behavioural model. Because the seed is fixed,
running this script again always produces the exact same files, so every number
in the report can be reproduced by anyone who clones the repository.

WHY SYNTHETIC?
--------------
A portfolio project needs a dataset that is (a) legally shareable and
(b) rich enough to support real analysis. Simulating one lets us control the
data-quality problems we want to practise cleaning, without using anyone's
private data. Every conclusion in the report is drawn from these simulated
records, and the report states this openly.

WHAT IS SIMULATED
-----------------
Each customer is followed month by month from the month they signed up until
they either churn or reach the end of the reporting period. For every active
month we simulate:
    * how many of their seats were actually used (adoption)
    * how many support tickets they raised
    * whether their seat count grew, shrank or stayed the same
    * whether they cancelled

CANCELLATION MODEL
------------------
The probability of cancelling in a month is calculated with a logistic model
("log-odds") where a bigger value means a higher chance of leaving. The drivers
are deliberately realistic:

    low adoption                      -> more likely to churn
    many support tickets              -> more likely to churn
    low NPS score                     -> more likely to churn
    heavy discount                    -> more likely to churn
    Starter plan / very small company -> more likely to churn
    Annual billing                    -> less likely to churn
    longer tenure                     -> less likely to churn

We do NOT hard-code any conclusion. The analysis scripts later rediscover these
patterns from the generated files, exactly as they would with real data.

OUTPUT FILES (written to data/raw/)
-----------------------------------
    customers_raw.csv            1 row per customer   (master table)
    customer_monthly_raw.csv     1 row per customer per active month (fact table)

Both raw files contain realistic data-quality problems on purpose:
duplicate rows, inconsistent text casing, mixed date formats, currency symbols
inside numbers, missing values and a few impossible values. See
INJECTED DATA QUALITY ISSUES below.

HOW TO RUN
----------
    python data/generate_dataset.py
================================================================================
"""

import os
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 1. SETTINGS
# -----------------------------------------------------------------------------

# A fixed seed means the same data every time the script runs.
RANDOM_SEED = 20241231

# Number of customers to simulate.
N_CUSTOMERS = 1250

# The reporting period. Customers are simulated up to the end of 2024.
SIGNUP_START = "2023-01-01"
REPORT_END = "2024-12-31"

def signup_rate_shape(f):
    """Target number of new customers per month at position `f` (0 = first month,
    1 = last month) of the reporting period.

    The shape is a business that ramps up hard in 2023, peaks in late summer
    2023, and then loses momentum until it is adding very few customers by the
    end of 2024 - new business falls below the number of customers churning,
    which is what turns MRR growth into MRR decline. Written as one function so
    the growth story is visible in a single place and easy to tune.
    """
    return 35.0 + 70.0 * np.sin(np.pi * f ** 0.62) - 25.0 * f ** 1.5

# Where to save the files (../data/raw relative to this script).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw")

# Reference values used across the project.
PLAN_FEE = {           # platform fee per month, by plan tier
    "Starter": 49,
    "Growth": 149,
    "Business": 399,
    "Enterprise": 899,
}
SEAT_PRICE = {         # list price per seat per month, by plan tier
    "Starter": 12,
    "Growth": 22,
    "Business": 32,
    "Enterprise": 45,
}

rng = np.random.default_rng(RANDOM_SEED)


# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def sigmoid(x):
    """Turn a log-odds number into a probability between 0 and 1."""
    return 1.0 / (1.0 + np.exp(-x))


def add_months(date, n):
    """Return the date n months after `date` (same day of month where possible)."""
    month_index = date.month - 1 + n
    year = date.year + month_index // 12
    month = month_index % 12 + 1
    # Clamp the day so that 31 Jan + 1 month becomes 28/29 Feb instead of crashing.
    day = min(date.day, 28)
    return pd.Timestamp(year=year, month=month, day=day)


def month_label(date):
    """Turn a date into the first day of that month, e.g. 2023-04-01."""
    return pd.Timestamp(year=date.year, month=date.month, day=1)


def weighted_choice(options, weights, size=None):
    """Pick values from `options` using `weights` (same trick as numpy.choice,
    but written out longhand so it is easy to follow)."""
    options = np.array(options)
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    if size is None:
        return options[rng.choice(len(options), p=weights)]
    return rng.choice(options, size=size, p=weights)


# -----------------------------------------------------------------------------
# 3. REFERENCE LISTS (the "dimensions" of the business)
# -----------------------------------------------------------------------------

PLAN_TIERS = ["Starter", "Growth", "Business", "Enterprise"]
PLAN_TIER_WEIGHTS = [0.34, 0.31, 0.24, 0.11]

INDUSTRIES = [
    "Technology", "Healthcare", "Financial Services", "Retail",
    "Manufacturing", "Education", "Logistics", "Media",
]
INDUSTRY_WEIGHTS = [0.22, 0.15, 0.14, 0.15, 0.12, 0.09, 0.08, 0.05]

# Country -> region. Used to derive the region automatically.
COUNTRY_REGION = {
    "United States": "North America",
    "Canada": "North America",
    "United Kingdom": "Europe",
    "Germany": "Europe",
    "France": "Europe",
    "Spain": "Europe",
    "India": "Asia Pacific",
    "Singapore": "Asia Pacific",
    "Australia": "Asia Pacific",
    "Japan": "Asia Pacific",
    "Brazil": "Latin America",
    "Mexico": "Latin America",
}
COUNTRY_WEIGHTS = [
    0.30, 0.06, 0.11, 0.09, 0.07, 0.04,
    0.10, 0.04, 0.05, 0.04,
    0.06, 0.04,
]

COMPANY_SIZES = ["1-50", "51-200", "201-1000", "1000+"]
COMPANY_SIZE_WEIGHTS = [0.42, 0.31, 0.18, 0.09]

ACQUISITION_CHANNELS = [
    "Organic", "Paid Search", "Referral",
    "Partner", "Outbound Sales", "Events",
]
CHANNEL_WEIGHTS = [0.24, 0.24, 0.18, 0.14, 0.12, 0.08]

CHURN_REASONS = [
    "Too expensive",
    "Missing features",
    "Poor support experience",
    "Low adoption / unused seats",
    "Switched to competitor",
    "Company downsizing",
    "Business priorities changed",
]
CHURN_REASON_WEIGHTS = [0.26, 0.17, 0.13, 0.18, 0.14, 0.07, 0.05]


# -----------------------------------------------------------------------------
# 4. BUILD THE CUSTOMER MASTER TABLE
# -----------------------------------------------------------------------------

def build_customers():
    """Create one row per customer with their starting profile."""
    start = pd.Timestamp(SIGNUP_START)
    end = pd.Timestamp(REPORT_END)

    # Signup dates cover the WHOLE reporting period, and the number of new
    # customers per month follows signup_rate_shape(): a business that ramps up
    # through 2023 and then stalls through 2024. That stall, combined with
    # churn, is what produces the MRR peak-and-decline shape the analysis stage
    # is designed to investigate.
    #
    # Implementation: turn the shape into an exact count per month, then place
    # each customer in the month where the running total passes their index.
    n_months = (end.year - start.year) * 12 + end.month - start.month + 1
    midpoints = (np.arange(n_months) + 0.5) / n_months
    rate = np.clip(signup_rate_shape(midpoints), 1.0, None)
    per_month = np.round(rate / rate.sum() * N_CUSTOMERS).astype(int)

    # Fix any rounding drift so the month counts still add up to N_CUSTOMERS.
    drift = N_CUSTOMERS - per_month.sum()
    per_month[np.argmax(per_month)] += drift

    month_starts = [start + pd.offsets.MonthBegin(i) for i in range(n_months)]
    signup_dates = []
    for month_start_i, count in zip(month_starts, per_month):
        for _ in range(int(count)):
            day = int(rng.integers(0, 28))       # day 0-27, inside the month
            signup_dates.append(month_start_i + pd.Timedelta(days=day))
    signup_dates = sorted(signup_dates)

    rows = []
    for i in range(N_CUSTOMERS):
        # How far through the reporting period this customer signed up (0 at the
        # start, 1 at the end). Signup dates are sorted, so i tracks time.
        progress = i / max(N_CUSTOMERS - 1, 1)

        # MIX SHIFT. A business that is losing momentum does not stop signing
        # customers - it starts signing SMALLER, CHEAPER ones. Later in the
        # period the mix tilts towards Starter plans and small companies. This
        # is the hidden reason revenue growth stalls even while logo growth
        # continues, and it is the single most important thing in the dataset.
        tilt = progress ** 1.5
        plan_weights = PLAN_TIER_WEIGHTS * np.array(
            [1.0 + 0.60 * tilt, 1.0 - 0.05 * tilt,
             1.0 - 0.30 * tilt, 1.0 - 0.65 * tilt])
        size_weights = COMPANY_SIZE_WEIGHTS * np.array(
            [1.0 + 0.18 * tilt, 1.0 + 0.05 * tilt,
             1.0 - 0.30 * tilt, 1.0 - 0.60 * tilt])

        plan = weighted_choice(PLAN_TIERS, plan_weights)
        industry = weighted_choice(INDUSTRIES, INDUSTRY_WEIGHTS)
        country = weighted_choice(list(COUNTRY_REGION), COUNTRY_WEIGHTS)
        region = COUNTRY_REGION[country]
        company_size = weighted_choice(COMPANY_SIZES, size_weights)
        channel = weighted_choice(ACQUISITION_CHANNELS, CHANNEL_WEIGHTS)

        # Annual billing is more common for bigger companies.
        annual_prob = {"1-50": 0.18, "51-200": 0.34,
                       "201-1000": 0.55, "1000+": 0.72}[company_size]
        billing_cycle = "Annual" if rng.random() < annual_prob else "Monthly"

        # --- Seat count -----------------------------------------------------
        # Bigger companies and higher plans get more seats.
        size_mu = {"1-50": 1.6, "51-200": 2.6,
                   "201-1000": 3.6, "1000+": 4.7}[company_size]
        tier_boost = {"Starter": -0.5, "Growth": 0.0,
                      "Business": 0.6, "Enterprise": 1.2}[plan]
        seats = int(np.clip(
            np.exp(rng.normal(size_mu + tier_boost, 0.55)), 1, 900))
        seats = max(seats, 1)

        # --- Discount -------------------------------------------------------
        # Partners and outbound give bigger discounts; organic gives none.
        base_discount = {
            "Organic": 0.0, "Paid Search": 8.0, "Referral": 5.0,
            "Partner": 15.0, "Outbound Sales": 12.0, "Events": 10.0,
        }[channel]
        size_extra = {"1-50": 0.0, "51-200": 2.0,
                      "201-1000": 5.0, "1000+": 7.0}[company_size]
        discount_pct = float(np.clip(
            rng.normal(base_discount + size_extra, 4.0), 0, 35))
        discount_pct = round(discount_pct, 1)

        # --- Starting MRR ---------------------------------------------------
        seat_revenue = seats * SEAT_PRICE[plan] * (1 - discount_pct / 100)
        mrr = seat_revenue + PLAN_FEE[plan]
        mrr = round(float(mrr), 2)

        # --- Latent qualities that drive later behaviour ---------------------
        adoption_base = float(rng.beta(2.0, 3.0))          # 0-1, mean ~0.40
        adoption_drift = float(rng.normal(-0.006, 0.022))  # slow decline
        support_propensity = float(rng.normal(1.0, 0.45))
        nps_base = float(rng.normal(7.4, 1.9))
        seat_growth_rate = float(rng.normal(0.006, 0.018))

        rows.append({
            "customer_id": f"CUST-{10001 + i}",
            "signup_date": signup_dates[i],
            "plan_tier": plan,
            "industry": industry,
            "country": country,
            "region": region,
            "company_size": company_size,
            "seats": seats,
            "billing_cycle": billing_cycle,
            "acquisition_channel": channel,
            "discount_pct": discount_pct,
            "mrr_at_signup": mrr,
            # hidden simulation inputs (dropped before saving)
            "_adoption_base": adoption_base,
            "_adoption_drift": adoption_drift,
            "_support_propensity": support_propensity,
            "_nps_base": nps_base,
            "_seat_growth_rate": seat_growth_rate,
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 5. SIMULATE EACH CUSTOMER MONTH BY MONTH
# -----------------------------------------------------------------------------

def simulate(customers):
    """
    Walk every customer forward one month at a time.

    For each active month we record: revenue, seats, adoption, support tickets
    and whether they are still active. The first month they are no longer
    active, we record the churn date and churn reason.
    """
    report_end = pd.Timestamp(REPORT_END)

    monthly_rows = []
    final_customers = []

    for cust in customers.to_dict("records"):
        plan = cust["plan_tier"]
        discount = cust["discount_pct"] / 100.0
        start = cust["signup_date"]
        current_seats = float(cust["seats"])

        churned = False
        churn_date = None
        churn_reason = None
        tenure_months = 0
        last_login = start

        month_cursor = month_label(start)

        while month_cursor <= report_end and not churned:
            # ---- tenure in months ------------------------------------------
            t = (month_cursor.year - start.year) * 12 + (month_cursor.month - start.month)

            # ---- adoption: share of seats actually used --------------------
            adoption = cust["_adoption_base"] + cust["_adoption_drift"] * t
            adoption = adoption + rng.normal(0, 0.07)
            adoption = float(np.clip(adoption, 0.03, 1.0))
            active_users = int(round(current_seats * adoption))

            # ---- support tickets -------------------------------------------
            ticket_mean = cust["_support_propensity"] * (1.6 - adoption)
            tickets = int(max(0, rng.poisson(max(ticket_mean, 0.05))))

            # ---- NPS score (0-10) ------------------------------------------
            nps = cust["_nps_base"] + 2.4 * (adoption - 0.40) - 0.22 * tickets
            nps = int(np.clip(round(nps), 0, 10))

            # ---- seat growth / contraction ---------------------------------
            change = rng.normal(cust["_seat_growth_rate"], 0.05)
            current_seats = max(1.0, current_seats * (1 + change))
            seats_now = int(round(current_seats))

            # ---- revenue this month ----------------------------------------
            mrr = seats_now * SEAT_PRICE[plan] * (1 - discount) + PLAN_FEE[plan]
            mrr = round(float(mrr), 2)

            # ---- churn probability for this month --------------------------
            # The intercept is the baseline monthly risk for an average,
            # healthy account. The terms below then push that risk up or down.
            # Every behavioural term is CENTRED on its average, so the
            # intercept alone sets the risk of a "typical" customer and the
            # drivers move it up or down from there. This keeps the model
            # understandable and stops the weights from fighting each other.
            log_odds = -3.55
            log_odds += -1.30 * (adoption - 0.40)          # engagement protects
            log_odds += 0.030 * min(tickets, 12)          # support issues hurt
            log_odds += -0.050 * (nps - 7)                # promoters protect
            log_odds += 1.60 * (discount / 0.35)          # discounting is risky
            log_odds += {"Starter": 0.45, "Growth": 0.10,
                         "Business": -0.10, "Enterprise": -0.30}[plan]
            log_odds += {"1-50": 0.36, "51-200": 0.10,
                         "201-1000": -0.12, "1000+": -0.26}[cust["company_size"]]
            log_odds += -0.55 if cust["billing_cycle"] == "Annual" else 0.0
            log_odds += {"Organic": -0.16, "Paid Search": 0.26, "Referral": -0.06,
                         "Partner": -0.10, "Outbound Sales": 0.16,
                         "Events": 0.12}[cust["acquisition_channel"]]
            # Tenure protection, SMOOTH and front-loaded. New accounts are the
            # most fragile, so protection has to build up quickly and then level
            # off - a straight log term protects too slowly and leaves a cliff
            # at the point where the first renewal hits.
            log_odds += -0.90 * (1.0 - np.exp(-t / 3.5))
            # Renewal pressure. Accounts that reach 12+ months face a genuine
            # renewal decision, and this is the single biggest driver of the
            # 2024 MRR decline: the 2023 cohort came up for renewal and a large
            # share of it did not renew. It ramps in over months 12-18.
            log_odds += 0.50 * min(max(t - 11, 0) / 6.0, 1.0)

            p_churn = float(sigmoid(log_odds))

            monthly_rows.append({
                "customer_id": cust["customer_id"],
                "month_start": month_cursor,
                "month_number": month_cursor.month,
                "year": month_cursor.year,
                "tenure_month": t,
                "mrr": mrr,
                "seats": seats_now,
                "active_users": active_users,
                "adoption_rate": round(adoption, 4),
                "support_tickets": tickets,
                "nps_score": nps,
                "is_active": 1,
            })

            last_login = month_cursor + pd.Timedelta(
                days=int(rng.integers(0, 27)))
            tenure_months = t + 1

            # ---- decide whether they leave at the end of this month --------
            month_cursor = add_months(month_cursor, 1)
            # Only count a cancellation if it takes effect INSIDE the reporting
            # period. A customer who cancels on 1 Jan 2025 was still an active
            # customer on 31 Dec 2024, so they must stay 'Active' in this file.
            if rng.random() < p_churn and month_cursor <= report_end:
                churned = True
                churn_date = month_cursor
                churn_reason = weighted_choice(CHURN_REASONS, CHURN_REASON_WEIGHTS)

        # ---- write the finished customer record ---------------------------
        rec = {k: v for k, v in cust.items() if not k.startswith("_")}
        rec["status"] = "Churned" if churned else "Active"
        rec["churn_date"] = churn_date
        rec["churn_reason"] = churn_reason
        rec["tenure_months"] = tenure_months
        rec["nps_score"] = nps
        rec["last_login_date"] = last_login
        rec["seats_current"] = int(round(current_seats))
        final_customers.append(rec)

    return pd.DataFrame(final_customers), pd.DataFrame(monthly_rows)


# -----------------------------------------------------------------------------
# 6. INJECT REALISTIC DATA-QUALITY PROBLEMS
# -----------------------------------------------------------------------------
# A real exported file is never perfect. We deliberately damage the clean data
# so that the cleaning stage of the project has genuine work to do.
# Each issue below matches something a data analyst meets in practice.

def inject_quality_problems(customers, monthly):
    """Return 'raw' copies of the tables containing realistic defects."""
    cust = customers.copy()
    mon = monthly.copy()

    # Some columns deliberately end up holding mixed types (numbers AND text),
    # so we widen them to `object` first. Otherwise pandas refuses to store a
    # text value in a numeric column - which is exactly the real-world problem
    # we want the cleaning stage to discover.
    for col in ["mrr_at_signup", "seats", "nps_score", "churn_date",
                "last_login_date", "churn_reason", "billing_cycle",
                "plan_tier", "region"]:
        cust[col] = cust[col].astype(object)
    mon["adoption_rate"] = mon["adoption_rate"].astype(object)

    # -- Issue 1: a few MRR rows stored as TEXT with a currency symbol --------
    money_rows = cust.sample(n=14, random_state=1).index
    cust.loc[money_rows, "mrr_at_signup"] = (
        "$" + cust.loc[money_rows, "mrr_at_signup"].map(lambda v: f"{v:,.2f}")
    )

    # -- Issue 2: a few seat counts stored as the text 'n/a' -----------------
    seat_rows = cust.sample(n=7, random_state=2).index
    cust.loc[seat_rows, "seats"] = "n/a"

    # -- Issue 3: inconsistent capitalisation in billing_cycle ---------------
    cyc_rows = cust.sample(n=22, random_state=3).index
    cust.loc[cyc_rows, "billing_cycle"] = "monthly"          # lower case

    # -- Issue 4: stray leading/trailing spaces in plan_tier -----------------
    plan_rows = cust.sample(n=16, random_state=4).index
    cust.loc[plan_rows, "plan_tier"] = (
        " " + cust.loc[plan_rows, "plan_tier"].str.lower() + "  "
    )                                                       # e.g. "  growth  "

    # -- Issue 5: region written three different ways ------------------------
    reg_na = cust.sample(n=9, random_state=5).index
    cust.loc[reg_na, "region"] = "north america"
    reg_emea = cust.sample(n=7, random_state=6).index
    cust.loc[reg_emea, "region"] = "EMEA "                # trailing space

    # -- Issue 6: churn_date written as DD/MM/YYYY instead of YYYY-MM-DD ----
    churned = cust[cust["churn_date"].notna()]
    date_rows = churned.sample(n=26, random_state=7).index
    cust.loc[date_rows, "churn_date"] = (
        pd.to_datetime(cust.loc[date_rows, "churn_date"]).dt.strftime("%d/%m/%Y")
    )

    # -- Issue 7: missing values --------------------------------------------
    # NPS not answered by every customer.
    nps_missing = cust.sample(n=185, random_state=8).index
    cust.loc[nps_missing, "nps_score"] = np.nan
    # last_login_date not tracked for a minority of accounts.
    login_missing = cust.sample(n=96, random_state=9).index
    cust.loc[login_missing, "last_login_date"] = np.nan
    # A handful of churned customers have no recorded reason (forgot to fill in).
    reason_missing = cust[(cust["status"] == "Churned")].sample(
        n=41, random_state=10).index
    cust.loc[reason_missing, "churn_reason"] = np.nan

    # -- Issue 8: impossible / erroneous values -----------------------------
    # Three accounts carry a negative MRR (an unapplied credit note).
    neg_rows = cust.sample(n=3, random_state=11).index
    cust.loc[neg_rows, "mrr_at_signup"] = -abs(
        cust.loc[neg_rows, "mrr_at_signup"].astype(float))
    # Two accounts have an implausibly large seat count (double data entry).
    huge_rows = cust.sample(n=2, random_state=12).index
    cust.loc[huge_rows, "seats"] = "50000"

    # -- Issue 9: exact duplicate rows (an ETL double-load) ------------------
    dupes = cust.sample(n=23, random_state=13)
    cust = pd.concat([cust, dupes], ignore_index=True)

    # Same problem in the monthly fact table.
    mon_dupes = mon.sample(n=31, random_state=14)
    mon = pd.concat([mon, mon_dupes], ignore_index=True)

    # -- Issue 10: some monthly adoption_rate stored as a percentage string ---
    pct_rows = mon.sample(n=140, random_state=15).index
    mon.loc[pct_rows, "adoption_rate"] = (
        (mon.loc[pct_rows, "adoption_rate"].astype(float) * 100)
        .map(lambda v: f"{v:.1f}")
    )

    return cust, mon


# -----------------------------------------------------------------------------
# 7. SAVE THE FILES
# -----------------------------------------------------------------------------

def save_files(customers, monthly):
    os.makedirs(RAW_DIR, exist_ok=True)

    cust_path = os.path.join(RAW_DIR, "customers_raw.csv")
    mon_path = os.path.join(RAW_DIR, "customer_monthly_raw.csv")

    customers.to_csv(cust_path, index=False)
    monthly.to_csv(mon_path, index=False)

    print("=" * 74)
    print("SYNTHETIC DATASET CREATED")
    print("=" * 74)
    print(f"  customers_raw.csv          {len(customers):>6,} rows x "
          f"{customers.shape[1]} cols")
    print(f"  customer_monthly_raw.csv   {len(monthly):>6,} rows x "
          f"{monthly.shape[1]} cols")
    print(f"  random seed                {RANDOM_SEED}")
    print(f"  saved to                   {RAW_DIR}")
    print("=" * 74)


# -----------------------------------------------------------------------------
# 8. MAIN
# -----------------------------------------------------------------------------

def main():
    print("Building customer profiles...")
    customers = build_customers()

    print("Simulating monthly behaviour (this loops 1,250 customers)...")
    customers_done, monthly = simulate(customers)

    print("Injecting realistic data-quality problems...")
    raw_customers, raw_monthly = inject_quality_problems(customers_done, monthly)

    save_files(raw_customers, raw_monthly)

    # A short console summary so the generator itself is easy to verify.
    churn_rate = (customers_done["status"] == "Churned").mean()
    print(f"  customers simulated        {len(customers_done):,}")
    print(f"  churned                   "
          f"{(customers_done['status'] == 'Churned').sum():,}")
    print(f"  overall churn rate        {churn_rate:.1%}")
    print(f"  total active MRR (Dec 24) "
          f"{monthly[monthly['month_start'] == pd.Timestamp('2024-12-01')]['mrr'].sum():,.0f}")

    # Churn timing check. Real B2B SaaS loses customers earliest in the
    # relationship, so if this simulation does not, the data is not realistic
    # enough to support an onboarding story.
    churned = customers_done[customers_done["status"] == "Churned"]
    if len(churned) > 0:
        within_3 = (churned["tenure_months"] <= 3).mean()
        within_6 = (churned["tenure_months"] <= 6).mean()
        print(f"  churned within 3 months   {within_3:.1%} of all churned customers")
        print(f"  churned within 6 months   {within_6:.1%} of all churned customers")
        print(f"  median tenure before churn {churned['tenure_months'].median():.0f} months")


if __name__ == "__main__":
    main()
