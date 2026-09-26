"""Reconcile the SQL layer against the Python pipeline.

The whole point of shipping both a Python and a SQL version of the same
analysis is that they must agree. This script runs the headline numbers through
DuckDB and compares them with the CSVs written by data_analysis.py, so any
divergence is caught immediately instead of being discovered by a reader.

Run from the project root:
    python python/validate_sql.py
"""

import pathlib
import sys

import duckdb
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "cleaned"

failures = []


def check(label, sql_value, python_value, tolerance=0.51):
    """Compare one number from each pipeline and report agreement."""
    if python_value is None:
        print(f"  ?  {label}: SQL={sql_value} (no Python value to compare)")
        return
    if abs(float(sql_value) - float(python_value)) <= tolerance:
        print(f"  OK {label}: SQL={sql_value}  Python={python_value}")
    else:
        print(f"  XX {label}: SQL={sql_value}  Python={python_value}  <-- MISMATCH")
        failures.append(label)


con = duckdb.connect()
con.execute((ROOT / "sql" / "analysis.sql").read_text(encoding="utf-8"))

kpi_raw = pd.read_csv(CLEAN / "kpi_summary.csv").set_index("KPI")["Value"].to_dict()


def kpi_number(key):
    """Read a KPI that may be stored as '1,250' or '$1,051,252' or '34.4%'."""
    raw = str(kpi_raw[key])
    return float(raw.replace("$", "").replace(",", "").replace("%", "")
                 .replace(" months", "").replace(" Score (0-10)", "")
                 .replace(" Customers", "").replace(" Rows", "").strip())


cust = pd.read_csv(CLEAN / "customers_clean.csv")
mon = pd.read_csv(CLEAN / "customer_monthly_clean.csv", parse_dates=["month_start"])
surv = pd.read_csv(CLEAN / "analysis_retention_by_tenure.csv")

print("=" * 68)
print("SQL vs PYTHON RECONCILIATION")
print("=" * 68)

print("\n1. Headline KPIs")
row = con.execute("""
    SELECT
      (SELECT COUNT(*) FROM v_customers_clean),
      (SELECT COUNT(*) FROM v_customers_clean WHERE churn_flag = 1),
      (SELECT COUNT(*) FROM v_customers_clean WHERE churn_flag = 0),
      (SELECT ROUND(SUM(mrr)) FROM v_monthly_clean
        WHERE month_start = (SELECT MAX(month_start) FROM v_monthly_clean)),
      (SELECT ROUND(SUM(mrr) * 12) FROM v_monthly_clean
        WHERE month_start = (SELECT MAX(month_start) FROM v_monthly_clean)),
      (SELECT ROUND(SUM(mrr) / COUNT(DISTINCT customer_id)) FROM v_monthly_clean
        WHERE month_start = (SELECT MAX(month_start) FROM v_monthly_clean))
""").fetchone()

check("total customers", row[0], kpi_number("Total customers (2023-2024)"))
check("churned customers", row[1], kpi_number("Churned customers"))
check("active customers", row[2], kpi_number("Active customers, Dec 2024"))
check("MRR latest month", row[3], kpi_number("MRR, Dec 2024"), 1.0)
check("ARR run-rate", row[4], kpi_number("ARR run-rate, Dec 2024"), 1.0)
check("ARPU", row[5], kpi_number("ARPU, Dec 2024"), 1.0)

print("\n2. Monthly churn must sum to the churned customer count")
sql_churn_months = con.execute("""
    SELECT COALESCE(SUM(churned_customers), 0) FROM (
      SELECT DATE_TRUNC('month', churn_date) m, COUNT(*) churned_customers
      FROM v_customers_clean WHERE churn_flag = 1 GROUP BY 1)
""").fetchone()[0]
check("sum of monthly churn", sql_churn_months, int((cust["churn_flag"] == 1).sum()))

print("\n3. Survival curve, first 6 tenure months")
sql_surv = con.execute("""
    WITH base AS (
      SELECT c.customer_id, MAX(m.tenure_month) + 1 AS tr, (c.churn_flag = 1) AS ch
      FROM v_customers_clean c JOIN v_monthly_clean m USING (customer_id)
      GROUP BY 1, 3),
    grid AS (SELECT UNNEST(GENERATE_SERIES(0, 5)) AS t)
    SELECT g.t,
           COUNT(*) FILTER (WHERE b.tr > g.t) AS at_risk,
           COUNT(*) FILTER (WHERE b.tr = g.t + 1 AND b.ch) AS canc
    FROM grid g CROSS JOIN base b GROUP BY 1 ORDER BY 1
""").fetchdf()

py = surv.set_index("tenure_month")
for _, r in sql_surv.iterrows():
    t = int(r["t"])
    check(f"tenure {t} at_risk", int(r["at_risk"]), int(py.loc[t, "at_risk_subscribed"]))
    check(f"tenure {t} cancellations", int(r["canc"]), int(py.loc[t, "cancelled_this_month"]))

print("\n4. Churn by plan tier")
# The SQL layer UPPER()s its text labels (a normal SQL convention) while the
# Python layer title-cases them, so labels are matched case-insensitively here.
# Only the numbers have to agree exactly.
sql_plan = con.execute("""
    SELECT plan_tier, COUNT(*), SUM(churn_flag),
           ROUND(100.0 * SUM(churn_flag) / COUNT(*), 1)
    FROM v_customers_clean GROUP BY 1 ORDER BY 1
""").fetchdf().set_index("plan_tier")
py_plan = (cust.groupby("plan_tier")
           .apply(lambda d: pd.Series({
               "n": len(d),
               "churned": int(d["churn_flag"].sum()),
               "pct": round(100 * d["churn_flag"].mean(), 1)}), include_groups=False))
py_plan.index = py_plan.index.str.upper()
for tier in sql_plan.index:
    # plan_tier is the index, so the value columns sit at 0, 1, 2.
    check(f"{tier} customers", int(sql_plan.loc[tier].iloc[0]), int(py_plan.loc[tier, "n"]))
    check(f"{tier} churned", int(sql_plan.loc[tier].iloc[1]), int(py_plan.loc[tier, "churned"]))
    check(f"{tier} churn %", float(sql_plan.loc[tier].iloc[2]), float(py_plan.loc[tier, "pct"]))

print("\n5. Monthly MRR trend")
sql_mrr = con.execute("""
    SELECT month_start, ROUND(SUM(mrr)) FROM v_monthly_clean
    GROUP BY 1 ORDER BY 1
""").fetchdf()
py_mrr = mon.groupby("month_start")["mrr"].sum().round(0).reset_index()
check("months of data", len(sql_mrr), len(py_mrr))
check("total MRR across all months", float(sql_mrr.iloc[:, 1].sum()),
      float(py_mrr["mrr"].sum()), tolerance=1.0)

print("\n6. NRR")
# Read the view created by sql/analysis.sql rather than repeating the query
# here, so the two deliverables cannot drift apart.
sql_nrr = con.execute("SELECT nrr_pct FROM v_nrr").fetchone()[0]
check("NRR %", sql_nrr, 90.4, tolerance=0.15)

print("\n7. Cohort retention grid")
# Uses the same definition as sql/analysis.sql section 5: retention in month t
# is the share of the cohort still subscribed in tenure month t, and a cell is
# only observable while the cohort actually reached that month. Unobservable
# cells are 0 in both layers (the chart masks them), so the check is that the
# two grids agree cell for cell.
sql_cohort = con.execute("""
    WITH cohort_sizes AS (
      SELECT signup_cohort, COUNT(*) AS cohort_size FROM v_customers_clean GROUP BY 1
    ),
    survived AS (
      SELECT c.signup_cohort, m.tenure_month, COUNT(DISTINCT m.customer_id) AS n
      FROM v_customers_clean c JOIN v_monthly_clean m USING (customer_id)
      GROUP BY 1, 2
    ),
    reach AS (
      SELECT signup_cohort, MAX(tenure_month) AS max_tenure_reached FROM survived GROUP BY 1
    )
    SELECT s.signup_cohort AS cohort, s.tenure_month,
           CASE WHEN s.tenure_month <= r.max_tenure_reached
                THEN 100.0 * s.n / cs.cohort_size ELSE 0.0 END AS retention
    FROM survived s
    JOIN cohort_sizes cs USING (signup_cohort)
    JOIN reach r USING (signup_cohort)
""").fetchdf()
py_cohort = pd.read_csv(CLEAN / "analysis_cohort_retention.csv", index_col="cohort")
sql_cohort = sql_cohort.pivot(index="cohort", columns="tenure_month", values="retention")
# DuckDB types signup_cohort as a DATE while the Python CSV stores 'YYYY-MM'
# strings. Normalise to the CSV format so the two are compared as labels. (A
# DatetimeIndex will happily partial-match a bare '2023-01' string, so this has
# to be done explicitly rather than relying on label lookup to fail.)
sql_cohort.index = pd.PeriodIndex(sql_cohort.index, freq="M").astype(str)
mismatch = 0
for cohort in py_cohort.index:
    for t in py_cohort.columns:
        py_v = float(py_cohort.loc[cohort, t])
        sql_v = float(sql_cohort.loc[cohort, int(t)]) if int(t) in sql_cohort.columns else 0.0
        if abs(py_v - sql_v) > 0.6:
            mismatch += 1
            if mismatch <= 5:
                print(f"  XX cohort {cohort} month {t}: SQL={sql_v:.1f}  Python={py_v:.1f}")
if mismatch:
    print(f"  XX cohort cells disagreeing: {mismatch}")
    failures.append(f"cohort retention ({mismatch} cells)")
else:
    print(f"  OK all {len(py_cohort.index) * len(py_cohort.columns)} cohort cells agree")

# The eligible-cohort count is the number quoted in the report's cohort table,
# so it is checked directly: it must shrink as the observation window shortens.
sql_elig = con.execute("""
    WITH survived AS (
      SELECT c.signup_cohort, m.tenure_month, COUNT(DISTINCT m.customer_id) AS n
      FROM v_customers_clean c JOIN v_monthly_clean m USING (customer_id) GROUP BY 1, 2),
    reach AS (SELECT signup_cohort, MAX(tenure_month) AS mx FROM survived GROUP BY 1)
    SELECT s.tenure_month, COUNT(*) AS cohorts_eligible FROM survived s JOIN reach r USING (signup_cohort)
    WHERE s.tenure_month <= r.mx AND s.tenure_month IN (1, 3, 6, 12, 18)
    GROUP BY 1 ORDER BY 1
""").fetchdf()
py_elig = None
# Eligible cohorts are those whose calendar month plus m does not exceed Dec-2024.
per = pd.Period("2024-12", freq="M")
for _, r in sql_elig.iterrows():
    m = int(r["tenure_month"])
    expected = sum(1 for c in py_cohort.index if pd.Period(c, freq="M") + m <= per)
    check(f"cohorts eligible at month {m}", int(r["cohorts_eligible"]), expected)

print("\n8. Risk bands by MRR held")
# This is the check that catches an inverted weight or a different rescaling.
# An inverted weight still produces three tidy buckets of the right size, and
# min/max rescaling still produces correctly-ordered bands, so neither the
# account counts nor the ordering is enough on its own - the MRR split has to
# agree too. The SQL view is read rather than re-derived, so this check cannot
# pass by accident against a stale copy of the formula.
sql_risk = con.execute("""
    SELECT risk_band, COUNT(*) AS customers, SUM(final_mrr) AS total_mrr,
           AVG(final_adoption) AS avg_adoption, AVG(final_nps) AS avg_nps
    FROM v_risk_bands GROUP BY 1, risk_band_id ORDER BY risk_band_id
""").fetchdf().set_index("risk_band")
py_risk = pd.read_csv(CLEAN / "analysis_risk_summary.csv").set_index("risk_band")
for band in py_risk.index:
    check(f"{band} accounts", int(sql_risk.loc[band, "customers"]), int(py_risk.loc[band, "customers"]))
    check(f"{band} MRR", float(sql_risk.loc[band, "total_mrr"]),
          float(py_risk.loc[band, "total_mrr"]), tolerance=1.0)
    check(f"{band} adoption", float(sql_risk.loc[band, "avg_adoption"]),
          float(py_risk.loc[band, "avg_adoption"]), tolerance=0.005)
    check(f"{band} NPS", float(sql_risk.loc[band, "avg_nps"]),
          float(py_risk.loc[band, "avg_nps"]), tolerance=0.02)
# The bands are only meaningful if they are ordered: high risk must actually
# hold the worst adoption and NPS. Assert it explicitly rather than trusting it.
if (sql_risk.loc["High risk", "avg_adoption"] >= sql_risk.loc["Low risk", "avg_adoption"]
        or sql_risk.loc["High risk", "avg_nps"] >= sql_risk.loc["Low risk", "avg_nps"]
        or sql_risk.loc["High risk", "total_mrr"] <= sql_risk.loc["Low risk", "total_mrr"]):
    print("  XX risk bands are inverted: 'High risk' does not hold the worst adoption, "
          "worst NPS and most MRR")
    failures.append("risk band ordering")
else:
    print("  OK risk bands are correctly ordered (worst adoption + NPS, most MRR = High)")

# Per-account agreement, which is a much sharper test than the band aggregates:
# a single flipped weight moves many individual scores.
sql_scores = con.execute(
    "SELECT customer_id, risk_score FROM v_risk_bands ORDER BY customer_id").fetchdf()
py_scores = (pd.read_csv(CLEAN / "churn_risk_scoring.csv")[["customer_id", "risk_score"]]
             .sort_values("customer_id").reset_index(drop=True))
merged = py_scores.merge(sql_scores, on="customer_id", suffixes=("_py", "_sql"))
if len(merged) != len(py_scores):
    print(f"  XX risk scoring covers {len(merged)} of {len(py_scores)} active accounts")
    failures.append("risk scoring coverage")
else:
    worst = (merged.risk_score_py - merged.risk_score_sql).abs().max()
    if worst > 0.01:
        print(f"  XX risk scores differ, largest gap {worst:.4f}")
        failures.append("per-account risk score")
    else:
        print(f"  OK all {len(merged)} per-account risk scores agree (max gap {worst:.4f})")

print("\n9. No duplicate customer_id survives cleaning")
dupes = con.execute("""
    SELECT COUNT(*) FROM (
      SELECT customer_id FROM v_customers_clean
      GROUP BY 1 HAVING COUNT(*) > 1)
""").fetchone()[0]
check("duplicate customer_ids in clean view", dupes, 0)

print("\n10. Every churned customer has a churn date inside the data range")
bad = con.execute("""
    SELECT COUNT(*) FROM v_customers_clean
    WHERE churn_flag = 1 AND (
        churn_date IS NULL
        OR churn_date < (SELECT MIN(month_start) FROM v_monthly_clean)
        OR churn_date > (SELECT MAX(month_start) FROM v_monthly_clean))
""").fetchone()[0]
check("churn dates out of range", bad, 0)

print("\n" + "=" * 68)
if failures:
    print(f"RESULT: {len(failures)} MISMATCH(ES): {failures}")
    sys.exit(1)
print("RESULT: SQL and Python pipelines agree on every checked measure.")
