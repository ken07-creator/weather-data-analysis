"""
analysis.py
===========
Statistics, monthly summaries, correlation and plain-English interpretations.

Every number produced here is calculated from the DataFrame that is passed in.
Nothing is hardcoded, and every function copes with missing columns or empty data
(it returns an empty table / None instead of crashing).

Expected input: the `clean` DataFrame from data_processing.preprocess(), which uses the
internal column names: date, year, month, temperature, rainfall, humidity, ...
"""

from __future__ import annotations

import calendar
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TEMP_UNIT = "°C"

MONTHLY_TEMP_COLUMNS = ["month", "month_name", "avg_temperature", "lowest_reading",
                        "highest_reading", "observations", "years_covered",
                        "avg_of_yearly_means"]
MONTHLY_RAIN_COLUMNS = ["month", "month_name", "total_rainfall", "observations",
                        "missing_observations", "data_coverage_percent", "incomplete_month",
                        "rainy_observations", "years_covered", "avg_monthly_total_per_year"]
INCOMPLETE_MISSING_SHARE = 0.10     # more than 10% of a month's readings missing -> incomplete
INCOMPLETE_COVERAGE_PERCENT = 90.0  # fewer than 90% of the month's days present -> incomplete
MIN_PAIRS_FOR_CORRELATION = 10


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _valid(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Rows where every listed column exists and has a value; empty frame otherwise."""
    if df is None or any(c not in df.columns for c in columns):
        return pd.DataFrame(columns=columns)
    return df.dropna(subset=columns)


def rainfall_unit_label(unit: str) -> str:
    """Text to put in brackets after 'rainfall' in labels."""
    return unit if unit and unit != "unknown" else "unit not confirmed"


# --------------------------------------------------------------------------- #
# Descriptive statistics
# --------------------------------------------------------------------------- #

def temperature_stats(df: pd.DataFrame, column: str = "temperature") -> Dict[str, Optional[object]]:
    """
    Count, mean, median, min, max and standard deviation of valid temperature readings.

    The standard deviation is the *sample* standard deviation (n - 1 in the denominator),
    which is pandas' default. Values are None when they cannot be calculated.
    """
    empty = {"count": 0, "mean": None, "median": None, "min": None, "max": None,
             "std": None, "min_date": None, "max_date": None}
    if df is None or column not in df.columns:
        return empty
    values = df[column].dropna()
    if values.empty:
        return empty
    result = {
        "count": int(values.count()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": float(values.std()) if values.count() > 1 else None,
        "min_date": None,
        "max_date": None,
    }
    if "date" in df.columns:
        rows = df.dropna(subset=[column, "date"]).reset_index(drop=True)
        if not rows.empty:
            result["min_date"] = rows.loc[rows[column].idxmin(), "date"]
            result["max_date"] = rows.loc[rows[column].idxmax(), "date"]
    return result


def stats_table(stats: Dict[str, Optional[object]], unit: str = TEMP_UNIT) -> pd.DataFrame:
    """Turn the dictionary from temperature_stats into a table for display/download."""
    def show(value):
        return "n/a" if value is None else f"{value:.2f}"

    return pd.DataFrame({
        "statistic": ["Valid observations", f"Mean ({unit})", f"Median ({unit})",
                      f"Minimum ({unit})", f"Maximum ({unit})", f"Standard deviation ({unit})"],
        "value": [str(stats["count"]), show(stats["mean"]), show(stats["median"]),
                  show(stats["min"]), show(stats["max"]), show(stats["std"])],
    })


# --------------------------------------------------------------------------- #
# Monthly summaries
# --------------------------------------------------------------------------- #

def monthly_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average temperature for each CALENDAR month (Jan ... Dec) present in the data.

    Two different averages are reported, because they differ when years have unequal
    numbers of readings:
      avg_temperature     = mean of ALL readings in that calendar month (years pooled).
                            This is what the bar chart shows.
      avg_of_yearly_means = first average each year's month, then average those yearly
                            means (every year counts equally, however many readings it has).
    With only one year of data the two numbers are identical.
    """
    valid = _valid(df, ["month", "temperature"])
    if valid.empty:
        return pd.DataFrame(columns=MONTHLY_TEMP_COLUMNS)
    valid = valid.assign(month=valid["month"].astype(int))
    grouped = valid.groupby("month")["temperature"].agg(
        avg_temperature="mean", lowest_reading="min", highest_reading="max",
        observations="count").reset_index()
    if "year" in valid.columns:
        per_year = valid.groupby(["month", "year"])["temperature"].mean()
        grouped["years_covered"] = grouped["month"].map(per_year.groupby("month").size())
        grouped["avg_of_yearly_means"] = grouped["month"].map(per_year.groupby("month").mean())
    else:
        grouped["years_covered"] = 1
        grouped["avg_of_yearly_means"] = grouped["avg_temperature"]
    grouped.insert(1, "month_name", grouped["month"].map(lambda m: calendar.month_abbr[int(m)]))
    return grouped[MONTHLY_TEMP_COLUMNS]


def monthly_rainfall(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rainfall per CALENDAR month.

    total_rainfall             = SUM of all rainfall readings in that calendar month
                                 (all years pooled).
    avg_monthly_total_per_year = total_rainfall / number of distinct years that have data
                                 for that month (an average of monthly TOTALS, not of
                                 daily values).
    Months with no valid rainfall reading are absent (never shown as 0); their names are
    stored in result.attrs["months_without_data"].

    Missing readings are NEVER counted as 0, so a month with missing days has a total that is
    too low. To make that visible each month also reports:
      missing_observations   = rows in that month whose rainfall is missing or invalid
      data_coverage_percent  = days with a valid reading / days in the month (all years pooled).
                               Only calculated for daily data; blank otherwise.
      incomplete_month       = True if more than 10% of readings are missing or fewer than 90%
                               of the days are present. Totals of such months are under-counted.
    """
    valid = _valid(df, ["month", "rainfall"])
    if valid.empty:
        empty = pd.DataFrame(columns=MONTHLY_RAIN_COLUMNS)
        empty.attrs["months_without_data"] = _months_with_rows(df)
        return empty
    valid = valid.assign(month=valid["month"].astype(int))
    grouped = valid.groupby("month")
    result = grouped["rainfall"].agg(
        total_rainfall="sum", observations="count",
        rainy_observations=lambda s: int((s > 0).sum())).reset_index()
    if "year" in valid.columns:
        years = grouped["year"].nunique().reindex(result["month"]).values
    else:
        years = np.ones(len(result))
    result["years_covered"] = years.astype(int)
    result["avg_monthly_total_per_year"] = result["total_rainfall"] / result["years_covered"]
    result.insert(1, "month_name", result["month"].map(lambda m: calendar.month_abbr[int(m)]))

    # ---- disclosure of missing / incomplete data ----
    missing_rows = df[df["month"].notna() & df["rainfall"].isna()]
    missing_by_month = missing_rows.groupby(missing_rows["month"].astype(int)).size()
    result["missing_observations"] = result["month"].map(missing_by_month).fillna(0).astype(int)
    result["data_coverage_percent"] = _day_coverage(df, valid, result["month"])
    share_missing = result["missing_observations"] / (
        result["observations"] + result["missing_observations"])
    low_coverage = result["data_coverage_percent"] < INCOMPLETE_COVERAGE_PERCENT   # NaN -> False
    result["incomplete_month"] = (share_missing > INCOMPLETE_MISSING_SHARE) | low_coverage
    result = result[MONTHLY_RAIN_COLUMNS]
    present = set(result["month"])
    result.attrs["months_without_data"] = [m for m in _months_with_rows(df) if m not in present]
    return result


def _months_with_rows(df: pd.DataFrame) -> List[int]:
    """Calendar months that have at least one row (valid date), whatever the rainfall is."""
    if df is None or "month" not in df.columns:
        return []
    return sorted(int(m) for m in df["month"].dropna().unique())


def _day_coverage(df: pd.DataFrame, valid: pd.DataFrame, months: pd.Series) -> pd.Series:
    """% of calendar days with a valid rainfall reading, per month. NaN if data are not daily."""
    nan = pd.Series(np.nan, index=months.index)
    if "date" not in df.columns or "year" not in valid.columns:
        return nan
    days = df["date"].dropna().dt.normalize().drop_duplicates().sort_values()
    if len(days) < 5 or days.diff().dropna().median() > pd.Timedelta(days=1):
        return nan                                    # monthly/weekly data: coverage is meaningless
    with_data = valid.assign(_day=valid["date"].dt.normalize()).groupby("month")["_day"].nunique()
    expected = {}
    for month, year in valid[["month", "year"]].drop_duplicates().itertuples(index=False):
        expected[int(month)] = expected.get(int(month), 0) + calendar.monthrange(int(year), int(month))[1]
    coverage = months.map(lambda m: min(100.0, with_data.get(m, 0) / expected[int(m)] * 100))
    return coverage.astype(float)


# --------------------------------------------------------------------------- #
# Correlation and distribution
# --------------------------------------------------------------------------- #

def correlation_strength(r: float) -> str:
    """Words for the size of a correlation coefficient."""
    size = abs(r)
    if size < 0.2:
        return "very weak"
    if size < 0.4:
        return "weak"
    if size < 0.6:
        return "moderate"
    if size < 0.8:
        return "strong"
    return "very strong"


def temperature_humidity_correlation(df: pd.DataFrame,
                                     min_pairs: int = MIN_PAIRS_FOR_CORRELATION) -> Dict[str, object]:
    """
    Pearson correlation between temperature and humidity.

    Only rows where BOTH values exist are used. Returns {"available": False, "reason": ...}
    when there are too few pairs or one variable never changes.
    """
    pairs = _valid(df, ["temperature", "humidity"])
    n = len(pairs)
    if n < min_pairs:
        return {"available": False, "n": n,
                "reason": f"Only {n} valid temperature-humidity pair(s); at least {min_pairs} "
                          f"are needed for a meaningful correlation."}
    if pairs["temperature"].std() == 0 or pairs["humidity"].std() == 0:
        return {"available": False, "n": n,
                "reason": "One of the variables is constant, so a correlation cannot be calculated."}
    r = float(pairs["temperature"].corr(pairs["humidity"], method="pearson"))
    return {"available": True, "n": n, "r": r, "strength": correlation_strength(r),
            "direction": "negative" if r < 0 else "positive"}


def distribution_summary(df: pd.DataFrame, column: str = "temperature") -> Optional[Dict[str, float]]:
    """Mean, median, quartiles, skewness and the share of readings within one SD of the mean."""
    if df is None or column not in df.columns:
        return None
    values = df[column].dropna()
    if len(values) < 2:
        return None
    mean, std = float(values.mean()), float(values.std())
    within = float(((values - mean).abs() <= std).mean() * 100) if std > 0 else 100.0
    return {
        "n": int(len(values)), "mean": mean, "median": float(values.median()), "std": std,
        "q1": float(values.quantile(0.25)), "q3": float(values.quantile(0.75)),
        "skew": float(values.skew()) if len(values) >= 3 else float("nan"),
        "within_1sd_percent": within,
    }


# --------------------------------------------------------------------------- #
# Which charts can be drawn?
# --------------------------------------------------------------------------- #

CHART_TITLES = {
    "temperature_over_time": "Temperature over time",
    "monthly_temperature": "Average temperature by month",
    "monthly_rainfall": "Rainfall by month",
    "temperature_vs_humidity": "Temperature vs humidity",
    "temperature_distribution": "Temperature distribution",
}

_NEEDS = {
    "temperature_over_time": (["date", "temperature"], 2),
    "monthly_temperature": (["month", "temperature"], 1),
    "monthly_rainfall": (["month", "rainfall"], 1),
    "temperature_vs_humidity": (["temperature", "humidity"], 2),
    "temperature_distribution": (["temperature"], 2),
}


def chart_availability(df: pd.DataFrame) -> Dict[str, Tuple[bool, str]]:
    """
    For each of the five charts: (can_draw, reason_if_not).

    This lets the interface explain WHY a chart is missing and still show the others.
    """
    names = {"month": "date", "date": "date", "temperature": "temperature",
             "rainfall": "rainfall", "humidity": "humidity"}
    result: Dict[str, Tuple[bool, str]] = {}
    for chart, (columns, minimum) in _NEEDS.items():
        absent = sorted({names[c] for c in columns if df is None or c not in df.columns})
        if absent:
            result[chart] = (False, "This chart cannot be generated because the dataset has no "
                                    + " or ".join(absent) + " column (or it has no valid values).")
            continue
        n = len(df.dropna(subset=columns))
        if n < minimum:
            result[chart] = (False, f"This chart cannot be generated: only {n} row(s) have valid "
                                    f"values for {', '.join(names[c] for c in columns)} "
                                    f"(at least {minimum} needed).")
        else:
            result[chart] = (True, "")
    return result


# --------------------------------------------------------------------------- #
# Plain-English interpretations (all numbers come from the data)
# --------------------------------------------------------------------------- #

def interpret_time_series(df: pd.DataFrame) -> Optional[str]:
    d = _valid(df, ["date", "temperature"]).sort_values("date").reset_index(drop=True)
    if len(d) < 2:
        return None
    hi, lo = d.loc[d["temperature"].idxmax()], d.loc[d["temperature"].idxmin()]
    text = (f"The chart shows {len(d)} temperature readings from {d['date'].iloc[0]:%d %b %Y} "
            f"to {d['date'].iloc[-1]:%d %b %Y}. The highest reading was {hi['temperature']:.1f} "
            f"{TEMP_UNIT} on {hi['date']:%d %b %Y} and the lowest was {lo['temperature']:.1f} "
            f"{TEMP_UNIT} on {lo['date']:%d %b %Y}.")
    gap = d["date"].diff().dt.days.max()
    if pd.notna(gap) and gap > 1:
        text += f" The longest gap between two consecutive readings is {int(gap)} days."
    return text


def interpret_monthly_temperature(monthly: pd.DataFrame, n_years: int = 1) -> Optional[str]:
    if monthly is None or monthly.empty:
        return None
    hot = monthly.loc[monthly["avg_temperature"].idxmax()]
    cold = monthly.loc[monthly["avg_temperature"].idxmin()]
    text = (f"The data covers {len(monthly)} of 12 calendar months. The warmest month on average "
            f"is {hot['month_name']} ({hot['avg_temperature']:.1f} {TEMP_UNIT}) and the coolest "
            f"is {cold['month_name']} ({cold['avg_temperature']:.1f} {TEMP_UNIT}), a difference "
            f"of {hot['avg_temperature'] - cold['avg_temperature']:.1f} {TEMP_UNIT}.")
    if n_years > 1:
        text += f" Each bar combines readings from {n_years} different years."
    fewest = monthly.loc[monthly["observations"].idxmin()]
    if fewest["observations"] < 5:
        text += (f" Caution: {fewest['month_name']} is based on only "
                 f"{int(fewest['observations'])} reading(s), so its average is not reliable.")
    return text


def interpret_rainfall(monthly: pd.DataFrame, mode: str = "total",
                       unit: str = "unknown") -> Optional[str]:
    if monthly is None or monthly.empty:
        return None
    column = "total_rainfall" if mode == "total" else "avg_monthly_total_per_year"
    wet, dry = monthly.loc[monthly[column].idxmax()], monthly.loc[monthly[column].idxmin()]
    unit_text = rainfall_unit_label(unit)
    kind = ("total rainfall (sum of all readings)" if mode == "total"
            else "average monthly total per year")
    text = (f"Measured as {kind} in {unit_text}: the wettest month is {wet['month_name']} "
            f"({wet[column]:.1f}) and the driest month shown is {dry['month_name']} "
            f"({dry[column]:.1f}).")
    grand_total = monthly[column].sum()      # same quantity as the bars, so the % matches
    if grand_total > 0:
        share = wet[column] / grand_total * 100
        text += (f" {wet['month_name']} accounts for {share:.0f}% of the combined bar heights "
                 f"shown in the chart.")
    elif grand_total == 0:
        text += " No rainfall at all was recorded in the selected data."
    text += _rainfall_data_notes(monthly)
    return text


def _rainfall_data_notes(monthly: pd.DataFrame) -> str:
    """Sentences about missing/incomplete rainfall data (empty string when all is complete)."""
    notes = ""
    if "incomplete_month" in monthly.columns and monthly["incomplete_month"].any():
        parts = []
        for _, row in monthly[monthly["incomplete_month"]].iterrows():
            detail = f"{int(row['missing_observations'])} missing reading(s)"
            if pd.notna(row["data_coverage_percent"]):
                detail += f", {row['data_coverage_percent']:.0f}% of days present"
            parts.append(f"{row['month_name']} ({detail})")
        notes += (" CAUTION: " + "; ".join(parts) + ". Missing readings are not counted as zero, "
                  "so these totals are too low and cannot be compared fairly with complete months.")
    elif "missing_observations" in monthly.columns and monthly["missing_observations"].sum() > 0:
        notes += (f" Note: {int(monthly['missing_observations'].sum())} rainfall reading(s) are "
                  f"missing and are left out of the totals.")
    absent = monthly.attrs.get("months_without_data", [])
    if absent:
        names = ", ".join(calendar.month_abbr[m] for m in absent)
        notes += f" No valid rainfall readings exist for: {names} (not shown, not zero)."
    return notes


def interpret_correlation(corr: Dict[str, object]) -> str:
    if not corr.get("available"):
        return f"A correlation was not calculated. {corr.get('reason', '')}".strip()
    r = corr["r"]
    if corr["strength"] == "very weak":
        meaning = "there is essentially no linear relationship between temperature and humidity"
    elif r < 0:
        meaning = "higher temperatures tended to go with lower humidity"
    else:
        meaning = "higher temperatures tended to go with higher humidity"
    return (f"Pearson correlation r = {r:.2f} from {corr['n']} paired readings: a "
            f"{corr['strength']} {corr['direction']} relationship; in this data {meaning}. "
            f"Correlation does not prove causation: other factors such as season or "
            f"rainfall may influence both variables.")


def interpret_distribution(summary: Optional[Dict[str, float]]) -> Optional[str]:
    if not summary:
        return None
    skew = summary["skew"]
    if np.isnan(skew) or abs(skew) < 0.5:
        shape = "roughly symmetric"
    elif skew > 0:
        shape = "skewed to the right (a longer tail of higher temperatures)"
    else:
        shape = "skewed to the left (a longer tail of lower temperatures)"
    return (f"Across {summary['n']} readings the mean is {summary['mean']:.1f} {TEMP_UNIT} and "
            f"the median is {summary['median']:.1f} {TEMP_UNIT}. The middle 50% of readings lie "
            f"between {summary['q1']:.1f} and {summary['q3']:.1f} {TEMP_UNIT}. The standard "
            f"deviation is {summary['std']:.1f} {TEMP_UNIT}, and "
            f"{summary['within_1sd_percent']:.0f}% of readings fall within one standard "
            f"deviation of the mean. The distribution looks {shape}.")


def generate_interpretations(df: pd.DataFrame, rainfall_unit: str = "unknown",
                             rainfall_mode: str = "total") -> Dict[str, Optional[str]]:
    """One interpretation sentence block per chart (None when not available)."""
    n_years = int(df["year"].nunique()) if df is not None and "year" in df.columns else 1
    return {
        "temperature_over_time": interpret_time_series(df),
        "monthly_temperature": interpret_monthly_temperature(monthly_temperature(df), n_years),
        "monthly_rainfall": interpret_rainfall(monthly_rainfall(df), rainfall_mode, rainfall_unit),
        "temperature_vs_humidity": (interpret_correlation(temperature_humidity_correlation(df))
                                    if chart_availability(df)["temperature_vs_humidity"][0]
                                    else None),
        "temperature_distribution": interpret_distribution(distribution_summary(df)),
    }


def build_limitations(report: Dict[str, object], units: Dict[str, str], n_rows: int,
                      derived_temperature: bool = False, is_demo: bool = False) -> List[str]:
    """Honest limitations that apply to THIS dataset and the analysis."""
    notes = []
    if is_demo:
        notes.append("The data shown are SYNTHETIC demonstration data created by a program, "
                     "not real weather observations. Do not draw real-world conclusions.")
    if derived_temperature:
        notes.append("Temperature was derived as the midpoint of minimum and maximum "
                     "temperature, which only approximates the true daily mean.")
    if units.get("rainfall", "unknown") == "unknown":
        notes.append("The rainfall unit could not be confirmed, so rainfall totals have no unit.")
    if n_rows < 30:
        notes.append(f"Only {n_rows} rows are selected; patterns from so few records can be "
                     f"misleading.")
    if report.get("missing_rainfall"):
        notes.append(f"{report['missing_rainfall']} rainfall reading(s) are missing. Missing "
                     f"readings are not counted as zero, so monthly rainfall totals for months "
                     f"with gaps are too low (see the 'incomplete_month' column).")
    if report.get("invalid_dates"):
        notes.append(f"{report['invalid_dates']} unreadable date(s) were treated as missing; "
                     f"those rows are absent from time-based charts.")
    if report.get("implausible") and sum(report["implausible"].values()):
        notes.append(f"{sum(report['implausible'].values())} physically impossible value(s) were "
                     f"set to missing, which slightly reduces the sample.")
    if report.get("outliers_flagged"):
        notes.append(f"{report['outliers_flagged']} extreme value(s) were flagged but kept; they "
                     f"influence the mean and standard deviation.")
    notes.append("Monthly averages combine all years in the selected range, so they do not show "
                 "year-to-year changes or long-term climate trends.")
    notes.append("Correlation measures linear association only and does not prove causation.")
    return notes
