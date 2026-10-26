"""
visualizations.py
=================
The five required charts. Each function:

* takes the cleaned DataFrame (internal column names),
* draws ONLY from the real data passed in,
* returns a matplotlib Figure, or None if the chart cannot be drawn
  (missing column / too few valid rows).

Figures are created with matplotlib's `Figure` class directly (not pyplot), so no figures
pile up in memory when Streamlit reruns the script.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from analysis import (MIN_PAIRS_FOR_CORRELATION, chart_availability, monthly_rainfall, monthly_temperature, rainfall_unit_label,
                      temperature_humidity_correlation, TEMP_UNIT)

BLUE, ORANGE, GREEN, PURPLE = "#2a6f97", "#e4572e", "#2d9d78", "#6d597a"


def _new_figure(width: float = 10.5, height: float = 4.8):
    figure = Figure(figsize=(width, height))
    ax = figure.subplots()
    ax.grid(True, alpha=0.3)
    return figure, ax


def _rows(df: pd.DataFrame, columns, minimum: int = 1) -> Optional[pd.DataFrame]:
    """Valid rows for the chart, or None if columns are absent or too few rows exist."""
    if df is None or any(c not in df.columns for c in columns):
        return None
    valid = df.dropna(subset=list(columns))
    return valid if len(valid) >= minimum else None


def _break_at_gaps(dates: pd.Series, values: pd.Series):
    """
    Insert a missing value wherever readings are missing, so matplotlib leaves a visible
    GAP instead of drawing a straight line over days that have no observation.
    A gap is any spacing larger than 1.5 x the typical (median) spacing between readings.
    """
    dates = pd.Series(dates).reset_index(drop=True)
    values = pd.Series(values).reset_index(drop=True)
    steps = dates.diff()
    typical = steps.median()
    if pd.isna(typical) or typical <= pd.Timedelta(0):
        return dates, values
    big = np.flatnonzero((steps > typical * 1.5).to_numpy())       # positions that follow a gap
    if big.size == 0:
        return dates, values
    midpoints = (dates.iloc[big].to_numpy() - steps.iloc[big].to_numpy() / 2)
    new_dates = np.insert(dates.to_numpy(), big, midpoints)         # a NaN point inside each gap
    new_values = np.insert(values.to_numpy(dtype=float), big, np.nan)
    return pd.Series(new_dates), pd.Series(new_values)


# --------------------------------------------------------------------------- #
# 1. Temperature over time (line chart)
# --------------------------------------------------------------------------- #
def plot_temperature_over_time(df: pd.DataFrame) -> Optional[Figure]:
    data = _rows(df, ["date", "temperature"], minimum=2)
    if data is None:
        return None
    data = data.sort_values("date")                       # chronological order
    figure, ax = _new_figure(11, 4.8)
    dates, temps = _break_at_gaps(data["date"], data["temperature"])
    ax.plot(dates, temps, color=ORANGE, linewidth=1.3,
            marker="o" if len(data) <= 60 else None, markersize=3.5)
    ax.set_title("Temperature Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Temperature ({TEMP_UNIT})")
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------- #
# 2. Average temperature by month (bar chart)
# --------------------------------------------------------------------------- #
def plot_monthly_temperature(df: pd.DataFrame) -> Optional[Figure]:
    monthly = monthly_temperature(df)
    if monthly.empty:
        return None
    figure, ax = _new_figure(9, 4.8)
    bars = ax.bar(monthly["month_name"], monthly["avg_temperature"], color=ORANGE, width=0.65)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set_title("Average Temperature by Calendar Month\n"
                 "(each bar = mean of all valid readings in that calendar month)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel(f"Average temperature ({TEMP_UNIT})")
    ax.margins(y=0.12)
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------- #
# 3. Rainfall by month (bar chart)
# --------------------------------------------------------------------------- #
def plot_monthly_rainfall(df: pd.DataFrame, unit: str = "unknown",
                          mode: str = "total") -> Optional[Figure]:
    """
    mode = "total"   -> sum of all rainfall readings in each calendar month
    mode = "average" -> average monthly TOTAL per year (only differs when data span >1 year)
    """
    monthly = monthly_rainfall(df)
    if monthly.empty:
        return None
    column = "total_rainfall" if mode == "total" else "avg_monthly_total_per_year"
    kind = "Total rainfall" if mode == "total" else "Average monthly total rainfall per year"
    figure, ax = _new_figure(9, 4.8)
    bars = ax.bar(monthly["month_name"], monthly[column], color=BLUE, width=0.65)
    incomplete = monthly["incomplete_month"].to_numpy(dtype=bool)
    for bar, is_incomplete in zip(bars, incomplete):
        if is_incomplete:                         # missing days -> total is too low: show it
            bar.set_facecolor("#a9c6dc")
            bar.set_hatch("//")
            bar.set_edgecolor(BLUE)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    title = f"{kind} by Calendar Month"
    if incomplete.any():
        title += "\nHatched bars: incomplete month, total too low"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel(f"{kind} ({rainfall_unit_label(unit)})")
    ax.margins(y=0.12)
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------- #
# 4. Temperature vs humidity (scatter plot)
# --------------------------------------------------------------------------- #
def plot_temperature_vs_humidity(df: pd.DataFrame) -> Optional[Figure]:
    data = _rows(df, ["temperature", "humidity"], minimum=2)
    if data is None:
        return None
    corr = temperature_humidity_correlation(df)
    figure, ax = _new_figure(8.5, 5.2)
    ax.scatter(data["temperature"], data["humidity"], alpha=0.5, s=22, color=GREEN)
    title = "Temperature vs Humidity"
    if corr["available"]:
        slope, intercept = np.polyfit(data["temperature"], data["humidity"], 1)
        xs = np.linspace(data["temperature"].min(), data["temperature"].max(), 50)
        ax.plot(xs, slope * xs + intercept, color=PURPLE, linewidth=2, label="Trend line")
        ax.legend()
        title += f"\n(Pearson r = {corr['r']:.2f}, n = {corr['n']})"
    else:
        short = (f"only {corr['n']} valid pairs, at least {MIN_PAIRS_FOR_CORRELATION} needed"
                 if corr["n"] < MIN_PAIRS_FOR_CORRELATION else "one variable is constant")
        title += f"\n(correlation not calculated: {short})"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(f"Temperature ({TEMP_UNIT})")
    ax.set_ylabel("Humidity (%)")
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------- #
# 5. Temperature distribution (histogram)
# --------------------------------------------------------------------------- #
def plot_temperature_distribution(df: pd.DataFrame) -> Optional[Figure]:
    data = _rows(df, ["temperature"], minimum=2)
    if data is None:
        return None
    values = data["temperature"]
    bins = int(min(30, max(5, np.sqrt(len(values)))))
    figure, ax = _new_figure(9, 4.8)
    ax.hist(values, bins=bins, color=BLUE, edgecolor="white")
    ax.axvline(values.mean(), color=ORANGE, linestyle="--", linewidth=2,
               label=f"Mean = {values.mean():.1f} {TEMP_UNIT}")
    ax.axvline(values.median(), color=PURPLE, linestyle=":", linewidth=2,
               label=f"Median = {values.median():.1f} {TEMP_UNIT}")
    ax.set_title("Distribution of Temperature Readings", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"Temperature ({TEMP_UNIT})")
    ax.set_ylabel("Number of readings")
    ax.legend()
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------- #
# One entry point for interfaces that let the user pick a chart by name
# --------------------------------------------------------------------------- #
CHART_BUILDERS = {
    "temperature_over_time": lambda df, unit, mode: plot_temperature_over_time(df),
    "monthly_temperature": lambda df, unit, mode: plot_monthly_temperature(df),
    "monthly_rainfall": lambda df, unit, mode: plot_monthly_rainfall(df, unit, mode),
    "temperature_vs_humidity": lambda df, unit, mode: plot_temperature_vs_humidity(df),
    "temperature_distribution": lambda df, unit, mode: plot_temperature_distribution(df),
}


def build_chart(key: str, df: pd.DataFrame, rainfall_unit: str = "unknown",
                rainfall_mode: str = "total"):
    """
    Build one of the five charts by key (see CHART_BUILDERS).

    Returns (figure, message). `figure` is None when the chart cannot be drawn, and
    `message` then explains why in plain language. Never raises for missing data.
    """
    if key not in CHART_BUILDERS:
        return None, f"Unknown chart: {key}."
    if df is None or len(df) == 0:
        return None, "There is no data to plot."
    available, reason = chart_availability(df)[key]
    if not available:
        return None, reason
    try:
        figure = CHART_BUILDERS[key](df, rainfall_unit, rainfall_mode)
    except Exception as exc:                      # keep the desktop app alive
        return None, f"The chart could not be drawn: {exc}"
    if figure is None:
        return None, "This chart cannot be generated from the available data."
    return figure, ""
