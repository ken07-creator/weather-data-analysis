"""Tests for statistics and monthly aggregation (analysis.py) using a small known dataset."""
import numpy as np
import pandas as pd
import pytest

from analysis import (build_limitations, chart_availability, generate_interpretations,
                      monthly_rainfall, monthly_temperature, stats_table,
                      temperature_humidity_correlation, temperature_stats)
from data_processing import preprocess
from tests.test_data_processing import small_raw


@pytest.fixture
def clean():
    return preprocess(small_raw()).clean          # 5 rows after removing the duplicate


def test_temperature_statistics_known_values(clean):
    stats = temperature_stats(clean)              # temps 20, 22, 24, 30, 32
    assert stats["count"] == 5
    assert stats["mean"] == pytest.approx(25.6)
    assert stats["median"] == 24
    assert stats["min"] == 20 and stats["max"] == 32
    assert stats["std"] == pytest.approx(np.std([20, 22, 24, 30, 32], ddof=1))
    assert stats["max_date"] == pd.Timestamp("2024-02-02")
    assert stats_table(stats).loc[0, "value"] == "5"


def test_monthly_temperature_averages(clean):
    monthly = monthly_temperature(clean).set_index("month")
    assert monthly.loc[1, "avg_temperature"] == pytest.approx(22.0)     # (20+22+24)/3
    assert monthly.loc[2, "avg_temperature"] == pytest.approx(31.0)     # (30+32)/2
    assert monthly.loc[1, "observations"] == 3


def test_monthly_rainfall_is_total_not_average(clean):
    monthly = monthly_rainfall(clean).set_index("month")
    assert monthly.loc[1, "total_rainfall"] == 15          # 0 + 10 + 5
    assert monthly.loc[2, "total_rainfall"] == 20          # 0 + 20
    assert monthly.loc[1, "rainy_observations"] == 2
    assert monthly.loc[1, "years_covered"] == 1


def test_monthly_rainfall_average_per_year_with_two_years():
    raw = pd.DataFrame({"Date": ["2023-01-05", "2024-01-05", "2024-01-06"],
                        "Rainfall": [10.0, 20.0, 10.0]})
    monthly = monthly_rainfall(preprocess(raw).clean).set_index("month")
    assert monthly.loc[1, "total_rainfall"] == 40
    assert monthly.loc[1, "years_covered"] == 2
    assert monthly.loc[1, "avg_monthly_total_per_year"] == 20


def test_month_with_only_missing_rainfall_is_absent_not_zero():
    raw = pd.DataFrame({"Date": ["2024-01-05", "2024-02-05"], "Rainfall": [5.0, np.nan]})
    monthly = monthly_rainfall(preprocess(raw).clean)
    assert monthly["month"].tolist() == [1]


def test_correlation_perfect_negative():
    df = pd.DataFrame({"temperature": np.arange(1.0, 13.0)})
    df["humidity"] = 100 - 2 * df["temperature"]
    corr = temperature_humidity_correlation(df)
    assert corr["available"] and corr["r"] == pytest.approx(-1.0)
    assert corr["direction"] == "negative" and corr["n"] == 12


def test_correlation_refused_when_too_few_pairs_or_constant():
    few = pd.DataFrame({"temperature": [1.0, 2, 3], "humidity": [3.0, 2, 1]})
    assert temperature_humidity_correlation(few)["available"] is False
    constant = pd.DataFrame({"temperature": np.arange(20.0), "humidity": 50.0})
    assert "constant" in temperature_humidity_correlation(constant)["reason"]


def test_correlation_uses_only_complete_pairs():
    df = pd.DataFrame({"temperature": np.arange(1.0, 13.0), "humidity": 100 - 2 * np.arange(1.0, 13.0)})
    df.loc[0, "humidity"] = np.nan
    assert temperature_humidity_correlation(df)["n"] == 11


@pytest.mark.parametrize("df", [pd.DataFrame(), pd.DataFrame({"temperature": []}),
                                pd.DataFrame({"temperature": [np.nan, np.nan]})])
def test_analysis_functions_do_not_crash_on_empty_or_missing_columns(df):
    assert temperature_stats(df)["count"] == 0
    assert monthly_temperature(df).empty
    assert monthly_rainfall(df).empty
    assert temperature_humidity_correlation(df)["available"] is False
    assert not any(ok for ok, _ in chart_availability(df).values())
    assert all(text is None for text in generate_interpretations(df).values())


def test_missing_columns_give_helpful_messages():
    df = preprocess(pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"],
                                  "Temperature": [20.0, 21.0]})).clean
    availability = chart_availability(df)
    assert availability["temperature_over_time"][0] is True
    assert availability["temperature_distribution"][0] is True
    ok, reason = availability["monthly_rainfall"]
    assert ok is False and "rainfall" in reason
    ok, reason = availability["temperature_vs_humidity"]
    assert ok is False and "humidity" in reason


def test_interpretations_use_real_numbers(clean):
    text = generate_interpretations(clean)
    assert "25.6" in text["temperature_distribution"]
    assert "32.0" in text["temperature_over_time"] and "02 Feb 2024" in text["temperature_over_time"]
    assert "Feb" in text["monthly_temperature"] and "31.0" in text["monthly_temperature"]
    assert "unit not confirmed" in text["monthly_rainfall"]


def test_limitations_mention_demo_and_unknown_unit():
    notes = build_limitations({}, {"rainfall": "unknown"}, 10, is_demo=True)
    joined = " ".join(notes)
    assert "SYNTHETIC" in joined and "rainfall unit" in joined and "Only 10 rows" in joined
