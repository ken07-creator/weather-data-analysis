"""
Regression tests for problems found in the final audit. Each test states the expected
value that was worked out by hand, not copied from the program's output.
"""
import io
import time

import numpy as np
import pandas as pd
import pytest

from analysis import (build_limitations, interpret_correlation, interpret_rainfall,
                      monthly_rainfall)
from data_processing import (DataLoadError, cleaned_export_frame, guess_rainfall_unit, load_csv,
                             parse_dates, preprocess, to_csv_bytes)
from demo_data import demo_csv_bytes
from visualizations import _break_at_gaps, plot_monthly_rainfall, plot_temperature_vs_humidity


# ---------------- dates ----------------
@pytest.mark.parametrize("values", [[20240101, 20240102, 20240103],
                                    ["20240101", "20240102", "20240103"],
                                    [20240101.0, 20240102.0, 20240103.0]])
def test_compact_yyyymmdd_dates_are_not_read_as_1970(values):
    parsed = parse_dates(pd.Series(values))
    assert [str(d.date()) for d in parsed] == ["2024-01-01", "2024-01-02", "2024-01-03"]


@pytest.mark.parametrize("dayfirst", [False, True])
def test_year_first_slash_dates_are_never_day_first(dayfirst):
    parsed = parse_dates(pd.Series(["2024/01/05", "2024.02.06"]), dayfirst=dayfirst)
    assert [str(d.date()) for d in parsed] == ["2024-01-05", "2024-02-06"]


def test_plain_numbers_that_are_not_dates_become_unreadable_not_1970():
    raw = pd.DataFrame({"Date": [45292, 45293, 45294], "Temperature": [20.0, 21.0, 22.0]})
    result = preprocess(raw)                                  # 45292 = an Excel serial number
    assert result.report["invalid_dates"] == 3
    assert not any(str(d).startswith("1970") for d in result.clean.get("date", []))


def test_real_datetime_column_still_parses():
    raw = pd.DataFrame({"Date": pd.date_range("2024-03-01", periods=3), "Temperature": [1.0, 2, 3]})
    assert preprocess(raw).clean["month"].tolist() == [3, 3, 3]


# ---------------- loading ----------------
@pytest.mark.parametrize("blob", [b"PK\x03\x04\x14\x00" + bytes(range(256)) * 3, b"\x00\x01,\n\x00"])
def test_binary_files_get_the_binary_message_not_a_parser_error(blob):
    with pytest.raises(DataLoadError, match="binary file"):
        load_csv(blob)


# ---------------- silent failures now explained ----------------
def test_unlabelled_fahrenheit_is_explained_not_silent():
    raw = pd.DataFrame({"Date": pd.date_range("2024-07-01", periods=10).astype(str),
                        "Temperature": [68, 70, 75, 80, 85, 90, 95, 72, 66, 88]})
    result = preprocess(raw)
    assert result.clean["temperature"].isna().all()
    assert any("Fahrenheit" in w and "10 of 10" in w for w in result.warnings)
    fixed = preprocess(raw, temperature_unit="F")             # the user's remedy
    assert fixed.clean["temperature"].notna().all()
    assert fixed.clean["temperature"].iloc[0] == pytest.approx(20.0)      # (68-32)*5/9


def test_decimal_comma_file_gets_a_hint():
    raw = load_csv(b"Date;Temp\n2024-01-01;20,5\n2024-01-02;21,5\n2024-01-03;19,0\n")
    result = preprocess(raw)
    assert any("decimal point" in w and "20,5" in w for w in result.warnings)


def test_humidity_fractions_are_flagged():
    raw = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=6).astype(str),
                        "Humidity": [0.4, 0.5, 0.6, 0.7, 0.55, 0.65]})
    assert any("fractions" in w for w in preprocess(raw).warnings)


def test_duplicate_date_warning_is_accurate():
    raw = pd.DataFrame({"Date": ["2024-01-01", "2024-01-01", "2024-01-02"], "Temperature": [20, 20, 21]})
    kept = preprocess(raw, remove_duplicates=False)
    assert any("switched off" in w for w in kept.warnings)
    assert not any("different values" in w for w in kept.warnings)
    differing = pd.DataFrame({"Date": ["2024-01-01", "2024-01-01"], "Temperature": [20, 25]})
    assert any("different values" in w for w in preprocess(differing).warnings)


def test_rainfall_unit_names():
    assert guess_rainfall_unit("rainfall_inches") == "inches"
    assert guess_rainfall_unit("Rainfall (in)") == "inches"
    assert guess_rainfall_unit("rainfall_cm") == "cm"
    assert guess_rainfall_unit("Rainfall (mm)") == "mm"
    assert guess_rainfall_unit("Rainfall") == "unknown"


# ---------------- missing rainfall is disclosed ----------------
def january_with_missing_days():
    """Jan 2024: 31 rows of 10 mm, but the first 20 are missing. Feb: 29 complete days of 10 mm."""
    dates = pd.date_range("2024-01-01", "2024-02-29")
    rain = np.full(len(dates), 10.0)
    rain[:20] = np.nan
    return preprocess(pd.DataFrame({"Date": dates.astype(str), "Temperature": 25.0,
                                    "Rainfall (mm)": rain}))


def test_monthly_rainfall_reports_missing_readings_and_coverage():
    monthly = monthly_rainfall(january_with_missing_days().clean).set_index("month")
    assert monthly.loc[1, "total_rainfall"] == 110              # 11 days x 10 mm
    assert monthly.loc[1, "missing_observations"] == 20
    assert monthly.loc[1, "data_coverage_percent"] == pytest.approx(11 / 31 * 100)
    assert bool(monthly.loc[1, "incomplete_month"]) is True
    assert monthly.loc[2, "total_rainfall"] == 290              # 29 days x 10 mm
    assert monthly.loc[2, "missing_observations"] == 0
    assert monthly.loc[2, "data_coverage_percent"] == pytest.approx(100.0)
    assert bool(monthly.loc[2, "incomplete_month"]) is False


def test_interpretation_warns_about_incomplete_months():
    result = january_with_missing_days()
    text = interpret_rainfall(monthly_rainfall(result.clean), "total", "mm")
    assert "CAUTION" in text and "Jan (20 missing reading(s), 35% of days present)" in text
    assert "not counted as zero" in text
    assert "Feb (" not in text.split("CAUTION")[1]              # complete month is not blamed


def test_cleaning_warning_and_limitations_mention_missing_rainfall():
    result = january_with_missing_days()
    assert result.report["missing_rainfall"] == 20
    assert any("20 rainfall reading(s) are missing" in w for w in result.warnings)
    assert any("20 rainfall reading(s) are missing" in n
               for n in build_limitations(result.report, result.units, len(result.clean)))


def test_rainfall_chart_marks_incomplete_bars():
    fig = plot_monthly_rainfall(january_with_missing_days().clean, "mm")
    ax = fig.axes[0]
    assert [p.get_hatch() for p in ax.patches] == ["//", None] or [p.get_hatch() for p in ax.patches] == ["//", ""]
    assert [p.get_height() for p in ax.patches] == [110, 290]   # bars still show the real totals
    assert "Hatched bars" in ax.get_title()
    assert max(len(line) for line in ax.get_title().split("\n")) <= 50   # must fit the figure width


def test_complete_data_has_no_hatching_and_no_caution():
    dates = pd.date_range("2024-01-01", "2024-02-29")
    clean = preprocess(pd.DataFrame({"Date": dates.astype(str), "Temperature": 25.0,
                                     "Rainfall (mm)": 1.0})).clean
    assert not monthly_rainfall(clean)["incomplete_month"].any()
    fig = plot_monthly_rainfall(clean, "mm")
    assert all(not p.get_hatch() for p in fig.axes[0].patches)
    assert "Hatched" not in fig.axes[0].get_title()
    assert "CAUTION" not in interpret_rainfall(monthly_rainfall(clean), "total", "mm")


def test_single_missing_reading_is_a_note_not_a_caution():
    clean = preprocess(load_csv(demo_csv_bytes())).clean          # demo has one invalid rainfall
    text = interpret_rainfall(monthly_rainfall(clean), "total", "mm")
    assert "CAUTION" not in text and "Note: 1 rainfall reading(s) are missing" in text


def test_month_without_any_valid_rainfall_is_named_not_shown_as_zero():
    raw = pd.DataFrame({"Date": ["2024-01-05", "2024-02-05", "2024-02-06"],
                        "Temperature": 20.0, "Rainfall": [np.nan, 5.0, 3.0]})
    monthly = monthly_rainfall(preprocess(raw).clean)
    assert monthly["month"].tolist() == [2]
    assert monthly.attrs["months_without_data"] == [1]
    assert "No valid rainfall readings exist for: Jan" in interpret_rainfall(monthly, "total", "mm")


def test_coverage_is_blank_for_non_daily_data():
    raw = pd.DataFrame({"Date": ["2024-01-15", "2024-02-15", "2024-03-15", "2024-04-15",
                                 "2024-05-15", "2024-06-15"],
                        "Temperature": 20.0, "Rainfall": [50.0, 60, 70, 80, 90, 100]})
    monthly = monthly_rainfall(preprocess(raw).clean)
    assert monthly["data_coverage_percent"].isna().all()          # monthly data: not "3% complete"
    assert not monthly["incomplete_month"].any()


# ---------------- exported CSV loads back with the same meaning ----------------
def test_export_names_carry_units():
    result = preprocess(load_csv(demo_csv_bytes()))
    exported = cleaned_export_frame(result.clean, result.units)
    assert {"temperature_c", "min_temperature_c", "max_temperature_c", "humidity_percent",
            "rainfall_mm", "wind_speed", "date"} <= set(exported.columns)
    assert "temperature" not in exported.columns


def test_export_without_known_rainfall_unit_keeps_plain_name():
    raw = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"], "Temperature": [20.0, 21.0],
                        "Rainfall": [1.0, 2.0]})
    result = preprocess(raw)
    assert "rainfall" in cleaned_export_frame(result.clean, result.units).columns


@pytest.mark.parametrize("unit", ["mm", "cm", "inches"])
def test_export_round_trip_keeps_units_and_values(unit):
    raw = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=5).astype(str),
                        "Temperature": [20.5, 21.0, 22.5, 19.0, 23.0],
                        "Rainfall": [0.0, 1.5, 0.0, 2.5, 0.5], "Humidity": [60, 65, 70, 55, 50]})
    first = preprocess(raw, rainfall_unit=unit)
    again = preprocess(load_csv(to_csv_bytes(cleaned_export_frame(first.clean, first.units))))
    assert again.units["rainfall"] == unit
    assert again.units["temperature"] == "°C"
    for column in ["temperature", "rainfall", "humidity"]:
        assert again.clean[column].tolist() == first.clean[column].tolist()
    assert again.clean["date"].tolist() == first.clean["date"].tolist()
    assert len(again.flagged) == 0 and len(again.clean) == 5


def test_fahrenheit_export_is_not_converted_twice_on_reload():
    raw = pd.DataFrame({"Date": pd.date_range("2024-07-01", periods=4).astype(str),
                        "Temperature (F)": [68.0, 77.0, 86.0, 95.0]})
    first = preprocess(raw)
    assert first.clean["temperature"].tolist() == pytest.approx([20, 25, 30, 35])
    again = preprocess(load_csv(to_csv_bytes(cleaned_export_frame(first.clean, first.units))))
    assert again.clean["temperature"].tolist() == pytest.approx([20, 25, 30, 35])


def test_demo_export_round_trip_is_lossless():
    first = preprocess(load_csv(demo_csv_bytes()))
    again = preprocess(load_csv(to_csv_bytes(cleaned_export_frame(first.clean, first.units))))
    assert again.units["rainfall"] == "mm" and len(again.clean) == len(first.clean) == 731
    for column in ["temperature", "min_temperature", "max_temperature", "rainfall", "humidity",
                   "wind_speed"]:
        pd.testing.assert_series_equal(again.clean[column], first.clean[column])


# ---------------- chart labels and wording ----------------
def test_scatter_title_gives_the_real_reason():
    df = pd.DataFrame({"temperature": np.full(20, 25.0), "humidity": np.linspace(40, 80, 20)})
    title = plot_temperature_vs_humidity(df).axes[0].get_title()
    assert "constant" in title and "too few" not in title
    few = pd.DataFrame({"temperature": [1.0, 2, 3], "humidity": [3.0, 2, 1]})
    assert "only 3 valid pairs, at least 10 needed" in plot_temperature_vs_humidity(few).axes[0].get_title()


def test_very_weak_correlation_is_not_described_as_a_tendency():
    text = interpret_correlation({"available": True, "n": 50, "r": 0.05,
                                  "strength": "very weak", "direction": "positive"})
    assert "essentially no linear relationship" in text and "tended" not in text
    strong = interpret_correlation({"available": True, "n": 50, "r": -0.7,
                                    "strength": "strong", "direction": "negative"})
    assert "tended to go with lower humidity" in strong


# ---------------- performance ----------------
def test_gap_breaking_inserts_nan_and_is_fast():
    dates = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03",
                                      "2024-01-10", "2024-01-11"]))
    new_dates, new_values = _break_at_gaps(dates, pd.Series([1.0, 2, 3, 4, 5]))
    assert list(new_values[:3]) == [1, 2, 3] and np.isnan(new_values[3])
    assert str(new_dates[3].date()) == "2024-01-06"               # inside the gap
    assert list(new_values[4:]) == [4, 5]
    big_dates = pd.Series(pd.date_range("2000-01-01", periods=200_000, freq="h"))
    start = time.time()
    _break_at_gaps(big_dates, pd.Series(np.random.rand(200_000)))
    assert time.time() - start < 1.0                              # the old loop took ~3.5 s


def test_large_file_cleans_and_charts_in_reasonable_time():
    from visualizations import CHART_BUILDERS, build_chart
    n = 60_000
    rng = np.random.default_rng(0)
    csv = pd.DataFrame({"Date": pd.date_range("2010-01-01", periods=n, freq="h").strftime("%Y-%m-%d %H:%M"),
                        "Temperature (C)": rng.normal(25, 5, n).round(1),
                        "Rainfall (mm)": np.where(rng.random(n) < .1, 2.0, 0.0),
                        "Humidity (%)": rng.uniform(30, 90, n).round(1)}).to_csv(index=False).encode()
    start = time.time()
    clean = preprocess(load_csv(csv)).clean
    assert len(clean) == n
    for key in CHART_BUILDERS:
        figure, message = build_chart(key, clean, "mm")
        assert figure is not None, message
    assert time.time() - start < 30
