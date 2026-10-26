"""
Tests that check NUMBERS against answers worked out by hand (or with Python's standard
`statistics` module, which is independent of pandas).

The small dataset below is used throughout:

    date        temp  rain  humidity
    2024-01-01   10     2     50
    2024-01-02   20     4     60
    2024-01-03   30     0     70
    2025-01-01   40     6     80
    2025-02-01    5    10     90
"""
import io
import statistics

import numpy as np
import pandas as pd
import pytest

from analysis import (chart_availability, interpret_rainfall, monthly_rainfall,
                      monthly_temperature, stats_table, temperature_humidity_correlation,
                      temperature_stats)
from data_processing import (DataLoadError, count_duplicates, detect_columns, filter_by_date,
                             load_csv, missing_value_table, parse_dates, preprocess,
                             to_csv_bytes, to_numeric_clean)
from visualizations import (plot_monthly_rainfall, plot_monthly_temperature,
                            plot_temperature_distribution, plot_temperature_over_time,
                            plot_temperature_vs_humidity)

SAMPLE_FILE = "data/standin_weather_2025_SYNTHETIC.csv"


def known_raw():
    return pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03", "2025-01-01", "2025-02-01"],
        "Temperature": [10, 20, 30, 40, 5],
        "Rainfall (mm)": [2, 4, 0, 6, 10],
        "Humidity": [50, 60, 70, 80, 90],
    })


@pytest.fixture
def clean():
    return preprocess(known_raw()).clean


# ------------------------------------------------------------------ loading the sample file
def test_sample_file_loads_and_is_understood():
    raw = load_csv(SAMPLE_FILE)
    assert len(raw) == 368                        # 365 days + 3 repeated rows
    assert detect_columns(raw.columns) == {
        "date": "Date", "min_temperature": "Min_Temperature_C",
        "max_temperature": "Max_Temperature_C", "temperature": "Temperature_C",
        "rainfall": "Rainfall_mm", "humidity": "Humidity_percent",
        "wind_speed": "Wind_Speed_kmh"}


def test_sample_file_missing_values_and_duplicates_are_detected():
    raw = load_csv(SAMPLE_FILE)
    assert count_duplicates(raw) == 3
    missing = missing_value_table(raw).set_index("column")["missing"]
    # 3 / 1 / 2 missing cells were planted, one of the temperature rows is also repeated
    assert missing["Temperature_C"] >= 3 and missing["Rainfall_mm"] >= 1
    result = preprocess(raw)
    assert result.report["duplicates_found"] == 3
    assert result.report["rows_after"] == 365
    assert result.clean["date"].is_monotonic_increasing
    assert result.clean["date"].is_unique            # one row per day after de-duplication
    # Nothing was invented: missing cells stay missing
    assert result.report["after_missing"]["temperature"] == 3
    assert result.report["after_missing"]["rainfall"] == 1
    assert result.report["after_missing"]["humidity"] == 2


def test_sample_file_all_five_charts_draw():
    clean = preprocess(load_csv(SAMPLE_FILE)).clean
    assert all(ok for ok, _ in chart_availability(clean).values())
    for plot in (plot_temperature_over_time, plot_monthly_temperature, plot_monthly_rainfall,
                 plot_temperature_vs_humidity, plot_temperature_distribution):
        assert plot(clean) is not None


# ------------------------------------------------------------------ dates and sorting
def test_unsorted_dates_are_sorted_chronologically():
    raw = known_raw().iloc[[3, 0, 4, 2, 1]].reset_index(drop=True)
    clean = preprocess(raw).clean
    assert clean["temperature"].tolist() == [10, 20, 30, 40, 5]
    assert clean["date"].is_monotonic_increasing


def test_iso_dates_are_not_swapped_when_dayfirst_is_ticked():
    # Regression test: pandas used to read 2025-01-02 as 1 February when dayfirst=True.
    parsed = parse_dates(pd.Series(["2025-01-02"]), dayfirst=True)
    assert parsed.iloc[0] == pd.Timestamp("2025-01-02")
    # ...while genuinely day-first text is still read day-first.
    assert parse_dates(pd.Series(["02/03/2025"]), dayfirst=True).iloc[0] == pd.Timestamp("2025-03-02")


# ------------------------------------------------------------------ numeric conversion
def test_numeric_conversion_is_safe():
    out = to_numeric_clean(pd.Series(["12.5", " 7 ", "1e2", "error", "", None, "inf"]))
    assert out.iloc[:3].tolist() == [12.5, 7.0, 100.0]
    assert out.iloc[3:].isna().all()


def test_before_and_after_missing_counts_agree_with_the_flagged_table():
    raw = pd.DataFrame({
        "Date": [f"2024-01-{d:02d}" for d in range(1, 7)],
        "Temperature": [20, 21, 75, 22, np.nan, 23],       # 75 C impossible, one blank
        "Rainfall (mm)": [0, 1, 2, "error", 3, 4],         # one text value
        "Humidity": [50, 140, 55, 60, 65, 70]})            # 140 % impossible
    result = preprocess(raw)
    rows = result.summary.set_index("Item")
    assert rows.loc["Missing / invalid temperature"].tolist() == ["1", "2"]   # blank | + 75 C
    assert rows.loc["Missing / invalid rainfall"].tolist() == ["1", "1"]      # 'error' counted
    assert rows.loc["Missing / invalid humidity"].tolist() == ["0", "1"]      # 140 % -> missing
    assert len(result.flagged) == 3                                           # 75, error, 140


# ------------------------------------------------------------------ duplicates
def test_exact_duplicates_removed_but_same_values_on_different_days_kept():
    raw = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"],
                        "Temperature": [25, 25, 25, 25]})   # same value on 3 different days
    result = preprocess(raw)
    assert len(result.clean) == 3                            # only the repeated 2 Jan row goes
    assert result.clean["date"].dt.day.tolist() == [1, 2, 3]


def test_repeated_rows_without_a_date_column_are_kept_and_reported():
    raw = pd.DataFrame({"Temperature": [25, 25, 25, 30], "Humidity": [60, 60, 60, 50]})
    result = preprocess(raw)
    assert len(result.clean) == 4
    assert any("no date column" in w for w in result.warnings)


def test_same_date_with_different_values_is_kept_and_warned():
    raw = pd.DataFrame({"Date": ["2024-01-01", "2024-01-01", "2024-01-02"],
                        "Temperature": [20, 25, 21]})
    result = preprocess(raw)
    assert len(result.clean) == 3
    assert any("more than one row" in w for w in result.warnings)


# ------------------------------------------------------------------ units
def test_fahrenheit_is_detected_from_the_header_and_converted():
    raw = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"], "Temp (F)": [50, 68]})
    result = preprocess(raw)                                 # default: auto-detect
    assert result.clean["temperature"].round(6).tolist() == [10.0, 20.0]


def test_rainfall_unit_is_read_from_the_header_or_stays_unknown():
    assert preprocess(known_raw()).units["rainfall"] == "mm"
    raw = known_raw().rename(columns={"Rainfall (mm)": "Rainfall"})
    assert preprocess(raw).units["rainfall"] == "unknown"


# ------------------------------------------------------------------ monthly temperature
def test_monthly_temperature_pooled_mean_and_mean_of_yearly_means(clean):
    jan = monthly_temperature(clean).set_index("month").loc[1]
    assert jan["observations"] == 4
    assert jan["avg_temperature"] == pytest.approx(statistics.mean([10, 20, 30, 40]))   # 25
    # 2024 January mean = 20, 2025 January mean = 40  ->  mean of the two = 30
    assert jan["avg_of_yearly_means"] == pytest.approx(30.0)
    assert jan["years_covered"] == 2
    assert jan["lowest_reading"] == 10 and jan["highest_reading"] == 40
    feb = monthly_temperature(clean).set_index("month").loc[2]
    assert feb["avg_temperature"] == 5 and feb["avg_of_yearly_means"] == 5


def test_monthly_temperature_bar_heights_match_the_table(clean):
    heights = [p.get_height() for p in plot_monthly_temperature(clean).axes[0].patches]
    assert heights == pytest.approx([25.0, 5.0])


# ------------------------------------------------------------------ monthly rainfall
def test_monthly_rainfall_total_versus_average_per_year(clean):
    table = monthly_rainfall(clean).set_index("month")
    assert table.loc[1, "total_rainfall"] == 12                  # 2 + 4 + 0 + 6
    assert table.loc[1, "avg_monthly_total_per_year"] == 6       # 12 / 2 years, NOT 12 / 4 days
    assert table.loc[1, "rainy_observations"] == 3
    assert table.loc[2, "total_rainfall"] == 10


def test_rainfall_chart_modes_show_different_numbers(clean):
    total = [p.get_height() for p in plot_monthly_rainfall(clean, "mm", "total").axes[0].patches]
    average = [p.get_height() for p in plot_monthly_rainfall(clean, "mm", "average").axes[0].patches]
    assert total == [12, 10] and average == [6, 10]


def test_rainfall_interpretation_matches_the_chart_mode(clean):
    monthly = monthly_rainfall(clean)
    total_text = interpret_rainfall(monthly, "total", "mm")
    avg_text = interpret_rainfall(monthly, "average", "mm")
    assert "Jan (12.0)" in total_text and "Feb (10.0)" in total_text
    assert "Feb (10.0)" in avg_text and "Jan (6.0)" in avg_text     # wettest is now Feb
    assert "Feb accounts for 62%" in avg_text                       # 10 / (6 + 10)


def test_missing_rainfall_is_never_counted_as_zero():
    raw = known_raw()
    raw.loc[4, "Rainfall (mm)"] = np.nan                          # February has no reading
    table = monthly_rainfall(preprocess(raw).clean)
    assert table["month"].tolist() == [1]                         # Feb absent, not 0


# ------------------------------------------------------------------ statistics
def test_temperature_statistics(clean):
    stats = temperature_stats(clean)
    values = [10, 20, 30, 40, 5]
    assert stats["count"] == 5
    assert stats["mean"] == pytest.approx(statistics.mean(values))        # 21
    assert stats["median"] == statistics.median(values)                   # 20
    assert stats["std"] == pytest.approx(statistics.stdev(values))        # sample SD 14.3178
    assert (stats["min"], stats["max"]) == (5, 40)
    assert stats["min_date"] == pd.Timestamp("2025-02-01")
    assert stats["max_date"] == pd.Timestamp("2025-01-01")
    table = stats_table(stats).set_index("statistic")["value"]
    assert table["Mean (°C)"] == "21.00" and table["Standard deviation (°C)"] == "14.32"


def test_statistics_ignore_missing_values():
    raw = known_raw()
    raw.loc[3, "Temperature"] = np.nan
    stats = temperature_stats(preprocess(raw).clean)
    assert stats["count"] == 4 and stats["mean"] == pytest.approx(statistics.mean([10, 20, 30, 5]))


# ------------------------------------------------------------------ correlation
def test_correlation_matches_the_standard_library():
    temp = [18, 21, 25, 27, 30, 31, 24, 22, 19, 33]
    hum = [80, 74, 66, 61, 52, 55, 70, 72, 79, 45]
    raw = pd.DataFrame({"Date": pd.date_range("2024-03-01", periods=10).strftime("%Y-%m-%d"),
                        "Temperature": temp, "Humidity": hum})
    result = temperature_humidity_correlation(preprocess(raw).clean)
    assert result["available"] and result["n"] == 10
    assert result["r"] == pytest.approx(statistics.correlation(temp, hum))
    assert result["direction"] == "negative" and result["strength"] == "very strong"


def test_correlation_refuses_too_few_pairs(clean):
    result = temperature_humidity_correlation(clean)             # only 5 pairs
    assert result["available"] is False and "at least 10" in result["reason"]


def test_correlation_uses_only_complete_pairs():
    temp = list(range(1, 13))
    hum = [2 * t for t in temp]
    raw = pd.DataFrame({"Temperature": temp, "Humidity": hum})
    raw.loc[0, "Humidity"] = np.nan
    result = temperature_humidity_correlation(preprocess(raw).clean)
    assert result["n"] == 11 and result["r"] == pytest.approx(1.0)


# ------------------------------------------------------------------ errors and empty data
def test_missing_required_columns_give_a_helpful_message():
    with pytest.raises(DataLoadError, match="Columns found: foo, bar"):
        preprocess(pd.DataFrame({"foo": [1, 2], "bar": [3, 4]}))


def test_missing_date_column_still_works_but_time_charts_are_unavailable():
    clean = preprocess(pd.DataFrame({"Temperature": [20, 21, 22]})).clean
    availability = chart_availability(clean)
    assert availability["temperature_over_time"][0] is False
    assert "date" in availability["temperature_over_time"][1]
    assert plot_temperature_over_time(clean) is None


@pytest.mark.parametrize("content", [b"", b"Date,Temperature\n"])
def test_empty_files_are_rejected(content):
    with pytest.raises(DataLoadError):
        load_csv(content)


def test_binary_file_gets_a_clear_message():
    with pytest.raises(DataLoadError, match="binary"):
        load_csv(b"PK\x03\x04\x00\x00\x00binary\x00\xff\xfe")


def test_no_valid_records_after_date_filter(clean):
    empty = filter_by_date(clean, "2030-01-01", "2030-12-31")
    assert len(empty) == 0
    assert temperature_stats(empty)["count"] == 0
    assert monthly_temperature(empty).empty and monthly_rainfall(empty).empty
    for plot in (plot_temperature_over_time, plot_monthly_temperature, plot_monthly_rainfall,
                 plot_temperature_vs_humidity, plot_temperature_distribution):
        assert plot(empty) is None


def test_all_temperatures_missing_is_handled():
    raw = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"], "Temperature": [np.nan, np.nan],
                        "Rainfall (mm)": [1, 2]})
    clean = preprocess(raw).clean
    assert temperature_stats(clean)["count"] == 0
    assert plot_temperature_distribution(clean) is None


# ------------------------------------------------------------------ exports
def test_cleaned_csv_export_round_trips_with_correct_values(clean):
    back = pd.read_csv(io.BytesIO(to_csv_bytes(clean)))
    assert back["temperature"].tolist() == [10, 20, 30, 40, 5]
    assert back["rainfall"].tolist() == [2, 4, 0, 6, 10]
    assert back["month"].tolist() == [1, 1, 1, 1, 2]


def test_summary_table_exports_keep_their_numbers(clean):
    monthly = pd.read_csv(io.BytesIO(to_csv_bytes(monthly_temperature(clean))))
    assert monthly.loc[monthly["month"] == 1, "avg_temperature"].iloc[0] == 25
    rain = pd.read_csv(io.BytesIO(to_csv_bytes(monthly_rainfall(clean))))
    assert rain.loc[rain["month"] == 1, "total_rainfall"].iloc[0] == 12


def test_original_data_is_not_modified_by_cleaning():
    raw = known_raw()
    before = raw.copy()
    preprocess(raw)
    pd.testing.assert_frame_equal(raw, before)
