"""Tests for loading, cleaning and validation (data_processing.py)."""
import io

import numpy as np
import pandas as pd
import pytest

from data_processing import (DataLoadError, count_duplicates, detect_columns, filter_by_date,
                             guess_rainfall_unit, load_csv, missing_value_table, parse_dates,
                             preprocess, standardize_column_name, to_csv_bytes)


def small_raw():
    """Known dataset: Jan + Feb, last row duplicates the one before it."""
    return pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-02-01", "2024-02-02", "2024-02-02"],
        "Temperature": [20, 22, 24, 30, 32, 32],
        "Rainfall (mm)": [0, 10, 5, 0, 20, 20],
        "Humidity": [60, 70, 80, 50, 40, 40],
    })


# ---------------- loading ----------------
def test_load_csv_from_bytes_and_path(tmp_path):
    text = "Date,Temperature\n2024-01-01,20\n2024-01-02,21\n"
    from_bytes = load_csv(text.encode())
    path = tmp_path / "w.csv"
    path.write_text(text)
    from_path = load_csv(str(path))
    from_buffer = load_csv(io.BytesIO(text.encode()))
    for df in (from_bytes, from_path, from_buffer):
        assert list(df.columns) == ["Date", "Temperature"]
        assert len(df) == 2


def test_load_csv_semicolon_separator():
    df = load_csv(b"Date;Temperature\n2024-01-01;20\n2024-01-02;21\n")
    assert list(df.columns) == ["Date", "Temperature"]


@pytest.mark.parametrize("content", [b"", b"   \n  ", b"Date,Temperature\n"])
def test_load_csv_rejects_empty_files(content):
    with pytest.raises(DataLoadError):
        load_csv(content)


def test_load_csv_missing_file():
    with pytest.raises(DataLoadError):
        load_csv("does_not_exist.csv")


# ---------------- column detection ----------------
def test_standardize_and_detect_columns():
    assert standardize_column_name("Temperature (°C)") == "temperature_c"
    found = detect_columns(["Date", "Min Temp", "Max Temp", "Temp", "Precipitation",
                            "Relative Humidity", "Wind Speed", "Wind Direction"])
    assert found == {"date": "Date", "min_temperature": "Min Temp", "max_temperature": "Max Temp",
                     "temperature": "Temp", "rainfall": "Precipitation",
                     "humidity": "Relative Humidity", "wind_speed": "Wind Speed"}


def test_guess_rainfall_unit():
    assert guess_rainfall_unit("Rainfall (mm)") == "mm"
    assert guess_rainfall_unit("Rainfall") == "unknown"


# ---------------- dates ----------------
def test_date_conversion_valid_and_invalid():
    parsed = parse_dates(pd.Series(["2024-01-05", "not a date", None]))
    assert parsed.iloc[0] == pd.Timestamp("2024-01-05")
    assert parsed.isna().tolist() == [False, True, True]


def test_dayfirst_option_changes_interpretation():
    assert parse_dates(pd.Series(["02/03/2024"]), dayfirst=False).iloc[0] == pd.Timestamp("2024-02-03")
    assert parse_dates(pd.Series(["02/03/2024"]), dayfirst=True).iloc[0] == pd.Timestamp("2024-03-02")


def test_preprocess_counts_unreadable_dates_and_adds_month_year():
    raw = pd.DataFrame({"Date": ["2024-03-01", "garbage", "2024-04-01"], "Temperature": [20, 21, 22]})
    result = preprocess(raw)
    assert result.report["invalid_dates"] == 1
    clean = result.clean
    assert clean["month"].tolist()[:2] == [3, 4]          # valid dates first, NaT last
    assert clean["year"].iloc[0] == 2024
    assert pd.isna(clean["month"].iloc[2])
    assert len(clean) == 3                                # row kept, not invented or dropped


# ---------------- missing values and duplicates ----------------
def test_missing_value_detection():
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 3]})
    table = missing_value_table(df).set_index("column")
    assert table.loc["a", "missing"] == 1
    assert table.loc["b", "missing"] == 2
    assert table.loc["b", "percent_missing"] == pytest.approx(66.67, abs=0.01)


def test_duplicate_detection_and_removal():
    raw = small_raw()
    assert count_duplicates(raw) == 1
    result = preprocess(raw)
    assert result.report["duplicates_found"] == 1
    assert len(result.clean) == 5
    kept = preprocess(raw, remove_duplicates=False)
    assert len(kept.clean) == 6


def test_missing_temperature_is_not_filled_in():
    raw = small_raw()
    raw.loc[1, "Temperature"] = np.nan
    result = preprocess(raw)
    assert result.clean["temperature"].isna().sum() == 1


# ---------------- numeric conversion and validation ----------------
def test_non_numeric_text_becomes_missing_and_is_reported():
    raw = small_raw()
    raw["Rainfall (mm)"] = raw["Rainfall (mm)"].astype(object)
    raw.loc[1, "Rainfall (mm)"] = "error"
    result = preprocess(raw)
    assert result.report["invalid_numeric"]["rainfall"] == 1
    assert result.clean["rainfall"].isna().sum() == 1
    assert (result.flagged["issue"] == "not a number").sum() == 1


def test_impossible_humidity_set_to_missing_and_flagged():
    raw = small_raw()
    raw.loc[0, "Humidity"] = 140
    result = preprocess(raw)
    assert result.clean["humidity"].isna().sum() == 1
    flagged = result.flagged[result.flagged["column"] == "humidity"]
    assert len(flagged) == 1 and flagged["source_row"].iloc[0] == 2   # first data row = file row 2


def test_extreme_outlier_is_flagged_but_kept():
    temps = [25.0, 26, 24, 25, 27, 26, 25, 24, 26, 25, 55]
    raw = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=11).astype(str),
                        "Temperature": temps})
    result = preprocess(raw)
    assert 55 in result.clean["temperature"].tolist()
    assert result.report["outliers_flagged"] == 1


def test_fahrenheit_conversion():
    raw = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"], "Temperature": [86.0, 50.0]})
    result = preprocess(raw, temperature_unit="F")
    assert result.clean["temperature"].tolist() == pytest.approx([30.0, 10.0])


def test_impossible_temperature_after_conversion_is_set_to_missing():
    raw = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"], "Temperature": [212.0, 50.0]})
    result = preprocess(raw, temperature_unit="F")           # 212 F = 100 C, not weather
    assert pd.isna(result.clean["temperature"].iloc[0])
    assert result.report["implausible"]["temperature"] == 1


def test_temperature_derived_from_min_max_is_reported():
    raw = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"],
                        "Min Temperature": [10.0, 12.0], "Max Temperature": [20.0, None]})
    result = preprocess(raw)
    assert result.derived_temperature is True
    assert result.clean["temperature"].iloc[0] == 15.0
    assert pd.isna(result.clean["temperature"].iloc[1])           # not invented
    assert any("DERIVED" in w for w in result.warnings)


# ---------------- missing columns / empty datasets ----------------
def test_missing_date_column_is_handled():
    result = preprocess(pd.DataFrame({"Temperature": [20, 21, 22]}))
    assert "date" not in result.clean.columns
    assert any("No date column" in w for w in result.warnings)


def test_no_weather_columns_raises_helpful_error():
    with pytest.raises(DataLoadError, match="could be identified"):
        preprocess(pd.DataFrame({"foo": [1, 2], "bar": [3, 4]}))


@pytest.mark.parametrize("raw", [pd.DataFrame(), pd.DataFrame({"Temperature": []}), None])
def test_empty_dataset_raises_clear_error(raw):
    with pytest.raises(DataLoadError):
        preprocess(raw)


# ---------------- filtering / export ----------------
def test_filter_by_date_is_inclusive():
    clean = preprocess(small_raw()).clean
    jan = filter_by_date(clean, "2024-01-01", "2024-01-31")
    assert len(jan) == 3
    assert len(filter_by_date(clean, "2024-02-02", "2024-02-02")) == 1


def test_csv_download_contains_expected_data():
    clean = preprocess(small_raw()).clean
    again = pd.read_csv(io.BytesIO(to_csv_bytes(clean)))
    assert len(again) == len(clean)
    assert again["temperature"].tolist() == clean["temperature"].tolist()
