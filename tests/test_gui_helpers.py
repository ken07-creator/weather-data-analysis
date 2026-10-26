"""Tests for the reusable logic added for the desktop GUI (no window needed)."""
import numpy as np
import pandas as pd
import pytest

from analysis import CHART_TITLES
from data_processing import (DataLoadError, ExportError, cleaning_report_text, dataset_summary,
                             has_valid_records, load_csv, preprocess, save_csv, summary_lines)
from demo_data import demo_csv_bytes
from tests.test_data_processing import small_raw
from visualizations import CHART_BUILDERS, build_chart


# ---------------- dataset summary ----------------
def test_dataset_summary_known_values():
    raw = small_raw()                                  # 6 rows, 1 duplicate, no blanks
    raw.loc[1, "Humidity"] = np.nan
    summary = dataset_summary(raw, preprocess(raw))
    assert summary["rows"] == 6 and summary["n_columns"] == 4
    assert summary["duplicates"] == 1
    assert summary["rows_after_cleaning"] == 5
    assert summary["missing_total"] == 1 and summary["missing_by_column"] == {"Humidity": 1}
    start, end = summary["date_range"]
    assert (str(start.date()), str(end.date())) == ("2024-01-01", "2024-02-02")
    assert summary["valid_records"] is True


def test_summary_lines_text():
    raw = small_raw()
    raw.loc[1, "Humidity"] = np.nan
    lines = dict(summary_lines(dataset_summary(raw, preprocess(raw))))
    assert lines["Rows"].startswith("6") and "after cleaning: 5" in lines["Rows"]
    assert lines["Date range"] == "2024-01-01 to 2024-02-02"
    assert "Humidity (1)" in lines["Missing values"]
    assert lines["Duplicate records"] == "1"


def test_summary_without_date_column_says_so():
    raw = pd.DataFrame({"Temperature": [20.0, 21.0]})
    lines = dict(summary_lines(dataset_summary(raw, preprocess(raw))))
    assert lines["Date range"] == "No valid dates found"
    assert lines["Missing values"] == "None"


# ---------------- valid records ----------------
def test_has_valid_records():
    good = preprocess(small_raw()).clean
    assert has_valid_records(good)
    bad = preprocess(pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"],
                                   "Temperature": ["abc", "def"]})).clean
    assert not has_valid_records(bad)
    assert not has_valid_records(pd.DataFrame())


# ---------------- cleaning report ----------------
def test_cleaning_report_text_contains_real_findings():
    result = preprocess(load_csv(demo_csv_bytes()))
    text = cleaning_report_text(result)
    assert "CLEANING REPORT" in text and "Before cleaning" in text
    assert "Removed 2 exact duplicate row(s)" in text
    assert "humidity" in text and "set to missing" in text
    assert f"Suspicious or invalid values: {len(result.flagged)}" in text


def test_cleaning_report_when_nothing_flagged():
    text = cleaning_report_text(preprocess(pd.DataFrame(
        {"Date": ["2024-01-01", "2024-01-02"], "Temperature": [20.0, 21.0]})))
    assert "None found." in text


# ---------------- export ----------------
def test_save_csv_round_trip(tmp_path):
    clean = preprocess(small_raw()).clean
    path = save_csv(clean, tmp_path / "out.csv")
    again = pd.read_csv(path)
    assert len(again) == len(clean) == 5
    assert again["temperature"].tolist() == clean["temperature"].tolist()


def test_save_csv_errors_are_friendly(tmp_path):
    with pytest.raises(ExportError, match="no cleaned data"):
        save_csv(pd.DataFrame(), tmp_path / "x.csv")
    with pytest.raises(ExportError, match="Could not save"):
        save_csv(preprocess(small_raw()).clean, tmp_path / "missing_folder" / "x.csv")


# ---------------- build_chart ----------------
def test_build_chart_covers_exactly_the_five_charts():
    assert set(CHART_BUILDERS) == set(CHART_TITLES) and len(CHART_BUILDERS) == 5


@pytest.mark.parametrize("key", list(CHART_TITLES))
def test_build_chart_draws_every_chart_from_demo_data(key):
    clean = preprocess(load_csv(demo_csv_bytes())).clean
    figure, message = build_chart(key, clean, "mm")
    assert figure is not None and message == ""
    assert len(figure.axes[0].patches) + len(figure.axes[0].lines) + len(
        figure.axes[0].collections) > 0


def test_build_chart_uses_the_data_it_is_given():
    clean = preprocess(small_raw()).clean                       # 5 rows, Jan + Feb
    figure, _ = build_chart("monthly_temperature", clean)
    assert [round(p.get_height(), 2) for p in figure.axes[0].patches] == [22.0, 31.0]
    figure, _ = build_chart("monthly_rainfall", clean, "mm")
    assert [p.get_height() for p in figure.axes[0].patches] == [15, 20]


def test_build_chart_explains_missing_columns_and_empty_data():
    temp_only = preprocess(pd.DataFrame({"Temperature": [20.0, 21.0, 22.0]})).clean
    figure, message = build_chart("monthly_rainfall", temp_only)
    assert figure is None and "rainfall" in message
    figure, message = build_chart("temperature_over_time", temp_only)
    assert figure is None and "date" in message
    assert build_chart("temperature_distribution", temp_only)[0] is not None
    assert build_chart("temperature_distribution", pd.DataFrame())[0] is None
    assert build_chart("temperature_distribution", None)[1] == "There is no data to plot."
    assert "Unknown chart" in build_chart("nope", temp_only)[1]
