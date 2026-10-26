"""Tests that charts are built from the real data, and that the demo dataset works end to end."""
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from analysis import chart_availability, monthly_rainfall, temperature_stats
from data_processing import load_csv, preprocess
from demo_data import demo_csv_bytes, generate_demo_dataframe
from tests.test_data_processing import small_raw
from visualizations import (plot_monthly_rainfall, plot_monthly_temperature,
                            plot_temperature_distribution, plot_temperature_over_time,
                            plot_temperature_vs_humidity)


@pytest.fixture
def clean():
    return preprocess(small_raw()).clean


def test_line_chart_uses_every_valid_reading_in_date_order(clean):
    fig = plot_temperature_over_time(clean)
    assert isinstance(fig, Figure)
    line = fig.axes[0].lines[0]
    ydata = list(line.get_ydata())
    # The 29-day gap between 3 Jan and 1 Feb is drawn as a break (NaN), not a joined line,
    # so only the real readings are non-NaN and they stay in date order.
    assert [y for y in ydata if not np.isnan(y)] == [20, 22, 24, 30, 32]
    assert sum(np.isnan(y) for y in ydata) == 1
    assert fig.axes[0].get_ylabel() == "Temperature (°C)"


def test_monthly_bars_match_calculated_averages(clean):
    ax = plot_monthly_temperature(clean).axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Jan", "Feb"]
    assert [round(p.get_height(), 2) for p in ax.patches] == [22.0, 31.0]


def test_rainfall_chart_labels_unit_honestly(clean):
    unknown = plot_monthly_rainfall(clean, "unknown").axes[0]
    confirmed = plot_monthly_rainfall(clean, "mm").axes[0]
    assert "unit not confirmed" in unknown.get_ylabel()
    assert "(mm)" in confirmed.get_ylabel() and "Total" in confirmed.get_ylabel()
    assert [p.get_height() for p in confirmed.patches] == [15, 20]


def test_scatter_and_histogram_are_created(clean):
    assert isinstance(plot_temperature_vs_humidity(clean), Figure)
    hist = plot_temperature_distribution(clean)
    assert sum(p.get_height() for p in hist.axes[0].patches) == 5      # 5 readings in bins


def test_charts_return_none_instead_of_crashing():
    empty = pd.DataFrame()
    only_temp = pd.DataFrame({"temperature": [20.0, 21.0, 22.0]})
    assert plot_temperature_over_time(empty) is None
    assert plot_monthly_temperature(only_temp) is None          # no date -> no months
    assert plot_monthly_rainfall(only_temp) is None
    assert plot_temperature_vs_humidity(only_temp) is None
    assert isinstance(plot_temperature_distribution(only_temp), Figure)
    assert plot_temperature_distribution(pd.DataFrame({"temperature": [np.nan]})) is None


def test_demo_dataset_is_labelled_and_exercises_cleaning():
    raw = load_csv(demo_csv_bytes())
    result = preprocess(raw)
    assert result.report["duplicates_found"] == 2
    assert result.report["invalid_numeric"]["rainfall"] == 1
    assert result.report["implausible"]["humidity"] == 1
    assert result.report["implausible"]["temperature"] == 1
    assert result.report["outliers_flagged"] >= 1
    assert result.report["min_above_max"] == 1
    assert len(result.clean) == len(raw) - 2


def test_demo_dataset_supports_all_five_charts():
    clean = preprocess(load_csv(demo_csv_bytes())).clean
    assert all(ok for ok, _ in chart_availability(clean).values())
    for plot in (plot_temperature_over_time, plot_monthly_temperature,
                 plot_temperature_vs_humidity, plot_temperature_distribution):
        assert plot(clean) is not None
    assert len(monthly_rainfall(clean)) == 12
    assert temperature_stats(clean)["count"] > 700


def test_demo_generator_is_reproducible():
    pd.testing.assert_frame_equal(generate_demo_dataframe(), generate_demo_dataframe())
