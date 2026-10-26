"""
Tests that drive the real Tkinter window (tkinter_app.py).

They are skipped automatically when Tk or a display is not available. On a Linux machine
without a screen, run them with:   xvfb-run -a python -m pytest -q
"""
import os
import sys

import pandas as pd
import pytest

tk = pytest.importorskip("tkinter")

from analysis import CHART_TITLES                    # noqa: E402
from data_processing import preprocess               # noqa: E402
from tests.test_data_processing import small_raw     # noqa: E402


@pytest.fixture
def app():
    import tkinter_app
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available for Tkinter")
    root.withdraw()
    window = tkinter_app.WeatherApp(root)
    window.messages = []                              # (kind, text) instead of pop-ups
    window.show_error = lambda text: window.messages.append(("error", text))
    window.show_warning = lambda text: window.messages.append(("warning", text))
    window.show_info = lambda text: window.messages.append(("info", text))
    yield window
    root.destroy()


def write_csv(tmp_path, text, name="data.csv"):
    path = tmp_path / name
    path.write_text(text)
    return path


def choose(app, key):
    app.chart_choice.set(CHART_TITLES[key])


# ---------------- startup and "no dataset" ----------------
def test_window_has_required_controls(app):
    assert app.root.title() == "Weather and Temperature Analysis"
    texts = set()
    def walk(widget):
        for child in widget.winfo_children():
            try:
                texts.add(child.cget("text"))
            except tk.TclError:
                pass
            walk(child)
    walk(app.root)
    for label in ["Open CSV", "Load Demo Dataset", "Dataset Summary", "Generate Chart",
                  "Cleaning Report", "Export Cleaned CSV"]:
        assert label in texts
    assert list(app.chart_box["values"]) == list(CHART_TITLES.values())


def test_actions_without_dataset_show_message_and_do_not_crash(app):
    app.generate_chart()
    app.export_cleaned()
    assert app.show_cleaning_report() is None
    assert [kind for kind, _ in app.messages] == ["info"] * 3
    assert all("No dataset has been selected" in text for _, text in app.messages)
    assert app.canvas is None


# ---------------- demo dataset and all five charts ----------------
def test_load_demo_fills_summary_and_shows_synthetic_banner(app):
    assert app.load_demo() is True
    assert app.summary_vars["Rows"].get().startswith("733")
    assert "2023-01-01 to 2024-12-31" == app.summary_vars["Date range"].get()
    assert app.summary_vars["Duplicate records"].get() == "2"
    assert "NOT real weather" in app.banner_var.get()
    assert app.messages == []


def test_all_five_charts_are_embedded_in_the_window(app):
    app.load_demo()
    for key, title in CHART_TITLES.items():
        app.chart_choice.set(title)
        app.generate_chart()
        assert app.canvas is not None, f"{title} was not displayed"
        figure = app.canvas.figure
        assert figure.axes and figure.axes[0].get_xlabel()
        assert app.canvas.get_tk_widget().winfo_exists()
        assert app.interpretation_var.get().startswith("Interpretation:")
        assert len(app.chart_holder.winfo_children()) == 1       # old chart replaced, not stacked
    assert app.messages == []


def test_chart_data_comes_from_the_loaded_dataset(app, tmp_path):
    path = write_csv(tmp_path, "Date,Temperature,Rainfall (mm)\n"
                     "2024-01-01,20,0\n2024-01-02,22,10\n2024-01-03,24,5\n"
                     "2024-02-01,30,0\n2024-02-02,32,20\n")
    assert app.load_path(path)
    choose(app, "monthly_temperature")
    app.generate_chart()
    assert [round(p.get_height(), 2) for p in app.canvas.figure.axes[0].patches] == [22.0, 31.0]
    choose(app, "monthly_rainfall")
    app.generate_chart()
    assert [p.get_height() for p in app.canvas.figure.axes[0].patches] == [15, 20]
    assert "(mm)" in app.canvas.figure.axes[0].get_ylabel()


# ---------------- CSV upload ----------------
def test_open_csv_uses_file_dialog_and_loads_file(app, tmp_path):
    path = write_csv(tmp_path, "Date,Temperature,Humidity\n2024-01-01,20,60\n"
                     "2024-01-02,22,70\n2024-01-02,22,70\n")
    app.ask_open_path = lambda: str(path)
    app.open_csv()
    assert app.dataset_name == "data.csv" and app.is_demo is False
    assert app.summary_vars["Rows"].get().startswith("3")
    assert app.summary_vars["Duplicate records"].get() == "1"
    assert app.banner_var.get() == ""


def test_cancelling_the_open_dialog_changes_nothing(app):
    app.ask_open_path = lambda: ""
    app.open_csv()
    assert app.result is None and app.messages == []


def test_unreadable_files_give_errors_and_keep_previous_dataset(app, tmp_path):
    app.load_demo()
    previous = app.result
    assert app.load_path(write_csv(tmp_path, "", "empty.csv")) is False
    assert app.load_path(tmp_path / "nope.csv") is False
    binary = tmp_path / "fake.csv"
    binary.write_bytes(b"PK\x03\x04\x00\x00binary")
    assert app.load_path(binary) is False
    assert [kind for kind, _ in app.messages] == ["error"] * 3
    assert "could not be loaded" in app.messages[0][1]
    assert app.result is previous                                   # window still usable


def test_missing_required_columns_message(app, tmp_path):
    assert app.load_path(write_csv(tmp_path, "foo,bar\n1,2\n3,4\n")) is False
    kind, text = app.messages[-1]
    assert kind == "error" and "required column" in text and "foo" in text


def test_chart_that_cannot_be_drawn_shows_reason_without_crashing(app, tmp_path):
    app.load_path(write_csv(tmp_path, "Date,Temperature\n2024-01-01,20\n2024-01-02,22\n"))
    choose(app, "monthly_rainfall")
    app.generate_chart()
    kind, text = app.messages[-1]
    assert kind == "warning" and "cannot be shown" in text and "rainfall" in text
    assert app.canvas is None
    choose(app, "temperature_over_time")                           # other charts still work
    app.generate_chart()
    assert app.canvas is not None


def test_dataset_with_no_valid_records(app, tmp_path):
    app.load_path(write_csv(tmp_path, "Date,Temperature\n2024-01-01,abc\n2024-01-02,xyz\n"))
    assert app.messages[-1][0] == "warning" and "no valid records" in app.messages[-1][1]
    app.generate_chart()
    app.export_cleaned()
    assert [k for k, _ in app.messages[-2:]] == ["error", "error"]
    assert all("no valid records" in t for _, t in app.messages[-2:])
    assert app.canvas is None
    assert app.show_cleaning_report() is not None                  # report still explains why


def test_day_first_checkbox_reprocesses_dates(app, tmp_path):
    app.load_path(write_csv(tmp_path, "Date,Temperature\n02/03/2024,20\n03/03/2024,21\n"))
    assert app.summary_vars["Date range"].get().startswith("2024-02-03")
    app.dayfirst.set(True)
    app._reprocess()
    assert app.summary_vars["Date range"].get().startswith("2024-03-02")


# ---------------- cleaning report ----------------
def test_cleaning_report_window_shows_report_text(app):
    app.load_demo()
    window = app.show_cleaning_report()
    text_widget = next(c for c in window.winfo_children() if isinstance(c, tk.Text))
    content = text_widget.get("1.0", "end")
    assert "CLEANING REPORT" in content and "Removed 2 exact duplicate row(s)" in content
    assert str(text_widget.cget("state")) == "disabled"
    app.show_cleaning_report()                                     # reopening replaces the window
    assert not window.winfo_exists()


# ---------------- export ----------------
def test_export_cleaned_csv_writes_the_cleaned_data(app, tmp_path):
    app.load_demo()
    target = tmp_path / "cleaned.csv"
    app.ask_save_path = lambda: str(target)
    app.export_cleaned()
    exported = pd.read_csv(target)
    assert len(exported) == len(app.result.clean) == 731            # 733 rows - 2 duplicates
    assert {"date", "temperature_c", "humidity_percent", "rainfall_mm", "month"} <= set(exported.columns)
    assert app.messages[-1][0] == "info" and "731" in app.messages[-1][1]


def test_export_failure_and_cancel(app, tmp_path):
    app.load_demo()
    app.ask_save_path = lambda: str(tmp_path / "no_such_folder" / "x.csv")
    app.export_cleaned()
    kind, text = app.messages[-1]
    assert kind == "error" and "Export failed" in text
    before = len(app.messages)
    app.ask_save_path = lambda: ""
    app.export_cleaned()
    assert len(app.messages) == before


# ---------------- starting without a screen ----------------
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="DISPLAY is a Linux/X11 concept")
def test_main_exits_politely_without_a_display(monkeypatch, capsys):
    import tkinter_app
    monkeypatch.delenv("DISPLAY", raising=False)
    assert tkinter_app.main() == 1
    assert "Could not open a window" in capsys.readouterr().out


# =========================================================================== #
# Added in the final audit
# =========================================================================== #
import gc                       # noqa: E402
import subprocess               # noqa: E402
import time                     # noqa: E402

import numpy as np              # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_unit_lists_rescue_an_unlabelled_fahrenheit_file(app, tmp_path):
    path = write_csv(tmp_path, "Date,Temperature,Rainfall\n"
                     + "\n".join(f"2024-07-{d:02d},{t},1.0" for d, t in
                                 zip(range(1, 9), [68, 70, 75, 80, 85, 90, 95, 72])))
    app.load_path(path)
    assert app.result.clean["temperature"].isna().all()          # Celsius assumed: all > 60
    choose(app, "temperature_distribution")
    app.generate_chart()
    assert app.canvas is None and app.messages[-1][0] == "warning"
    window = app.show_cleaning_report()
    text = next(c for c in window.winfo_children() if isinstance(c, tk.Text)).get("1.0", "end")
    assert "Fahrenheit" in text                                  # the report explains why
    app.temperature_choice.set("Fahrenheit (°F)")
    app.rainfall_choice.set("mm")
    app._reprocess()
    assert app.result.clean["temperature"].iloc[0] == pytest.approx(20.0)     # (68-32)*5/9
    assert app.result.units["rainfall"] == "mm"
    assert "rainfall mm" in app.units_var.get()
    app.generate_chart()
    assert app.canvas is not None


def test_units_line_says_when_rainfall_unit_is_unconfirmed(app, tmp_path):
    app.load_path(write_csv(tmp_path, "Date,Temperature,Rainfall\n2024-01-01,20,1\n2024-01-02,21,2\n"))
    assert "unit not confirmed" in app.units_var.get()


def test_cleaning_report_window_is_closed_when_a_new_dataset_loads(app, tmp_path):
    app.load_demo()
    window = app.show_cleaning_report()
    assert window.winfo_exists()
    app.load_path(write_csv(tmp_path, "Date,Temperature\n2024-01-01,20\n2024-01-02,21\n"))
    assert not window.winfo_exists()


def test_failed_load_leaves_open_report_alone(app, tmp_path):
    app.load_demo()
    window = app.show_cleaning_report()
    app.load_path(write_csv(tmp_path, "", "empty.csv"))
    assert window.winfo_exists()


def test_repeated_charts_do_not_pile_up_figures_or_windows(app):
    from matplotlib.figure import Figure
    app.load_demo()
    for _ in range(4):
        for title in CHART_TITLES.values():
            app.chart_choice.set(title)
            app.generate_chart()
    app.root.update()
    gc.collect()
    assert len(app.chart_holder.winfo_children()) == 1
    assert sum(isinstance(o, Figure) for o in gc.get_objects()) <= 2   # the current chart only
    assert [w for w in app.root.winfo_children() if isinstance(w, tk.Toplevel)] == []


def test_all_dates_unreadable_keeps_the_window_usable(app, tmp_path):
    app.load_path(write_csv(tmp_path, "Date,Temperature,Humidity\nxx,20,50\nyy,22,55\nzz,24,60\n"))
    assert app.summary_vars["Date range"].get() == "No valid dates found"
    results = {}
    for key, title in CHART_TITLES.items():
        app.chart_choice.set(title)
        app.generate_chart()
        results[key] = app.canvas is not None
    assert results == {"temperature_over_time": False, "monthly_temperature": False,
                       "monthly_rainfall": False, "temperature_vs_humidity": True,
                       "temperature_distribution": True}


def test_rainfall_chart_in_window_shows_missing_data_caution(app, tmp_path):
    rows = ["Date,Temperature,Rainfall (mm)"]
    for i, day in enumerate(pd.date_range("2024-01-01", "2024-02-29")):
        rows.append(f"{day:%Y-%m-%d},25," + ("" if i < 20 else "10"))
    app.load_path(write_csv(tmp_path, "\n".join(rows)))
    choose(app, "monthly_rainfall")
    app.generate_chart()
    assert "CAUTION" in app.interpretation_var.get() and "Jan" in app.interpretation_var.get()
    ax = app.canvas.figure.axes[0]
    assert [bool(p.get_hatch()) for p in ax.patches] == [True, False]


def test_large_file_stays_usable(app, tmp_path):
    n = 60_000
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"Date": pd.date_range("2012-01-01", periods=n, freq="h").strftime("%Y-%m-%d %H:%M"),
                          "Temperature (C)": rng.normal(25, 5, n).round(1),
                          "Rainfall (mm)": np.where(rng.random(n) < .1, 2.0, 0.0),
                          "Humidity (%)": rng.uniform(30, 90, n).round(1)})
    path = tmp_path / "big.csv"
    frame.to_csv(path, index=False)
    start = time.time()
    assert app.load_path(path)
    for title in CHART_TITLES.values():
        app.chart_choice.set(title)
        app.generate_chart()
        assert app.canvas is not None
    assert app.summary_vars["Rows"].get().startswith("60,000")
    assert time.time() - start < 40
    target = tmp_path / "big_clean.csv"
    app.ask_save_path = lambda: str(target)
    app.export_cleaned()
    assert len(pd.read_csv(target)) == n


def test_exported_file_can_be_opened_again_with_same_units(app, tmp_path):
    app.load_demo()
    target = tmp_path / "round_trip.csv"
    app.ask_save_path = lambda: str(target)
    app.export_cleaned()
    first = app.result
    assert app.load_path(target)
    assert app.result.units["rainfall"] == "mm"
    assert len(app.result.clean) == len(first.clean)
    pd.testing.assert_series_equal(app.result.clean["temperature"], first.clean["temperature"])
    assert app.is_demo is False and app.banner_var.get() == ""    # a file you opened is not labelled demo


# ---- real process tests: closing the window, no pyplot, friendly import error ----
def test_window_closes_cleanly_and_never_uses_pyplot():
    script = (
        "import sys, tkinter as tk, tkinter_app\n"
        "root = tk.Tk(); app = tkinter_app.WeatherApp(root)\n"
        "app.load_demo(); app.generate_chart(); app.show_cleaning_report()\n"
        "root.after(400, root.destroy)\n"
        "root.mainloop()\n"
        "print('pyplot_loaded=%s' % ('matplotlib.pyplot' in sys.modules))\n"
        "print('closed_ok')\n")
    done = subprocess.run([sys.executable, "-c", script], cwd=PROJECT_ROOT, capture_output=True,
                          text=True, timeout=60)
    if "no display" in done.stderr or "couldn't connect to display" in done.stderr:
        pytest.skip("no display available for Tkinter")
    assert done.returncode == 0, done.stderr
    assert "closed_ok" in done.stdout and "pyplot_loaded=False" in done.stdout
    assert done.stderr.strip() == ""


def test_missing_tkinter_gives_a_friendly_message():
    script = ("import sys, runpy; sys.modules['tkinter'] = None\n"
              "runpy.run_path('tkinter_app.py', run_name='__main__')\n")
    done = subprocess.run([sys.executable, "-c", script], cwd=PROJECT_ROOT, capture_output=True,
                          text=True, timeout=60)
    assert done.returncode != 0
    assert "Tkinter is not installed" in done.stderr and "streamlit run app.py" in done.stderr
    assert "Traceback" not in done.stderr
