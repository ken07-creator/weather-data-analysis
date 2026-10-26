"""
app.py
======
Streamlit dashboard for the "Weather and Temperature Analysis" project.

Run with:   streamlit run app.py

This file only builds the user interface. All calculations live in:
    data_processing.py  (load / validate / clean)
    analysis.py         (statistics and interpretations)
    visualizations.py   (the five charts)
"""

import streamlit as st

from analysis import (build_limitations, chart_availability, CHART_TITLES,
                      generate_interpretations, monthly_rainfall, monthly_temperature,
                      rainfall_unit_label, stats_table, temperature_humidity_correlation,
                      temperature_stats)
from data_processing import (cleaned_export_frame, DataLoadError, RAINFALL_UNITS, ROLE_LABELS, ROLES, column_info,
                             detect_columns, filter_by_date, guess_rainfall_unit,
                             guess_temperature_unit, load_csv, missing_value_table,
                             preprocess, to_csv_bytes)
from demo_data import DEMO_FILE_NAME, demo_csv_bytes
from visualizations import (plot_monthly_rainfall, plot_monthly_temperature,
                            plot_temperature_distribution, plot_temperature_over_time,
                            plot_temperature_vs_humidity)

st.set_page_config(page_title="Weather & Temperature Analysis", page_icon="🌦️", layout="wide")

UPLOAD_CHOICE = "Upload a CSV file"
DEMO_CHOICE = "Use demo dataset (SYNTHETIC)"


# --------------------------------------------------------------------------- #
# Sidebar: choose the data and the cleaning options
# --------------------------------------------------------------------------- #
def choose_dataset():
    """Return (raw DataFrame or None, dataset name, is_demo)."""
    st.sidebar.header("1. Data")
    choice = st.sidebar.radio("Data source", [UPLOAD_CHOICE, DEMO_CHOICE])
    if choice == DEMO_CHOICE:
        st.sidebar.warning("Demo data are computer-generated, NOT real observations.")
        return load_csv(demo_csv_bytes()), DEMO_FILE_NAME, True
    uploaded = st.sidebar.file_uploader("Weather data (CSV)", type=["csv"])
    if uploaded is None:
        return None, None, False
    try:
        return load_csv(uploaded.getvalue()), uploaded.name, False
    except DataLoadError as error:
        st.sidebar.error(str(error))
        return None, uploaded.name, False


def choose_options(raw, name):
    """Column mapping + cleaning options. Returns (column_map, options dict)."""
    detected = detect_columns(raw.columns)
    st.sidebar.header("2. Columns and cleaning")
    with st.sidebar.expander("Column mapping (auto-detected)", expanded=False):
        st.caption("Check that each role points at the right column.")
        none_label = "(none)"
        choices = [none_label] + list(raw.columns)
        column_map = {}
        for role in ROLES:
            default = detected.get(role)
            picked = st.selectbox(ROLE_LABELS[role], choices,
                                  index=choices.index(default) if default in choices else 0,
                                  key=f"map_{role}_{name}")
            if picked != none_label:
                column_map[role] = picked

    with st.sidebar.expander("Cleaning options", expanded=False):
        temp_guess = guess_temperature_unit(column_map.get("temperature"))
        temperature_unit = st.selectbox("Temperature unit in file", ["C", "F"],
                                        index=0 if temp_guess == "C" else 1,
                                        key=f"tunit_{name}",
                                        help="Fahrenheit values are converted to Celsius.")
        rain_guess = guess_rainfall_unit(column_map.get("rainfall"))
        rainfall_unit = st.selectbox("Rainfall unit in file", RAINFALL_UNITS,
                                     index=RAINFALL_UNITS.index(rain_guess),
                                     key=f"runit_{name}",
                                     help="Used only as a label; values are not converted.")
        dayfirst = st.checkbox("Dates are day-first (31/01/2024)", value=False,
                               key=f"dayfirst_{name}")
        remove_duplicates = st.checkbox("Remove exact duplicate rows", value=True,
                                        key=f"dups_{name}")
    return column_map, dict(temperature_unit=temperature_unit, rainfall_unit=rainfall_unit,
                            dayfirst=dayfirst, remove_duplicates=remove_duplicates)


def date_filter(clean):
    """Optional date-range filter in the sidebar."""
    if "date" not in clean.columns or clean["date"].notna().sum() == 0:
        return clean
    st.sidebar.header("3. Filter")
    first, last = clean["date"].min().date(), clean["date"].max().date()
    chosen = st.sidebar.date_input("Date range", value=(first, last),
                                   min_value=first, max_value=last)
    if isinstance(chosen, tuple) and len(chosen) == 2 and (chosen[0], chosen[1]) != (first, last):
        return filter_by_date(clean, chosen[0], chosen[1])
    return clean


def show_figure(figure):
    st.pyplot(figure)


def fmt(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.title("🌦️ Weather and Temperature Analysis")
st.caption("B.Tech III Semester academic project: Python · Pandas · NumPy · Matplotlib · "
           "Streamlit")

raw, dataset_name, is_demo = choose_dataset()

result = None
filtered = None
if raw is not None:
    column_map, options = choose_options(raw, dataset_name)
    try:
        result = preprocess(raw, column_map, **options)
    except DataLoadError as error:
        st.error(f"The dataset cannot be analysed: {error}")
    if result is not None:
        filtered = date_filter(result.clean)

if is_demo:
    st.warning("You are viewing SYNTHETIC demonstration data generated by the program. "
               "They are not real weather observations.")

tabs = st.tabs(["1 Overview", "2 Dataset Explorer", "3 Data Cleaning", "4 Statistics",
                "5 Visualizations", "6 Insights", "7 Downloads"])
needs_data = "Upload a CSV file (or choose the demo dataset) in the sidebar to see this section."

# ---- 1. Overview ---------------------------------------------------------- #
with tabs[0]:
    st.header("Project Overview")
    st.write(
        "This application loads a weather dataset, checks and cleans it, calculates "
        "statistics, and draws five charts that help you understand temperature, rainfall "
        "and humidity patterns. Every number and chart is computed from the file you provide.")
    st.markdown("**Workflow:** Data Upload → Validation → Preprocessing → Exploratory Data "
                "Analysis → Statistical Analysis → Visualization → Interpretation → "
                "Export Results")
    st.markdown(
        "**Expected CSV columns** (names are detected automatically; not all are required):\n"
        "- `Date`, `Temperature`, `Minimum Temperature`, `Maximum Temperature`\n"
        "- `Rainfall`, `Humidity` (percent), `Wind Speed`")
    if raw is None:
        st.info(needs_data)

# ---- 2. Dataset Explorer ------------------------------------------------- #
with tabs[1]:
    st.header("Dataset Explorer")
    if raw is None:
        st.info(needs_data)
    else:
        st.write(f"**File:** {dataset_name}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Records", f"{len(raw):,}")
        c2.metric("Columns", raw.shape[1])
        c3.metric("Duplicate rows", int(raw.duplicated().sum()))
        st.subheader("First rows")
        st.dataframe(raw.head(10))
        st.subheader("Columns and data types (as read from the file)")
        st.dataframe(column_info(raw))
        if result is not None:
            st.subheader("How columns were interpreted")
            st.dataframe({"Role": [ROLE_LABELS[r] for r in result.column_map],
                          "Column in file": list(result.column_map.values())})

# ---- 3. Data Cleaning --------------------------------------------------- #
with tabs[2]:
    st.header("Data Cleaning")
    if result is None:
        st.info(needs_data)
    else:
        st.subheader("Before-and-after summary")
        st.dataframe(result.summary)
        st.subheader("What was done")
        for line in result.log:
            st.markdown(f"- {line}")
        for warning in result.warnings:
            st.warning(warning)
        st.subheader("Missing values in the cleaned data")
        st.dataframe(missing_value_table(result.clean))
        st.subheader("Suspicious or invalid values")
        if result.flagged.empty:
            st.success("No invalid or suspicious values were found.")
        else:
            st.caption("'source_row' is the row number in your CSV file (header = row 1). "
                       "Impossible values are set to missing, never replaced with guesses. "
                       "Outliers are kept because unusual weather can be real.")
            st.dataframe(result.flagged)
        st.subheader("Original data vs cleaned data")
        st.caption("The original upload is never modified. 'source_row' in the cleaned data "
                   "is the row number in your CSV file, so you can find any row in the original.")
        left_col, right_col = st.columns(2)
        with left_col:
            st.markdown(f"**Original (as uploaded): {len(raw):,} rows**")
            st.dataframe(raw.head(15))
        with right_col:
            st.markdown(f"**Cleaned: {len(result.clean):,} rows**")
            st.dataframe(result.clean.head(15))

# ---- 4. Statistics -------------------------------------------------------- #
with tabs[3]:
    st.header("Statistical Summary")
    if filtered is None:
        st.info(needs_data)
    else:
        stats = temperature_stats(filtered)
        if stats["count"] == 0:
            st.warning("No valid temperature values are available, so temperature "
                       "statistics cannot be calculated.")
        else:
            cols = st.columns(6)
            cols[0].metric("Observations", stats["count"])
            cols[1].metric("Mean (°C)", fmt(stats["mean"]))
            cols[2].metric("Median (°C)", fmt(stats["median"]))
            cols[3].metric("Min (°C)", fmt(stats["min"]))
            cols[4].metric("Max (°C)", fmt(stats["max"]))
            cols[5].metric("Std. dev. (°C)", fmt(stats["std"]))
            with st.expander("What do these numbers mean?"):
                st.markdown(
                    "- **Mean**: the average of all valid temperature readings.\n"
                    "- **Median**: the middle reading when sorted. If it is far from the "
                    "mean, a few very high or low readings are pulling the mean.\n"
                    "- **Standard deviation**: the typical distance of readings from the "
                    "mean. Larger means more variable temperatures.\n"
                    "- **Observations**: only rows with a valid temperature are counted.")
            st.dataframe(stats_table(stats))
        left, right = st.columns(2)
        with left:
            st.subheader("Monthly average temperature")
            monthly_t = monthly_temperature(filtered)
            if monthly_t.empty:
                st.info("Not available: needs valid dates and temperatures.")
            else:
                st.caption("avg_temperature = mean of ALL readings in that calendar month. "
                           "avg_of_yearly_means = mean of each year's monthly mean "
                           "(equal when only one year is present).")
                st.dataframe(monthly_t.round(2))
        with right:
            st.subheader(f"Monthly rainfall ({rainfall_unit_label(result.units['rainfall'])})")
            monthly_r = monthly_rainfall(filtered)
            if monthly_r.empty:
                st.info("Not available: needs valid dates and rainfall values.")
            else:
                st.caption("total_rainfall = SUM of readings; avg_monthly_total_per_year = "
                           "total ÷ years covered. These are totals, not averages of daily values.")
                st.dataframe(monthly_r.round(2))
        st.subheader("Temperature-humidity correlation")
        corr = temperature_humidity_correlation(filtered)
        if corr["available"]:
            st.metric("Pearson r", f"{corr['r']:.3f}",
                      help=f"{corr['strength']} {corr['direction']}; n = {corr['n']}")
            st.caption(f"{corr['strength'].capitalize()} {corr['direction']} relationship "
                       f"from {corr['n']} paired readings. Correlation does not prove causation.")
        else:
            st.info(corr["reason"])
        st.subheader("Data quality counts (whole dataset, before date filter)")
        q1, q2, q3 = st.columns(3)
        q1.metric("Duplicate rows found", result.report["duplicates_found"])
        q2.metric("Unreadable dates", result.report["invalid_dates"])
        q3.metric("Values flagged", len(result.flagged))

# ---- 5. Visualizations ---------------------------------------------------------- #
with tabs[4]:
    st.header("Visualizations")
    if filtered is None:
        st.info(needs_data)
    else:
        availability = chart_availability(filtered)
        rain_mode = "total"
        if availability["monthly_rainfall"][0] and filtered["year"].nunique() > 1:
            label = st.radio("Rainfall bars show",
                             ["Total rainfall (all years added together)",
                              "Average monthly total per year"], horizontal=True)
            rain_mode = "total" if label.startswith("Total") else "average"
        notes = generate_interpretations(filtered, result.units["rainfall"], rain_mode)
        rain_unit = result.units["rainfall"]
        builders = {
            "temperature_over_time": lambda: plot_temperature_over_time(filtered),
            "monthly_temperature": lambda: plot_monthly_temperature(filtered),
            "monthly_rainfall": lambda: plot_monthly_rainfall(filtered, rain_unit, rain_mode),
            "temperature_vs_humidity": lambda: plot_temperature_vs_humidity(filtered),
            "temperature_distribution": lambda: plot_temperature_distribution(filtered),
        }
        for number, key in enumerate(builders, start=1):
            st.subheader(f"Visualization {number}: {CHART_TITLES[key]}")
            available, reason = availability[key]
            figure = builders[key]() if available else None
            if figure is None:
                st.warning(reason or "This chart cannot be generated from the selected data.")
                continue
            show_figure(figure)
            if notes.get(key):
                st.info(f"**Interpretation:** {notes[key]}")

# ---- 6. Insights ----------------------------------------------------------------- #
with tabs[5]:
    st.header("Insights and Observations")
    if filtered is None:
        st.info(needs_data)
    else:
        st.subheader("What the data show")
        shown = False
        for key, text in generate_interpretations(filtered, result.units["rainfall"]).items():
            if text:
                shown = True
                st.markdown(f"- **{CHART_TITLES[key]}:** {text}")
        if not shown:
            st.info("There is not enough valid data to produce observations.")
        st.subheader("Limitations")
        for note in build_limitations(result.report, result.units, len(filtered),
                                      result.derived_temperature, is_demo):
            st.markdown(f"- {note}")

# ---- 7. Downloads -------------------------------------------------------------------- #
with tabs[6]:
    st.header("Download Results")
    if filtered is None:
        st.info(needs_data)
    else:
        st.caption("Downloads reflect the current date-range filter, except the original "
                   "upload, the flagged values and the cleaning summary, which cover the "
                   "whole file.")
        if len(filtered) == 0:
            st.warning("No records fall inside the selected date range, so the tables below "
                       "are empty. Widen the date range in the sidebar.")
        stats_df = stats_table(temperature_stats(filtered))
        files = {
            "original_uploaded_data.csv": raw,
            "cleaned_weather_data.csv": cleaned_export_frame(filtered, result.units),
            "temperature_statistics.csv": stats_df,
            "monthly_temperature.csv": monthly_temperature(filtered),
            "monthly_rainfall.csv": monthly_rainfall(filtered),
            "flagged_values.csv": result.flagged,
            "cleaning_summary.csv": result.summary,
        }
        for file_name, table in files.items():
            if table is None or table.empty:
                st.button(f"{file_name} (not available for this data)", disabled=True,
                          key=f"disabled_{file_name}")
            else:
                st.download_button(f"Download {file_name}", to_csv_bytes(table), file_name,
                                   mime="text/csv", key=f"dl_{file_name}")
