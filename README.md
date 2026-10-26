# Weather and Temperature Analysis Using Data Analysis

A beginner-friendly Python project (B.Tech III Semester) that loads a weather CSV file, validates and
cleans it, calculates statistics, draws five charts, explains the results in plain English and lets
you download the outputs. It runs as an interactive Streamlit web app.

## 1. Problem statement
Raw weather records are messy (missing readings, duplicates, impossible values) and hard to
understand as rows of numbers. This project turns a weather CSV into a clean dataset, clear
statistics and visualizations that answer simple questions: how does temperature change over time,
which months are hottest or wettest, and how are temperature and humidity related?

## 2. Objectives
- Load a user-supplied CSV and adapt to its actual column names.
- Detect and report missing values, duplicates, unreadable dates and implausible values, without inventing data.
- Calculate temperature statistics, monthly averages, monthly rainfall totals and a temperature-humidity correlation.
- Present five visualizations, each with an interpretation computed from the data.
- Export the cleaned data and summary tables as CSV.

## 3. Scope
**Included:** CSV upload, optional demo dataset, cleaning report, statistics, five charts, date-range filter, CSV downloads, a Streamlit web dashboard (`app.py`), a Tkinter desktop window (`tkinter_app.py`), automated tests.
**Not included:** Excel upload, weather forecasting, machine learning, databases, live weather APIs.

## 4. Tools and technologies
Python 3.10+ (tested on 3.12), Pandas, NumPy, Matplotlib, Streamlit, Tkinter (part of Python's standard library, nothing to `pip install`), pytest (for the tests only).
(Seaborn and OpenPyXL are not needed: charts use plain Matplotlib and Excel files are not supported.)

## 5. Dataset format
A CSV file with a header row. Column names are detected automatically (case, spaces and units in
brackets do not matter) and can be corrected in the sidebar. No column is strictly required, but at
least one weather measurement column must be recognised.

| Role | Examples of recognised names |
|---|---|
| Date | Date, DateTime, Timestamp, obs_date |
| Temperature | Temperature, Temp, Temperature (C), tavg |
| Min / Max temperature | Min Temperature, tmin, Max Temp, tmax |
| Rainfall | Rainfall, Rainfall (mm), Precipitation, prcp |
| Humidity (percent) | Humidity, Relative Humidity, RH |
| Wind speed | Wind Speed, wspd |

If no temperature column exists but min and max do, temperature is **derived** as their midpoint
(clearly warned in the app). Semicolon- and tab-separated files are also read.

**Dates:** `2024-01-31`, `2024/01/31`, `31-01-2024` (tick *day-first*), `Jan 31, 2024`, `20240131`, and date-times are understood. Year-first dates are never read day-first. Plain numbers that are not 8-digit dates (such as Excel serial numbers) are reported as unreadable instead of being turned into wrong dates.

**Units:** read from the column name where possible (`Rainfall (mm)`, `Temperature (F)`). If the file does not say, rainfall is shown as "unit not confirmed" and temperature is assumed to be Celsius. If your temperatures are Fahrenheit but the name does not say so, the cleaning report warns you and you can pick Fahrenheit in the sidebar (Streamlit) or the *Temperature in file* list (desktop). **Decimal commas** (`20,5`) are not converted; the app warns you to save the file with a decimal point.

**Test dataset:** `data/standin_weather_2025_SYNTHETIC.csv` is a stand-in with the same shape as the
described test file (daily 2025 records, temperature / min / max / rainfall / humidity / wind, a few
missing cells and **3 duplicated rows**). If you have your own `weather_sample_synthetic_test.csv`,
upload it the same way (see section 10). Synthetic data are for software testing only.

**Demo data:** `data/demo_weather_SYNTHETIC.csv` (also selectable in the app) is computer-generated,
**not real weather**. It deliberately contains a few flaws (2 duplicates, missing values, one
non-numeric rainfall entry, impossible humidity 140 % and temperature 75 C, one outlier, one row
with min/max swapped) so the cleaning step has something to show. Regenerate it with `python demo_data.py`.

## 6. Installation
```bash
cd weather-temperature-analysis
python -m venv venv
# Windows:  venv\Scripts\activate        macOS/Linux:  source venv/bin/activate
pip install -r requirements.txt
```

## 7. Running
```bash
streamlit run app.py        # web dashboard, opens http://localhost:8501
python tkinter_app.py       # desktop window (Tkinter)
python -m pytest -q         # runs the automated tests
```

### Desktop GUI (Tkinter)
`python tkinter_app.py` opens a desktop window titled **Weather and Temperature Analysis**. It uses
the same loading, cleaning, statistics and chart code as the Streamlit dashboard, so both show the
same numbers.

| Control | What it does |
|---|---|
| **Open CSV** | Choose a weather CSV; it is validated and cleaned automatically. |
| **Load Demo Dataset** | Loads the built-in SYNTHETIC data (a red banner reminds you it is not real weather). |
| **Dataset Summary** | Rows, columns, date range, blank cells and duplicate rows of the loaded file. |
| **Visualization list + Generate Chart** | Draws one of the five charts inside the window, with a plain-English interpretation underneath. |
| **Cleaning Report** | Opens a scrollable report: before/after table, what was done, warnings, flagged values. |
| **Export Cleaned CSV** | Saves the cleaned table (same data as the dashboard's `cleaned_weather_data.csv`). |
| *Dates are day-first* | Tick for dates such as 31/01/2024; the file is re-processed immediately. |
| *Temperature in file* / *Rainfall unit* | Auto (from the column name), or choose Celsius/Fahrenheit and mm/cm/inches/Unknown. The file is re-processed immediately; a "Units used" line shows the result. |

Messages appear (and the app keeps running) if no dataset is selected, a file cannot be read, no weather
column is recognised, a chart needs a column the file lacks, there are no valid records, or saving fails.

Notes:
- Columns are auto-detected. The desktop window has unit lists but no column-mapping boxes; use the
  Streamlit dashboard if the automatic column detection is wrong.
- *Export Cleaned CSV* writes unit-carrying column names (`temperature_c`, `humidity_percent`,
  `rainfall_mm`, ...). The same names are used by the dashboard's `cleaned_weather_data.csv`, so an
  exported file opens again with the same units and values.
- Rainfall bars show totals pooled over all years in the file.
- **Linux:** Tkinter is sometimes packaged separately. If `import tkinter` fails, run
  `sudo apt install python3-tk`. Windows and macOS installers from python.org include it.
- **Tests:** the window tests need a screen. They are skipped automatically without one; on a Linux
  machine without a display run `xvfb-run -a python -m pytest -q` to include them.

## 8. Methodology
Data Upload → Validation → Preprocessing → Exploratory Data Analysis → Statistical Analysis → Visualization → Interpretation → Export Results

### Preprocessing steps (`data_processing.preprocess`)
1. **Detect columns** and map them to internal names (date, temperature, rainfall, ...).
2. **Count duplicates** (rows identical in every column, including the date) and remove them (option can be switched off). Without a date column nothing is removed, because identical rows could be genuine readings. A date that appears twice with *different* values is kept and warned about.
3. **Parse dates**; unreadable dates become missing (rows kept, but left out of time-based charts).
4. **Convert numeric columns**; text such as `error` becomes missing and is listed.
5. **Convert Fahrenheit to Celsius** when the column name says Fahrenheit or you select it (sidebar / *Temperature in file* list). If most temperatures fall outside the plausible range the report says so and suggests Fahrenheit.
6. **Plausibility checks:** humidity outside 0-100, negative rainfall or wind, and temperature outside -90 to 60 C are impossible, so they are set to *missing* and listed. Nothing is replaced by a guess.
7. **Flag, don't delete:** extreme temperature/humidity outliers (outside 3 x IQR) and rows with min > max are flagged and kept, because unusual weather can be real.
8. **Disclose missing rainfall:** missing rainfall is never counted as 0; the report counts the missing readings and warns that monthly totals leave them out.
9. **Add** year, month, month name, year-month; sort by date.
10. **Report** a before/after summary, a log of every action and a table of flagged values (with the row number in your CSV).

No scaling or encoding is applied because no implemented feature needs it.

## 9. The five visualizations
1. **Temperature over time** (line): readings sorted by date; days with no reading appear as a gap, not a joined-up line.
2. **Average temperature by month** (bar): mean of all readings in each *calendar* month; years are pooled. The table also shows `avg_of_yearly_means` (average of each year's monthly mean), which differs when years have unequal numbers of readings.
3. **Rainfall by month** (bar): the **sum** of rainfall per calendar month. For multi-year data you can switch to the *average monthly total per year*. The unit is shown only if the column name or your unit choice confirms it; otherwise it says "unit not confirmed". **Months with missing or incomplete rainfall data (more than 10% of readings missing, or under 90% of the days present in daily data) are drawn hatched and called out in the interpretation, because missing days are not zeros and make that month's total too low.** The monthly table has `missing_observations`, `data_coverage_percent` and `incomplete_month` columns.
4. **Temperature vs humidity** (scatter): Pearson r is calculated only with at least 10 complete pairs; a trend line is drawn only then. The app states that correlation is not causation.
5. **Temperature distribution** (histogram, plain counts): with mean and median lines and an interpretation (spread, skew, share within one standard deviation).

If a column is missing, the app says why that chart is unavailable and still shows the rest.

## 10. Example usage
1. Start the app, choose **Use demo dataset** (or **Upload a CSV file** and pick
   `weather_sample_synthetic_test.csv` / `data/standin_weather_2025_SYNTHETIC.csv`).
   Expected for the stand-in file: 368 records, 3 duplicates removed, 365 rows after cleaning.
2. Check *Data Cleaning* to see what was fixed or flagged.
3. Use the sidebar date range to focus on a period.
4. Read *Visualizations* and *Insights*, then download CSVs from *Downloads* (including the untouched `original_uploaded_data.csv`).

## 11. Project structure and module guide
```
app.py               Streamlit interface only (7 tabs)
tkinter_app.py       Tkinter desktop interface only (calls the modules below)
data_processing.py   load_csv, detect_columns, preprocess, filter_by_date, to_csv_bytes,
                     dataset_summary, cleaning_report_text, save_csv, cleaned_export_frame
                     (the last four are shared by both interfaces)
analysis.py          statistics, monthly tables, correlation, chart availability, interpretations
visualizations.py    the five chart functions (return a figure, or None if impossible)
                     + build_chart(key, df) used by the desktop GUI
demo_data.py         synthetic demo dataset generator
data/                demo CSV          outputs/  suggested place to save downloads
tests/               158 automated tests (see section 14): 133 run anywhere, 25 drive the Tkinter
                     window and need a display. test_known_values.py and test_audit_fixes.py check
                     hand-calculated numbers.
```

## 12. Limitations
- Monthly values pool all years; they do not show long-term trends.
- Humidity is assumed to be a percentage (a warning appears if all values are between 0 and 1). Units are taken only from column names or your choice, never guessed from the values.
- Rainfall totals leave out missing readings; incomplete months are marked, not corrected.
- Outliers are flagged but included in statistics.
- Only CSV input; one temperature column is analysed (or derived as min/max midpoint).
- Correlation measures linear association only.
- The app is meant for small to medium files (it recalculates on each interaction).

## 13. Future enhancements
Excel input, multiple-station comparison, rolling averages, seasonal and year-over-year comparison, PDF report export.

## 14. Testing and verification
Run from the project folder:
```bash
python -m pytest -q                 # any computer; window tests are skipped without a screen
xvfb-run -a python -m pytest -q     # Linux without a screen: runs ALL tests incl. the Tkinter window
```
Result at the last audit (Python 3.12, pandas 3.0.2): **158 passed** with a virtual display; **133 passed,
25 skipped** without one (the 25 skipped tests are the ones that open the Tkinter window). The whole suite also
passed on pandas 2.2.3 / numpy 1.26.4 / matplotlib 3.8.4 / Streamlit 1.35.0. `AUDIT_REPORT.md` lists the
problems found and fixed, and separates automated tests from manual visual checks.
