# Faculty Demonstration Checklist

**Before the demo**
- [ ] `pip install -r requirements.txt` done; `python -m pytest -q` shows all tests passing
- [ ] `streamlit run app.py` opens in the browser
- [ ] Have one real CSV ready (and know where it came from), plus the demo dataset as a backup

**During the demo**
1. [ ] Overview tab: state the problem and the workflow (Upload → ... → Export)
2. [ ] Upload a CSV: show first rows, record/column counts and data types (Dataset Explorer)
3. [ ] Data Cleaning: before/after table, duplicates, missing values, flagged values and the log
4. [ ] Explain why impossible values become *missing* (not guessed) and why outliers are kept
5. [ ] Statistics: mean, median, std. dev.; monthly tables; correlation (read the "what do these mean" box)
6. [ ] Visualizations: walk through all five charts and their interpretations
7. [ ] Change the date range in the sidebar and show that statistics and charts update
8. [ ] Un-map Rainfall in the sidebar to show the helpful "chart cannot be generated" message
9. [ ] Insights tab: read the limitations aloud
10. [ ] Download a CSV and open it; compare `original_uploaded_data.csv` with `cleaned_weather_data.csv`
11. [ ] Run `python -m pytest -q` live

**Be ready to explain**
- Total rainfall (sum) vs average rainfall
- Calendar-month average vs daily readings
- Correlation vs causation
- Why the demo data is synthetic and must never be presented as real weather

**Optional: desktop GUI (`python tkinter_app.py`)**
- [ ] Load Demo Dataset (or Open CSV) and read the Dataset Summary
- [ ] Pick each of the five visualizations and click Generate Chart
- [ ] Open the Cleaning Report; Export Cleaned CSV and open the saved file
- [ ] Click Generate Chart with no dataset loaded to show the friendly message
- [ ] Load a file with missing rainfall days and point out the hatched bar and the CAUTION text
- [ ] Switch *Temperature in file* to Fahrenheit on a Fahrenheit file and show the report update
- [ ] Export, then Open CSV on the exported file: units and numbers are unchanged
