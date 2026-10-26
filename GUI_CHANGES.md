# Desktop GUI: change summary

**New:** `tkinter_app.py` (run with `python tkinter_app.py`), `tests/test_gui_helpers.py`, `tests/test_tkinter_app.py`.

**Added (nothing removed or changed in behaviour):**
- `data_processing.py`: `has_valid_records`, `dataset_summary`, `summary_lines`, `cleaning_report_text`, `save_csv`, `ExportError`
- `visualizations.py`: `CHART_BUILDERS`, `build_chart(key, df, ...)` -> `(figure, message)`; its import line now also pulls in `chart_availability`
- `README.md`: desktop GUI section, structure and test counts. `DEMO_CHECKLIST.md`: GUI steps.

**Unchanged:** `app.py`, `analysis.py`, `demo_data.py`, `requirements.txt`, `conftest.py`, and the four existing test files.

**Tests when the GUI was added:** 114 (82 + 32). **Current count after the final audit: 158** (133 pass + 25 window tests skipped without a display; all 158 pass with `xvfb-run -a python -m pytest -q`). See `AUDIT_REPORT.md`.
