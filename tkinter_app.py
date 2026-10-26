"""
tkinter_app.py
==============
Desktop (Tkinter) interface for the "Weather and Temperature Analysis" project.

Run with:   python tkinter_app.py

This file only builds the window and reacts to button clicks. All the real work is done by
the same modules the Streamlit dashboard (app.py) uses:

    data_processing.py  load_csv, preprocess, dataset_summary, cleaning_report_text, save_csv
    analysis.py         generate_interpretations, CHART_TITLES
    visualizations.py   build_chart (the five chart functions)
    demo_data.py        the SYNTHETIC demonstration dataset

Columns are auto-detected and units follow the pipeline's "auto" logic, exactly as in the
pipeline defaults. (The Streamlit app additionally lets you override columns and units.)
"""

import os
import sys

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    import tkinter.font as tkfont
except ImportError:                                   # Tkinter is optional on some Linux installs
    if __name__ == "__main__":
        sys.exit("Tkinter is not installed for this Python.\n"
                 "Linux: sudo apt install python3-tk     Windows/macOS: reinstall Python from "
                 "python.org and keep 'tcl/tk' ticked.\n"
                 "The Streamlit dashboard (streamlit run app.py) does not need Tkinter.")
    raise

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from analysis import CHART_TITLES, generate_interpretations
from data_processing import (DataLoadError, ExportError, cleaned_export_frame,
                             cleaning_report_text, dataset_summary, has_valid_records, load_csv,
                             preprocess, save_csv, summary_lines)
from demo_data import DEMO_FILE_NAME, demo_csv_bytes
from visualizations import build_chart

WINDOW_TITLE = "Weather and Temperature Analysis"
DESCRIPTION = ("Load a weather CSV file (or the built-in synthetic demo), review the data "
               "quality, and explore temperature, rainfall and humidity patterns with five charts.")
NO_DATASET_MESSAGE = ("No dataset has been selected.\n\n"
                      "Click 'Open CSV' to choose a weather file, or 'Load Demo Dataset'.")
NO_VALID_RECORDS_MESSAGE = ("The dataset contains no valid records: every weather value is "
                            "missing or invalid after cleaning. Open the Cleaning Report to see why.")
DEMO_BANNER = "DEMO DATA: synthetic, computer-generated values. NOT real weather observations."
CHART_PLACEHOLDER = "Load a dataset, choose a visualization and click 'Generate Chart'."
# What the user sees -> the value data_processing.preprocess expects.
TEMPERATURE_CHOICES = {"Auto (from column name)": "auto", "Celsius (°C)": "C",
                       "Fahrenheit (°F)": "F"}
RAINFALL_CHOICES = {"Auto (from column name)": "auto", "mm": "mm", "cm": "cm",
                    "inches": "inches", "Unknown": "unknown"}


class WeatherApp:
    """The main window. One instance = one window."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.raw = None               # DataFrame exactly as read from the file
        self.result = None            # CleaningResult from data_processing.preprocess
        self.dataset_name = None
        self.is_demo = False
        self.canvas = None            # FigureCanvasTkAgg currently shown
        self.report_window = None

        self.chart_keys = {title: key for key, title in CHART_TITLES.items()}
        self.chart_choice = tk.StringVar(value=CHART_TITLES["temperature_over_time"])
        self.dayfirst = tk.BooleanVar(value=False)
        self.temperature_choice = tk.StringVar(value=list(TEMPERATURE_CHOICES)[0])
        self.rainfall_choice = tk.StringVar(value=list(RAINFALL_CHOICES)[0])
        self.units_var = tk.StringVar(value="")
        self.summary_vars = {}
        self.file_var = tk.StringVar(value="No file loaded")
        self.banner_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready. Open a CSV file or load the demo dataset.")
        self.interpretation_var = tk.StringVar(value="")

        self._build_ui()

    # ------------------------------------------------------------------ #
    # Message boxes (separate methods so tests can replace them)
    # ------------------------------------------------------------------ #
    def show_error(self, message: str):
        self.status_var.set("Error: " + message.splitlines()[0])
        messagebox.showerror(WINDOW_TITLE, message, parent=self.root)

    def show_warning(self, message: str):
        self.status_var.set("Warning: " + message.splitlines()[0])
        messagebox.showwarning(WINDOW_TITLE, message, parent=self.root)

    def show_info(self, message: str):
        self.status_var.set(message.splitlines()[0])
        messagebox.showinfo(WINDOW_TITLE, message, parent=self.root)

    def ask_open_path(self) -> str:
        return filedialog.askopenfilename(
            parent=self.root, title="Open weather CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])

    def ask_save_path(self) -> str:
        return filedialog.asksaveasfilename(
            parent=self.root, title="Export cleaned CSV", defaultextension=".csv",
            initialfile="cleaned_weather_data.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])

    # ------------------------------------------------------------------ #
    # Building the window
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        root = self.root
        root.title(WINDOW_TITLE)
        root.geometry("1180x740")
        root.minsize(980, 620)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        base = tkfont.nametofont("TkDefaultFont")
        title_font = base.copy()
        title_font.configure(size=base.cget("size") + 8, weight="bold")
        bold_font = base.copy()
        bold_font.configure(weight="bold")
        style = ttk.Style()
        style.configure("Title.TLabel", font=title_font, foreground="#1f4e79")
        style.configure("Key.TLabel", font=bold_font)
        style.configure("Banner.TLabel", foreground="#9c1c1c", font=bold_font)
        style.configure("Accent.TButton", font=bold_font)

        # ---- header ----
        header = ttk.Frame(root, padding=(14, 10, 14, 4))
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text=WINDOW_TITLE, style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text=DESCRIPTION, wraplength=1100).pack(anchor="w", pady=(2, 0))
        ttk.Label(header, textvariable=self.banner_var, style="Banner.TLabel").pack(anchor="w")

        # ---- body: controls on the left, chart on the right ----
        body = ttk.Frame(root, padding=(14, 4, 14, 4))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body, width=440)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        left.grid_propagate(False)
        left.columnconfigure(0, weight=1)

        data_box = ttk.LabelFrame(left, text="1. Dataset", padding=8)
        data_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        data_box.columnconfigure((0, 1), weight=1)
        self.open_button = ttk.Button(data_box, text="Open CSV", command=self.open_csv)
        self.open_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.demo_button = ttk.Button(data_box, text="Load Demo Dataset", command=self.load_demo)
        self.demo_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ttk.Checkbutton(data_box, text="Dates are day-first (31/01/2024)",
                        variable=self.dayfirst, command=self._reprocess).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(data_box, text="Temperature in file:").grid(row=2, column=0, sticky="w",
                                                              pady=(4, 0))
        temp_box = ttk.Combobox(data_box, textvariable=self.temperature_choice, state="readonly",
                                values=list(TEMPERATURE_CHOICES), width=20)
        temp_box.grid(row=2, column=1, sticky="ew", pady=(4, 0))
        temp_box.bind("<<ComboboxSelected>>", lambda event: self._reprocess())
        ttk.Label(data_box, text="Rainfall unit:").grid(row=3, column=0, sticky="w", pady=(4, 0))
        rain_box = ttk.Combobox(data_box, textvariable=self.rainfall_choice, state="readonly",
                                values=list(RAINFALL_CHOICES), width=20)
        rain_box.grid(row=3, column=1, sticky="ew", pady=(4, 0))
        rain_box.bind("<<ComboboxSelected>>", lambda event: self._reprocess())
        ttk.Label(data_box, textvariable=self.file_var, wraplength=390).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(data_box, textvariable=self.units_var, wraplength=390).grid(
            row=5, column=0, columnspan=2, sticky="w")

        summary_box = ttk.LabelFrame(left, text="Dataset Summary", padding=8)
        summary_box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        summary_box.columnconfigure(1, weight=1)
        for row, label in enumerate(["Rows", "Columns", "Date range", "Missing values",
                                     "Duplicate records"]):
            ttk.Label(summary_box, text=label + ":", style="Key.TLabel").grid(
                row=row, column=0, sticky="nw", pady=2, padx=(0, 6))
            variable = tk.StringVar(value="-")
            self.summary_vars[label] = variable
            ttk.Label(summary_box, textvariable=variable, wraplength=250,
                      justify="left").grid(row=row, column=1, sticky="nw", pady=2)

        action_box = ttk.LabelFrame(left, text="2. Analysis", padding=8)
        action_box.grid(row=2, column=0, sticky="ew")
        action_box.columnconfigure(0, weight=1)
        ttk.Label(action_box, text="Visualization:").grid(row=0, column=0, sticky="w")
        self.chart_box = ttk.Combobox(action_box, textvariable=self.chart_choice, state="readonly",
                                      values=list(self.chart_keys))
        self.chart_box.grid(row=1, column=0, sticky="ew", pady=(2, 6))
        self.generate_button = ttk.Button(action_box, text="Generate Chart",
                                          style="Accent.TButton", command=self.generate_chart)
        self.generate_button.grid(row=2, column=0, sticky="ew", pady=2)
        self.report_button = ttk.Button(action_box, text="Cleaning Report",
                                        command=self.show_cleaning_report)
        self.report_button.grid(row=3, column=0, sticky="ew", pady=2)
        self.export_button = ttk.Button(action_box, text="Export Cleaned CSV",
                                        command=self.export_cleaned)
        self.export_button.grid(row=4, column=0, sticky="ew", pady=2)

        right = ttk.LabelFrame(body, text="Chart", padding=6)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        self.chart_holder = ttk.Frame(right)
        self.chart_holder.grid(row=0, column=0, sticky="nsew")
        interpretation = ttk.Label(right, textvariable=self.interpretation_var, wraplength=620,
                                   justify="left")
        interpretation.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        right.bind("<Configure>", lambda event: interpretation.configure(
            wraplength=max(300, event.width - 30)))
        self._show_placeholder(CHART_PLACEHOLDER)

        # ---- status bar ----
        ttk.Label(root, textvariable=self.status_var, relief="sunken", anchor="w",
                  padding=(8, 3)).grid(row=2, column=0, sticky="ew")

    # ------------------------------------------------------------------ #
    # Loading data
    # ------------------------------------------------------------------ #
    def open_csv(self):
        path = self.ask_open_path()
        if path:
            self.load_path(path)

    def load_path(self, path) -> bool:
        """Read a CSV file and analyse it. Returns True on success."""
        try:
            raw = load_csv(path)
        except DataLoadError as error:
            self.show_error(f"The file could not be loaded.\n\n{error}")
            return False
        except Exception as error:                       # keep the window alive
            self.show_error(f"Unexpected problem while reading the file.\n\n{error}")
            return False
        return self._set_dataset(raw, os.path.basename(str(path)), is_demo=False)

    def load_demo(self) -> bool:
        """Load the SYNTHETIC demonstration dataset."""
        try:
            raw = load_csv(demo_csv_bytes())
        except Exception as error:
            self.show_error(f"The demo dataset could not be loaded.\n\n{error}")
            return False
        return self._set_dataset(raw, DEMO_FILE_NAME, is_demo=True)

    def _reprocess(self):
        """Re-run the cleaning after the day-first box or a unit list changed."""
        if self.raw is not None:
            self._set_dataset(self.raw, self.dataset_name, self.is_demo)

    def _busy(self, on: bool, message: str = ""):
        """Show a wait cursor (and a status message) during slow work."""
        self.root.configure(cursor="watch" if on else "")
        if message:
            self.status_var.set(message)
        self.root.update_idletasks()

    def _set_dataset(self, raw, name, is_demo) -> bool:
        self._busy(True, f"Cleaning {name} ...")
        try:
            result = preprocess(raw, dayfirst=self.dayfirst.get(),
                                temperature_unit=TEMPERATURE_CHOICES[self.temperature_choice.get()],
                                rainfall_unit=RAINFALL_CHOICES[self.rainfall_choice.get()])
        except DataLoadError as error:
            self._busy(False)
            self.show_error("The dataset cannot be analysed. A required column is missing or "
                            f"was not recognised.\n\n{error}")
            return False
        self._busy(False)
        self.raw, self.result = raw, result
        self.dataset_name, self.is_demo = name, is_demo
        if self.report_window is not None and self.report_window.winfo_exists():
            self.report_window.destroy()                 # it described the previous dataset
        self.file_var.set(f"File: {name}")
        self.banner_var.set(DEMO_BANNER if is_demo else "")
        self._update_summary()
        self._show_placeholder(CHART_PLACEHOLDER)
        if not has_valid_records(result.clean):
            self.show_warning(NO_VALID_RECORDS_MESSAGE)
            return True
        note = f" {len(result.warnings)} warning(s): see Cleaning Report." if result.warnings else ""
        self.status_var.set(f"Loaded {name}: {len(raw):,} rows.{note}")
        return True

    def _update_summary(self):
        summary = dataset_summary(self.raw, self.result)
        for label, text in summary_lines(summary):
            self.summary_vars[label].set(text)
        rain = self.result.units["rainfall"]
        self.units_var.set("Units used: temperature °C, rainfall "
                           + (rain if rain != "unknown" else "unit not confirmed"))

    # ------------------------------------------------------------------ #
    # Charts
    # ------------------------------------------------------------------ #
    def _require_data(self, need_valid_records: bool = True) -> bool:
        if self.result is None:
            self.show_info(NO_DATASET_MESSAGE)
            return False
        if need_valid_records and not has_valid_records(self.result.clean):
            self.show_error(NO_VALID_RECORDS_MESSAGE)
            return False
        return True

    def _clear_chart_area(self):
        for child in self.chart_holder.winfo_children():
            child.destroy()
        self.canvas = None
        self.interpretation_var.set("")

    def _show_placeholder(self, text: str):
        self._clear_chart_area()
        ttk.Label(self.chart_holder, text=text, wraplength=560, justify="center",
                  foreground="#555555").place(relx=0.5, rely=0.5, anchor="center")

    def generate_chart(self):
        """Draw the selected chart inside the window."""
        if not self._require_data():
            return
        key = self.chart_keys.get(self.chart_choice.get())
        if key is None:
            self.show_info("Please choose a visualization from the list first.")
            return
        self._busy(True, f"Drawing {CHART_TITLES[key]} ...")
        try:
            figure, message = build_chart(key, self.result.clean, self.result.units["rainfall"])
        finally:
            self._busy(False)
        if figure is None:
            self._show_placeholder(message)
            self.show_warning(f"{CHART_TITLES[key]} cannot be shown.\n\n{message}")
            return
        self._clear_chart_area()
        try:
            figure.set_layout_engine("tight")            # keep labels visible when resized
        except Exception:
            pass
        self.canvas = FigureCanvasTkAgg(figure, master=self.chart_holder)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.root.update_idletasks()                     # let Tk size the widget before drawing
        self.canvas.draw()
        text = generate_interpretations(self.result.clean, self.result.units["rainfall"]).get(key)
        self.interpretation_var.set(f"Interpretation: {text}" if text else "")
        suffix = " (synthetic demo data)" if self.is_demo else ""
        self.status_var.set(f"Showing: {CHART_TITLES[key]}{suffix}")

    # ------------------------------------------------------------------ #
    # Cleaning report and export
    # ------------------------------------------------------------------ #
    def show_cleaning_report(self):
        """Open the cleaning report in a scrollable window."""
        if not self._require_data(need_valid_records=False):
            return None
        if self.report_window is not None and self.report_window.winfo_exists():
            self.report_window.destroy()
        window = tk.Toplevel(self.root)
        window.title("Cleaning Report: " + str(self.dataset_name))
        window.geometry("820x560")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)
        text = tk.Text(window, wrap="none", font=("Courier", 10), padx=8, pady=8)
        y_scroll = ttk.Scrollbar(window, orient="vertical", command=text.yview)
        x_scroll = ttk.Scrollbar(window, orient="horizontal", command=text.xview)
        text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        text.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        text.insert("1.0", cleaning_report_text(self.result))
        text.configure(state="disabled")
        self.report_window = window
        return window

    def export_cleaned(self):
        """Save the cleaned data (same table the Streamlit app offers) as CSV."""
        if not self._require_data():
            return
        path = self.ask_save_path()
        if not path:
            return
        try:
            save_csv(cleaned_export_frame(self.result.clean, self.result.units), path)
        except ExportError as error:
            self.show_error(f"Export failed.\n\n{error}")
            return
        self.show_info(f"Saved {len(self.result.clean):,} cleaned rows to:\n{path}")


def main() -> int:
    try:
        root = tk.Tk()
    except tk.TclError as error:
        print("Could not open a window (is a display available?):", error)
        return 1
    WeatherApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
