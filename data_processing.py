"""
data_processing.py
==================
Loading, validating and cleaning a weather CSV file.

This module contains NO Streamlit code, so every function can be tested on its own.

Main entry points
-----------------
load_csv(source)                 -> raw DataFrame (exactly what is in the file)
detect_columns(columns)          -> guesses which column is "date", "temperature", ...
preprocess(raw, column_map, ...) -> CleaningResult (clean data + a full report)

Cleaning philosophy
-------------------
* Missing measurements are NEVER invented or filled in.
* Values that are physically impossible (e.g. humidity of 140 %) are set to
  "missing" and listed in a table, so nothing disappears silently.
* Values that are merely unusual (statistical outliers) are only FLAGGED and kept.
* Exact duplicate rows are removed (this can be switched off) and counted.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Internal ("standard") names for the columns the project understands.
ROLES = [
    "date",
    "temperature",
    "min_temperature",
    "max_temperature",
    "rainfall",
    "humidity",
    "wind_speed",
]
ROLE_LABELS = {
    "date": "Date",
    "temperature": "Temperature",
    "min_temperature": "Minimum temperature",
    "max_temperature": "Maximum temperature",
    "rainfall": "Rainfall",
    "humidity": "Humidity",
    "wind_speed": "Wind speed",
}
NUMERIC_ROLES = [r for r in ROLES if r != "date"]
TEMPERATURE_ROLES = ["temperature", "min_temperature", "max_temperature"]

# Physically plausible limits: (lowest, highest). None = no limit on that side.
# Temperature limits are in degrees Celsius (applied after any Fahrenheit conversion).
PLAUSIBLE_RANGES = {
    "temperature": (-90.0, 60.0),
    "min_temperature": (-90.0, 60.0),
    "max_temperature": (-90.0, 60.0),
    "rainfall": (0.0, None),
    "humidity": (0.0, 100.0),
    "wind_speed": (0.0, None),
}

# Statistical outlier rule (flag only): outside Q1 - 3*IQR ... Q3 + 3*IQR.
# Rainfall and wind are skipped because they are naturally very skewed.
OUTLIER_ROLES = TEMPERATURE_ROLES + ["humidity"]
OUTLIER_IQR_FACTOR = 3.0
MIN_ROWS_FOR_OUTLIERS = 8

RAINFALL_UNITS = ["unknown", "mm", "cm", "inches"]


class DataLoadError(ValueError):
    """Raised when a file cannot be used. The message is safe to show to the user."""


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _read_source_bytes(source) -> bytes:
    """Accept a file path, bytes or a file-like object and return raw bytes."""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if hasattr(source, "read"):
        data = source.read()
        return data.encode("utf-8") if isinstance(data, str) else data
    with open(source, "rb") as handle:
        return handle.read()


def load_csv(source: Union[str, bytes, io.IOBase]) -> pd.DataFrame:
    """
    Read a CSV file exactly as it is (no cleaning yet).

    Raises DataLoadError with a friendly message for empty or unreadable files.
    """
    try:
        data = _read_source_bytes(source)
    except FileNotFoundError:
        raise DataLoadError(f"File not found: {source}") from None

    if not data or not data.strip():
        raise DataLoadError("The file is empty.")
    if b"\x00" in data[:2048] or data.startswith(b"PK\x03\x04"):
        raise DataLoadError("This does not look like a CSV text file (it seems to be a "
                            "binary file such as Excel or a ZIP). Please save it as CSV.")

    df = None
    used_encoding = "utf-8-sig"
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=encoding)
            used_encoding = encoding
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise DataLoadError("The file contains no data.") from None
        except pd.errors.ParserError as exc:
            raise DataLoadError(f"The file could not be read as a CSV: {exc}") from None
    if df is None:
        raise DataLoadError("The file could not be decoded as text.")

    # A single column whose header contains ; or tab usually means a different separator.
    if df.shape[1] == 1:
        header = str(df.columns[0])
        for separator in (";", "\t", "|"):
            if separator in header:
                df = pd.read_csv(io.BytesIO(data), encoding=used_encoding, sep=separator)
                break

    if df.shape[1] == 0 or len(df) == 0:
        raise DataLoadError("The file has a header but no data rows.")

    df.columns = [str(c).strip() for c in df.columns]
    return df


# --------------------------------------------------------------------------- #
# Column detection
# --------------------------------------------------------------------------- #

def standardize_column_name(name) -> str:
    """'Temperature (°C)' -> 'temperature_c'; 'Wind Speed' -> 'wind_speed'."""
    cleaned = re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")
    return cleaned or "unnamed"


_EXACT_NAMES = {
    "date": {"date", "datetime", "date_time", "timestamp", "time", "obs_date",
             "observation_date", "dt"},
    "min_temperature": {"min_temperature", "minimum_temperature", "temp_min", "tmin",
                        "min_temp", "temperature_min", "mintemp", "tempmin",
                        "min_temperature_c", "min_temp_c"},
    "max_temperature": {"max_temperature", "maximum_temperature", "temp_max", "tmax",
                        "max_temp", "temperature_max", "maxtemp", "tempmax",
                        "max_temperature_c", "max_temp_c"},
    "temperature": {"temperature", "temp", "avg_temp", "mean_temp", "tavg", "temp_c",
                    "average_temperature", "mean_temperature", "avg_temperature",
                    "temperature_c", "temperature_f", "temperature_celsius"},
    "rainfall": {"rainfall", "rain", "precipitation", "prcp", "precip", "rainfall_mm",
                 "precipitation_mm", "rain_mm"},
    "humidity": {"humidity", "rh", "relative_humidity", "humidity_percent",
                 "humidity_pct"},
    "wind_speed": {"wind_speed", "windspeed", "wind", "wspd", "wind_speed_kmh",
                   "wind_speed_kph", "wind_speed_ms"},
}

_GUESSES = {
    "date": lambda s: "date" in s,
    "min_temperature": lambda s: "temp" in s and "min" in s,
    "max_temperature": lambda s: "temp" in s and "max" in s,
    "temperature": lambda s: "temp" in s and not any(
        word in s for word in ("min", "max", "dew", "low", "high", "feels", "apparent")),
    "rainfall": lambda s: any(word in s for word in ("rain", "precip", "prcp")),
    "humidity": lambda s: "humid" in s,
    "wind_speed": lambda s: "wind" in s and "dir" not in s,
}


def detect_columns(columns) -> Dict[str, str]:
    """
    Guess which original column plays which role.

    Returns {role: original_column_name} for the roles that were found.
    A column is never assigned to two roles.
    """
    standardized = {col: standardize_column_name(col) for col in columns}
    found: Dict[str, str] = {}
    used = set()
    # Order matters: min/max temperature must be claimed before plain "temperature".
    for role in ["date", "min_temperature", "max_temperature", "temperature",
                 "rainfall", "humidity", "wind_speed"]:
        for tester in (lambda s: s in _EXACT_NAMES[role], _GUESSES[role]):
            match = next((c for c, s in standardized.items()
                          if c not in used and tester(s)), None)
            if match is not None:
                found[role] = match
                used.add(match)
                break
    return found


def guess_rainfall_unit(original_name: Optional[str]) -> str:
    """Return 'mm', 'cm', 'inches' if the column name states it, else 'unknown'."""
    if not original_name:
        return "unknown"
    name = str(original_name).lower()
    if re.search(r"(\bmm\b|_mm\b|\(mm\)|millimet)", name):
        return "mm"
    if re.search(r"(\bcm\b|_cm\b|\(cm\)|centimet)", name):
        return "cm"
    if re.search(r"((?<![a-z])inch(es)?(?![a-z])|\(in\)|_in\b)", name):
        return "inches"
    return "unknown"


def guess_temperature_unit(original_name: Optional[str]) -> str:
    """Return 'F' if the column name clearly says Fahrenheit, otherwise 'C'."""
    name = standardize_column_name(original_name or "")
    if "fahrenheit" in name or name.endswith("_f") or "_f_" in name:
        return "F"
    return "C"


# --------------------------------------------------------------------------- #
# Small helpers used by the interface and the tests
# --------------------------------------------------------------------------- #

def _blank_mask(series: pd.Series) -> pd.Series:
    """True where a raw value is missing or just whitespace."""
    return series.isna() | series.astype(str).str.strip().eq("")


def parse_dates(series: pd.Series, dayfirst: bool = False) -> pd.Series:
    """
    Convert text to datetimes. Anything unreadable becomes NaT (not an exception).

    ISO dates written year-first (2025-01-02) are always read as year-month-day.
    pandas would otherwise swap day and month for them when dayfirst=True.
    """
    text = series.astype("string").str.strip()
    # Year-first dates (2025-01-02, 2025/01/02, 2025.01.02) are never day-first.
    iso = text.str.match(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}", na=False)
    # Compact dates such as 20250102 (or 20250102.0 when stored as a float column).
    compact = text.str.match(r"^\d{8}(\.0+)?$", na=False) & ~iso
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if iso.any():
        parsed[iso] = _to_datetime(text[iso], dayfirst=False)
    if compact.any():
        digits = text[compact].str.slice(0, 8)
        parsed[compact] = pd.to_datetime(digits, format="%Y%m%d", errors="coerce")
    rest = ~(iso | compact)
    if rest.any():
        # Other plain numbers (e.g. Excel serial numbers) are NOT dates we can trust:
        # pandas would read them as nanoseconds since 1970, so treat them as unreadable.
        numeric_like = text.str.match(r"^-?\d+(\.\d+)?$", na=False)
        usable = rest & ~numeric_like
        if usable.any():
            parsed[usable] = _to_datetime(text[usable], dayfirst=dayfirst)
    return parsed


def _to_datetime(series: pd.Series, dayfirst: bool) -> pd.Series:
    """pd.to_datetime that never raises, returns timezone-free datetime64[ns]."""
    try:
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst, format="mixed")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_localize(None)
    elif not pd.api.types.is_datetime64_any_dtype(parsed):
        parsed = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None)
    return parsed.astype("datetime64[ns]")


def to_numeric_clean(series: pd.Series) -> pd.Series:
    """Convert to numbers; text such as 'error' becomes NaN; infinities become NaN."""
    numbers = pd.to_numeric(series, errors="coerce").astype("float64")
    return numbers.replace([np.inf, -np.inf], np.nan)


def missing_value_table(df: pd.DataFrame) -> pd.DataFrame:
    """Missing count and percentage for every column."""
    total = len(df)
    missing = df.isna().sum()
    percent = (missing / total * 100).round(2) if total else missing * 0.0
    return pd.DataFrame({"column": df.columns.astype(str),
                         "missing": missing.values.astype(int),
                         "percent_missing": percent.values})


def count_duplicates(df: pd.DataFrame) -> int:
    """Number of rows that exactly repeat an earlier row."""
    return int(df.duplicated().sum())


def column_info(df: pd.DataFrame) -> pd.DataFrame:
    """Data type, non-missing count and missing count for each column."""
    return pd.DataFrame({
        "column": df.columns.astype(str),
        "data_type": [str(t) for t in df.dtypes],
        "non_missing": df.notna().sum().values.astype(int),
        "missing": df.isna().sum().values.astype(int),
    })


def filter_by_date(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Keep rows whose date is between start and end (inclusive)."""
    if "date" not in df.columns:
        return df
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Encode a DataFrame as CSV bytes for a download button."""
    return df.to_csv(index=False).encode("utf-8")


# --------------------------------------------------------------------------- #
# Preprocessing pipeline
# --------------------------------------------------------------------------- #

@dataclass
class CleaningResult:
    """Everything the interface needs to explain what the cleaning step did."""
    clean: pd.DataFrame
    column_map: Dict[str, str]
    units: Dict[str, str]
    log: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    flagged: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    report: Dict[str, object] = field(default_factory=dict)
    derived_temperature: bool = False


_FLAG_COLUMNS = ["source_row", "column", "value", "issue", "action"]


def _flag_rows(work: pd.DataFrame, mask: pd.Series, column: str, issue: str,
               action: str) -> pd.DataFrame:
    """Build rows for the 'suspicious values' table."""
    rows = work.loc[mask]
    return pd.DataFrame({
        "source_row": rows["source_row"].values,
        "column": column,
        "value": rows[column].astype(str).values,
        "issue": issue,
        "action": action,
    })


def preprocess(
    raw: pd.DataFrame,
    column_map: Optional[Dict[str, str]] = None,
    *,
    temperature_unit: str = "auto",
    rainfall_unit: str = "auto",
    dayfirst: bool = False,
    remove_duplicates: bool = True,
) -> CleaningResult:
    """
    Clean a raw weather DataFrame.

    Parameters
    ----------
    raw : the DataFrame returned by load_csv
    column_map : {role: original column}. If None, columns are detected automatically.
    temperature_unit : "C", "F" or "auto" (read from the column name, else Celsius).
                       Fahrenheit values are converted to Celsius.
    rainfall_unit : "mm", "cm", "inches", "unknown" or "auto" (read from the column name).
                    This is only a label; rainfall values are never converted.
    dayfirst : read 01/02/2024 as 1 February instead of 2 January.
    remove_duplicates : drop rows that exactly repeat an earlier row.
    """
    if raw is None or raw.shape[1] == 0 or len(raw) == 0:
        raise DataLoadError("The dataset is empty, so there is nothing to analyse.")

    raw = raw.copy()
    raw.columns = [str(c) for c in raw.columns]
    log: List[str] = []
    warnings: List[str] = []
    report: Dict[str, object] = {}

    # ---- 1. Decide which column plays which role ------------------------------
    if column_map is None:
        column_map = detect_columns(raw.columns)
    used_columns = set()
    mapping: Dict[str, str] = {}
    for role in ROLES:
        original = column_map.get(role)
        if original and original in raw.columns and original not in used_columns:
            mapping[role] = original
            used_columns.add(original)
        elif original and original in used_columns:
            warnings.append(f"Column '{original}' was selected for more than one role; "
                            f"it is used only for the first role.")
    if not any(role in mapping for role in NUMERIC_ROLES):
        raise DataLoadError(
            "None of the expected weather columns (temperature, rainfall, humidity, "
            "wind speed) could be identified. Columns found: "
            + ", ".join(map(str, raw.columns)))

    if str(temperature_unit).lower() == "auto":
        temperature_unit = guess_temperature_unit(mapping.get("temperature")
                                                  or mapping.get("max_temperature")
                                                  or mapping.get("min_temperature"))
    if str(rainfall_unit).lower() == "auto":
        rainfall_unit = guess_rainfall_unit(mapping.get("rainfall"))

    log.append("Detected columns: " + ", ".join(
        f"{ROLE_LABELS[r]} = '{c}'" for r, c in mapping.items()) + ".")
    missing_roles = [ROLE_LABELS[r] for r in ROLES if r not in mapping]
    if missing_roles:
        log.append("Not found / not mapped: " + ", ".join(missing_roles) + ".")

    # ---- 2. Record the 'before' picture ------------------------------------
    report["rows_before"] = len(raw)
    report["columns_before"] = raw.shape[1]
    # "Missing / invalid" BEFORE cleaning = blank cells + text that is not a number.
    # (Impossible values such as 140 % humidity are counted separately, in step 7.)
    before_missing = {}
    for role, col in mapping.items():
        blank = _blank_mask(raw[col])
        if role == "date":
            before_missing[role] = int((blank | parse_dates(raw[col], dayfirst).isna()).sum())
        else:
            before_missing[role] = int((blank | to_numeric_clean(raw[col]).isna()).sum())
    duplicates_found = count_duplicates(raw)
    report["duplicates_found"] = duplicates_found
    # Without a date column, two identical rows may be two genuine readings (for example
    # two days that both measured 25 C / 60 %). Deleting them would destroy real data,
    # so they are reported but only removed when a date column is mapped.
    can_remove_duplicates = "date" in mapping
    if remove_duplicates and duplicates_found and not can_remove_duplicates:
        warnings.append(f"{duplicates_found} repeated row(s) were found, but there is no date "
                        f"column, so they could be genuine separate readings. They were kept.")
        remove_duplicates = False

    # ---- 3. Work on a copy; keep the original CSV row number for traceability --
    work = raw.copy()
    work["source_row"] = np.arange(len(work)) + 2      # header is row 1 in the file
    duplicate_mask = raw.duplicated().values
    if remove_duplicates and duplicates_found:
        work = work.loc[~duplicate_mask].copy()
        log.append(f"Removed {duplicates_found} exact duplicate row(s) (kept the first copy).")
    elif duplicates_found:
        log.append(f"Found {duplicates_found} duplicate row(s) but kept them "
                   f"(duplicate removal is switched off).")
    else:
        log.append("No duplicate rows found.")
    report["duplicates_removed"] = duplicates_found if remove_duplicates else 0

    # Rename the mapped columns to the internal names (avoid clashes with extras).
    rename = {original: role for role, original in mapping.items()}
    for column in list(work.columns):
        if column not in rename and column in ROLES:
            rename[column] = f"extra_{column}"
    work = work.rename(columns=rename)
    work = work.reset_index(drop=True)

    # ---- 4. Dates -------------------------------------------------------------
    report["invalid_dates"] = 0
    report["missing_dates"] = 0
    if "date" in work.columns:
        original_dates = work["date"]
        blank = _blank_mask(original_dates)
        work["date"] = parse_dates(original_dates, dayfirst=dayfirst)
        invalid = (~blank) & work["date"].isna()
        report["missing_dates"] = int(blank.sum())
        report["invalid_dates"] = int(invalid.sum())
        log.append(f"Converted '{mapping['date']}' to dates: {int(work['date'].notna().sum())} "
                   f"valid, {int(blank.sum())} missing, {int(invalid.sum())} unreadable.")
        if invalid.any():
            warnings.append(f"{int(invalid.sum())} date value(s) could not be read and are "
                            f"treated as missing. Rows with no date are left out of "
                            f"time-based charts.")
        same_day = work["date"].notna() & work["date"].duplicated(keep=False)
        if same_day.any():
            n_days = int(work.loc[same_day, "date"].nunique())
            if remove_duplicates:
                warnings.append(f"{n_days} date(s) appear on more than one row with different "
                                f"values. These rows were KEPT (they are not exact duplicates); "
                                f"check them in your file.")
            else:
                warnings.append(f"{n_days} date(s) appear on more than one row. Duplicate "
                                f"removal is switched off, so all of these rows were kept "
                                f"(some may be exact duplicates).")
        if work["date"].notna().sum() == 0:
            warnings.append("No valid dates were found, so time-based analyses are unavailable.")
            work = work.drop(columns=["date"])
    else:
        warnings.append("No date column was found, so time-based analyses are unavailable.")

    # ---- 5. Numeric columns ----------------------------------------------------
    invalid_numeric: Dict[str, int] = {}
    for role in NUMERIC_ROLES:
        if role not in work.columns:
            continue
        original_values = work[role]
        blank = _blank_mask(original_values)
        numbers = to_numeric_clean(original_values)
        bad_text = (~blank) & numbers.isna()
        invalid_numeric[role] = int(bad_text.sum())
        non_blank = int((~blank).sum())
        if bad_text.sum() >= 3 and bad_text.sum() > 0.5 * non_blank:
            example = str(original_values[bad_text].iloc[0])
            warnings.append(
                f"Most values in the {ROLE_LABELS[role].lower()} column ({int(bad_text.sum())} of "
                f"{non_blank}) are not numbers, for example '{example}'. If your file writes "
                f"decimals with a comma (20,5), re-save it with a decimal point (20.5).")
        if bad_text.any():
            flagged_text = _flag_rows(work.assign(**{role: original_values}), bad_text, role,
                                      "not a number", "set to missing")
            report.setdefault("_flag_frames", []).append(flagged_text)
            log.append(f"{ROLE_LABELS[role]}: {int(bad_text.sum())} non-numeric value(s) "
                       f"set to missing.")
        work[role] = numbers
    report["invalid_numeric"] = invalid_numeric

    # ---- 6. Temperature unit ---------------------------------------------------
    if str(temperature_unit).upper().startswith("F"):
        for role in TEMPERATURE_ROLES:
            if role in work.columns:
                work[role] = (work[role] - 32.0) * 5.0 / 9.0
        log.append("Converted temperature columns from Fahrenheit to Celsius.")
    temp_unit_label = "°C"

    # ---- 7. Plausibility checks (impossible values -> missing, listed) ---------
    implausible: Dict[str, int] = {}
    for role in NUMERIC_ROLES:
        if role not in work.columns:
            continue
        low, high = PLAUSIBLE_RANGES[role]
        values = work[role]
        bad = pd.Series(False, index=work.index)
        if low is not None:
            bad |= values < low
        if high is not None:
            bad |= values > high
        implausible[role] = int(bad.sum())
        n_valid = int(values.notna().sum())
        if role in TEMPERATURE_ROLES and bad.sum() >= 2 and bad.sum() >= 0.5 * n_valid:
            warnings.append(
                f"{int(bad.sum())} of {n_valid} {ROLE_LABELS[role].lower()} values are outside "
                f"the plausible range for degrees Celsius (-90 to 60) and were set to missing. "
                f"If this column is in Fahrenheit, set the temperature unit to Fahrenheit "
                f"(Streamlit: sidebar > Cleaning options; desktop window: 'Temperature in "
                f"file' list).")
        if bad.any():
            limits = (f"{low if low is not None else '-inf'} to "
                      f"{high if high is not None else 'inf'}")
            report.setdefault("_flag_frames", []).append(_flag_rows(
                work, bad, role, f"outside plausible range ({limits})", "set to missing"))
            log.append(f"{ROLE_LABELS[role]}: {int(bad.sum())} impossible value(s) "
                       f"(outside {limits}) set to missing.")
            work.loc[bad, role] = np.nan
    report["implausible"] = implausible

    if "humidity" in work.columns:
        humidity_values = work["humidity"].dropna()
        if len(humidity_values) >= 5 and humidity_values.max() <= 1.0 and humidity_values.max() > 0:
            warnings.append("All humidity values are between 0 and 1, so they may be fractions "
                            "rather than percentages. They are shown as they are, labelled '%'.")
    report["missing_rainfall"] = (int(work["rainfall"].isna().sum())
                                  if "rainfall" in work.columns else 0)
    if report["missing_rainfall"]:
        warnings.append(f"{report['missing_rainfall']} rainfall reading(s) are missing or invalid. "
                        f"Monthly rainfall totals leave them out, so months with missing "
                        f"readings are under-counted (see the rainfall chart notes).")

    # min > max check (flag only)
    report["min_above_max"] = 0
    if "min_temperature" in work.columns and "max_temperature" in work.columns:
        swapped = (work["min_temperature"] > work["max_temperature"])
        report["min_above_max"] = int(swapped.sum())
        if swapped.any():
            rows = work.loc[swapped]
            report.setdefault("_flag_frames", []).append(pd.DataFrame({
                "source_row": rows["source_row"].values,
                "column": "min_temperature / max_temperature",
                "value": (rows["min_temperature"].round(2).astype(str) + " > "
                          + rows["max_temperature"].round(2).astype(str)).values,
                "issue": "minimum temperature is higher than maximum temperature",
                "action": "kept (flagged only)"}))
            log.append(f"{int(swapped.sum())} row(s) have minimum temperature above maximum "
                       f"temperature; flagged, not changed.")

    # Statistical outliers (flag only)
    outliers_flagged = 0
    for role in OUTLIER_ROLES:
        if role not in work.columns:
            continue
        values = work[role].dropna()
        if len(values) < MIN_ROWS_FOR_OUTLIERS:
            continue
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        low, high = q1 - OUTLIER_IQR_FACTOR * iqr, q3 + OUTLIER_IQR_FACTOR * iqr
        outlier = (work[role] < low) | (work[role] > high)
        if outlier.any():
            outliers_flagged += int(outlier.sum())
            report.setdefault("_flag_frames", []).append(_flag_rows(
                work, outlier, role,
                f"extreme outlier (outside {low:.1f} to {high:.1f}, i.e. 3 x IQR rule)",
                "kept (flagged only)"))
            log.append(f"{ROLE_LABELS[role]}: {int(outlier.sum())} extreme outlier(s) flagged "
                       f"but kept, because unusual weather can be real.")
    report["outliers_flagged"] = outliers_flagged

    # ---- 8. Derived temperature (only when no temperature column exists) -------
    derived = False
    if ("temperature" not in work.columns and "min_temperature" in work.columns
            and "max_temperature" in work.columns):
        work["temperature"] = (work["min_temperature"] + work["max_temperature"]) / 2.0
        derived = True
        warnings.append("No average temperature column was found. 'Temperature' was DERIVED "
                        "as the midpoint of minimum and maximum temperature, only for rows "
                        "where both exist. This is an approximation of the daily mean.")
        log.append("Derived temperature = (minimum + maximum) / 2 where both are available.")

    # ---- 9. Calendar columns ------------------------------------------------------
    if "date" in work.columns:
        work["year"] = work["date"].dt.year.astype("Int64")
        work["month"] = work["date"].dt.month.astype("Int64")
        work["month_name"] = work["date"].dt.month_name()
        work["year_month"] = work["date"].dt.strftime("%Y-%m")
        work = work.sort_values("date", kind="stable", na_position="last")
        log.append("Added year, month, month name and year-month columns; sorted by date.")
    work = work.reset_index(drop=True)

    # ---- 10. Units ----------------------------------------------------------------------
    units = {"temperature": temp_unit_label, "rainfall": rainfall_unit}
    if "rainfall" in work.columns and rainfall_unit == "unknown":
        warnings.append("The rainfall unit is not confirmed. Totals are shown without a unit "
                        "label; if you know the unit, choose it in the 'Rainfall unit' list "
                        "(Streamlit sidebar or desktop window).")

    # ---- 11. Assemble the report tables ----------------------------------------------------
    frames = report.pop("_flag_frames", [])
    flagged = (pd.concat(frames, ignore_index=True).sort_values(["source_row", "column"])
               .reset_index(drop=True) if frames else pd.DataFrame(columns=_FLAG_COLUMNS))
    report["rows_after"] = len(work)
    report["after_missing"] = {role: int(work[role].isna().sum())
                               for role in mapping if role in work.columns}
    report["before_missing"] = before_missing

    summary_rows = [
        ("Rows", len(raw), len(work)),
        ("Duplicate rows", duplicates_found,
         count_duplicates(work.drop(columns=["source_row"])) if len(work) else 0),
    ]
    for role in mapping:
        if role in work.columns:
            summary_rows.append((f"Missing / invalid {ROLE_LABELS[role].lower()}",
                                 before_missing[role], int(work[role].isna().sum())))
    # Impossible values (e.g. humidity 140 %) become missing, so "After" can be higher than
    # "Before" for those columns. This row explains the difference.
    impossible_total = int(sum(implausible.values()))
    summary_rows.append(("Impossible values set to missing (included in 'After' above)",
                         impossible_total, impossible_total))
    summary = pd.DataFrame(summary_rows, columns=["Item", "Before cleaning", "After cleaning"])
    summary["Before cleaning"] = summary["Before cleaning"].astype(str)
    summary["After cleaning"] = summary["After cleaning"].astype(str)

    return CleaningResult(
        clean=work, column_map=mapping, units=units, log=log, warnings=warnings,
        flagged=flagged, summary=summary, report=report, derived_temperature=derived)


# --------------------------------------------------------------------------- #
# Helpers shared by the desktop (Tkinter) interface
# --------------------------------------------------------------------------- #

class ExportError(Exception):
    """Raised when cleaned data cannot be written to a file. The message is user-friendly."""


def has_valid_records(clean: pd.DataFrame) -> bool:
    """True if at least one weather measurement survived cleaning."""
    return bool(clean is not None and len(clean) > 0 and any(
        role in clean.columns and clean[role].notna().any() for role in NUMERIC_ROLES))


def dataset_summary(raw: pd.DataFrame, result: "CleaningResult") -> Dict[str, object]:
    """
    Headline facts about an uploaded dataset: rows, columns, date range, missing values
    and duplicates. Counts describe the file AS UPLOADED (blank cells and duplicates before
    cleaning); the date range comes from the cleaned dates.
    """
    clean = result.clean
    missing = {str(col): int(_blank_mask(raw[col]).sum()) for col in raw.columns}
    missing = {col: count for col, count in missing.items() if count}
    date_range = None
    if "date" in clean.columns and clean["date"].notna().any():
        date_range = (clean["date"].min(), clean["date"].max())
    return {
        "rows": int(len(raw)),
        "n_columns": int(raw.shape[1]),
        "columns": [str(c) for c in raw.columns],
        "date_range": date_range,
        "missing_total": int(sum(missing.values())),
        "missing_by_column": missing,
        "duplicates": count_duplicates(raw),
        "rows_after_cleaning": int(len(clean)),
        "valid_records": has_valid_records(clean),
    }


def cleaning_report_text(result: "CleaningResult", max_flagged: int = 25) -> str:
    """Plain-text version of the cleaning report (same facts the Streamlit tab shows)."""
    lines = ["CLEANING REPORT", "=" * 60, "", "Before / after summary", "-" * 60,
             result.summary.to_string(index=False), "", "What was done", "-" * 60]
    lines += [f"- {entry}" for entry in result.log]
    if result.warnings:
        lines += ["", "Warnings", "-" * 60] + [f"! {warning}" for warning in result.warnings]
    lines += ["", f"Suspicious or invalid values: {len(result.flagged)}", "-" * 60]
    if result.flagged.empty:
        lines.append("None found.")
    else:
        lines.append("('source_row' = row number in your CSV file; header = row 1.)")
        lines.append(result.flagged.head(max_flagged).to_string(index=False))
        if len(result.flagged) > max_flagged:
            lines.append(f"... and {len(result.flagged) - max_flagged} more.")
    return "\n".join(lines)


def save_csv(df: pd.DataFrame, path) -> str:
    """Write a DataFrame to `path` as CSV. Raises ExportError with a clear message on failure."""
    if df is None or len(df) == 0:
        raise ExportError("There is no cleaned data to export.")
    try:
        with open(path, "wb") as handle:
            handle.write(to_csv_bytes(df))
    except OSError as exc:
        raise ExportError(f"Could not save the file '{path}': {exc.strerror or exc}") from None
    return str(path)


def summary_lines(summary: Dict[str, object], max_columns_listed: int = 6) -> List[tuple]:
    """Turn dataset_summary() into [(label, text), ...] rows for display."""
    columns = summary["columns"]
    shown = ", ".join(columns[:max_columns_listed])
    if len(columns) > max_columns_listed:
        shown += f", ... (+{len(columns) - max_columns_listed} more)"
    if summary["date_range"]:
        start, end = summary["date_range"]
        date_text = f"{start:%Y-%m-%d} to {end:%Y-%m-%d}"
    else:
        date_text = "No valid dates found"
    if summary["missing_total"]:
        worst = sorted(summary["missing_by_column"].items(), key=lambda item: -item[1])[:4]
        missing_text = (f"{summary['missing_total']:,} blank cell(s): "
                        + ", ".join(f"{col} ({n})" for col, n in worst))
    else:
        missing_text = "None"
    return [
        ("Rows", f"{summary['rows']:,}  (after cleaning: {summary['rows_after_cleaning']:,})"),
        ("Columns", f"{summary['n_columns']}: {shown}"),
        ("Date range", date_text),
        ("Missing values", missing_text),
        ("Duplicate records", f"{summary['duplicates']:,}"),
    ]


# --------------------------------------------------------------------------- #
# Exporting cleaned data so that it can be loaded again without losing the units
# --------------------------------------------------------------------------- #

_RAINFALL_SUFFIX = {"mm": "mm", "cm": "cm", "inches": "inches"}


def cleaned_export_frame(clean: pd.DataFrame, units: Dict[str, str]) -> pd.DataFrame:
    """
    The cleaned table with unit-carrying column names, ready to save as CSV.

    The internal names (temperature, rainfall, ...) have no units, so a reloaded file would
    lose the fact that rainfall was in mm or that temperatures were converted to Celsius.
    Names such as `temperature_c`, `humidity_percent` and `rainfall_mm` are recognised by
    detect_columns/guess_*_unit, so the exported file loads back with the same meaning.
    Values are not changed. The rainfall suffix is added only if the unit is known.
    """
    rename = {"temperature": "temperature_c", "min_temperature": "min_temperature_c",
              "max_temperature": "max_temperature_c", "humidity": "humidity_percent"}
    suffix = _RAINFALL_SUFFIX.get(str((units or {}).get("rainfall", "unknown")))
    if suffix:
        rename["rainfall"] = f"rainfall_{suffix}"
    return clean.rename(columns={old: new for old, new in rename.items() if old in clean.columns})
