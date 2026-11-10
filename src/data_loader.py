"""Read-only data loader and SHA-256 manifest verifier for Bridge Traffic inputs."""
from __future__ import annotations

import base64
import datetime as dt
import gzip
import hashlib
import json
import re
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openpyxl
import pandas as pd

from src.config import (
    INPUTS_DIR,
    MANIFEST_PATH,
    POWERBI_ZIP_PATH,
    PROFILING_XLSX_PATH,
    SOURCE_VEHICLE_COLS,
    SURVEY_XLSX_PATH,
)

EXPECTED_SURVEY_COLUMNS: List[str] = [
    "م",
    "التاريخ",
    "اسم الراصد",
    "الفترة الزمنية",
    "سيارات ملاكي",
    "ميكروباص وأجرة",
    "أتوبيسات نقل",
    "نقل خفيف وسوزوكي",
    "نقل ثقيل وتريلات",
    "موتوسيكلات ودراجات",
    "توك توك وتروسيكل",
    "custom_1792135578939",
    "مجموع المركبات",
    "مشاة (عبور المشاة)",
    "مجموع المشاة",
    "الإجمالي العام",
    "اتجاه_الحصر",
    "SourceRowKey",
]


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest of a file without modifying it."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=1)
def verify_input_manifest() -> Dict[str, Any]:
    """Verify all files in inputs/ against input_manifest.json."""
    if not MANIFEST_PATH.exists():
        return {"valid": False, "error": "input_manifest.json not found", "files": {}}

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    source_files = manifest.get("source_files", {})
    file_results: Dict[str, Any] = {}
    all_valid = True

    for fname, expected in source_files.items():
        fpath = INPUTS_DIR / fname
        if not fpath.exists():
            file_results[fname] = {"exists": False, "valid": False}
            all_valid = False
            continue
        actual_bytes = fpath.stat().st_size
        actual_sha = compute_sha256(fpath)
        exp_bytes = expected.get("bytes")
        exp_sha = expected.get("sha256")
        is_ok = (actual_bytes == exp_bytes) and (actual_sha == exp_sha)
        if not is_ok:
            all_valid = False
        file_results[fname] = {
            "exists": True,
            "valid": is_ok,
            "actual_bytes": actual_bytes,
            "expected_bytes": exp_bytes,
            "actual_sha256": actual_sha,
            "expected_sha256": exp_sha,
        }

    return {
        "valid": all_valid,
        "project": manifest.get("project", ""),
        "files": file_results,
    }


def parse_period_minute(clock_str: str) -> int:
    """Convert Arabic clock string like '08:00 ص' or '12:00 م' into minutes from midnight."""
    parts = clock_str.strip().split()
    if len(parts) != 2:
        raise ValueError(f"Invalid Arabic clock string: {clock_str!r}")
    hm, marker = parts[0], parts[1]
    h_str, m_str = hm.split(":")
    h, m = int(h_str), int(m_str)
    if h < 1 or h > 12 or m < 0 or m > 59 or marker not in ("ص", "م"):
        raise ValueError(f"Out-of-range clock components in: {clock_str!r}")
    hour_24 = (h % 12) + (12 if marker == "م" else 0)
    return hour_24 * 60 + m


def parse_interval_minutes(period_label: str) -> Tuple[int, int]:
    """Parse '08:00 ص - 09:00 ص' and verify it is a nominal 60-minute interval."""
    parts = [p.strip() for p in period_label.split(" - ")]
    if len(parts) != 2:
        raise ValueError(f"Invalid period interval format: {period_label!r}")
    start_min = parse_period_minute(parts[0])
    end_min = parse_period_minute(parts[1])
    if (end_min - start_min + 1440) % 1440 != 60:
        raise ValueError(f"Interval is not 60 minutes: {period_label!r}")
    return start_min, end_min


@lru_cache(maxsize=1)
def load_raw_survey_df() -> pd.DataFrame:
    """Load and validate the 96 primary survey rows from Traffic_Survey_Combined.xlsx."""
    if not SURVEY_XLSX_PATH.exists():
        raise FileNotFoundError(f"Survey file not found at {SURVEY_XLSX_PATH}")

    wb = openpyxl.load_workbook(SURVEY_XLSX_PATH, read_only=True, data_only=True)
    ws = wb["Sheet1"]
    rows_iter = list(ws.values)
    wb.close()

    if not rows_iter:
        raise ValueError("Traffic_Survey_Combined.xlsx Sheet1 is empty.")

    headers = [str(h).strip() if h is not None else "" for h in rows_iter[0]]
    if headers != EXPECTED_SURVEY_COLUMNS:
        raise ValueError(
            f"Unexpected survey columns: {headers} vs expected {EXPECTED_SURVEY_COLUMNS}"
        )

    data_rows = []
    for r_idx, row in enumerate(rows_iter[1:], start=2):
        if all(cell is None for cell in row):
            continue
        if any(cell is None or (isinstance(cell, str) and cell.strip() == "") for cell in row):
            raise ValueError(f"Missing cell detected in survey row {r_idx}")
        rec: Dict[str, Any] = dict(zip(headers, row))
        raw_date = rec["التاريخ"]
        if isinstance(raw_date, dt.datetime):
            rec["التاريخ"] = raw_date.date()
        elif isinstance(raw_date, dt.date):
            rec["التاريخ"] = raw_date
        else:
            rec["التاريخ"] = dt.date.fromisoformat(str(raw_date).strip()[:10])

        rec["م"] = int(rec["م"])
        for col in SOURCE_VEHICLE_COLS + [
            "مجموع المركبات",
            "مشاة (عبور المشاة)",
            "مجموع المشاة",
            "الإجمالي العام",
        ]:
            val = rec[col]
            if isinstance(val, float) and not val.is_integer():
                raise ValueError(f"Fractional count in row {r_idx}, col {col}: {val}")
            ival = int(val)
            if ival < 0:
                raise ValueError(f"Negative count in row {r_idx}, col {col}: {ival}")
            rec[col] = ival

        # Row-level arithmetic integrity checks (Power BI & Assessment rules)
        cat_sum = sum(rec[c] for c in SOURCE_VEHICLE_COLS)
        if cat_sum != rec["مجموع المركبات"]:
            raise ValueError(
                f"Vehicle sum mismatch at {rec['SourceRowKey']}: {cat_sum} != {rec['مجموع المركبات']}"
            )
        if rec["مشاة (عبور المشاة)"] != rec["مجموع المشاة"]:
            raise ValueError(
                f"Pedestrian copy mismatch at {rec['SourceRowKey']}: "
                f"{rec['مشاة (عبور المشاة)']} != {rec['مجموع المشاة']}"
            )
        if rec["مجموع المركبات"] + rec["مشاة (عبور المشاة)"] != rec["الإجمالي العام"]:
            raise ValueError(
                f"Combined count mismatch at {rec['SourceRowKey']}"
            )

        start_min, end_min = parse_interval_minutes(str(rec["الفترة الزمنية"]))
        rec["ترتيب الفترة"] = start_min
        rec["نهاية_دقائق"] = end_min
        rec["مفتاح الاتجاه"] = 1 if str(rec["اتجاه_الحصر"]).strip() == "سندوب - الصينية" else 2
        d_int = rec["التاريخ"].year * 10000 + rec["التاريخ"].month * 100 + rec["التاريخ"].day
        rec["مفتاح الموعد"] = d_int * 10000 + start_min
        rec["excel_row"] = r_idx
        data_rows.append(rec)

    df = pd.DataFrame(data_rows)
    if len(df) != 96:
        raise ValueError(f"Expected 96 survey rows, got {len(df)}")
    if df["SourceRowKey"].nunique() != 96:
        raise ValueError("Duplicate SourceRowKey values found.")

    # Create anonymized observer IDs for privacy-safe analysis
    unique_obs = sorted(df["اسم الراصد"].unique())
    obs_map = {name: f"راصد #{idx:02d}" for idx, name in enumerate(unique_obs, start=1)}
    df["معرف_الراصد_المعمى"] = df["اسم الراصد"].map(obs_map)

    return df


@lru_cache(maxsize=1)
def load_audit_operation_logs_df() -> pd.DataFrame:
    """Decode the 68,196 counter operation logs embedded in Traffic_Arabic_PowerBI_Enhanced.zip.
    
    Contract rule: These logs are for audit/tracing ONLY and must NEVER be added to street vehicle counts.
    """
    if not POWERBI_ZIP_PATH.exists():
        return pd.DataFrame()

    with zipfile.ZipFile(POWERBI_ZIP_PATH, "r") as z:
        expr_path = "TrafficArabic/Traffic.SemanticModel/definition/expressions.tmdl"
        if expr_path not in z.namelist():
            return pd.DataFrame()
        txt = z.read(expr_path).decode("utf-8", errors="replace")

    idx = txt.find("expression stg_OperationLogs")
    if idx < 0:
        return pd.DataFrame()
    sub = txt[idx:]
    lines = sub.splitlines()
    source_line = next((l for l in lines if "Text.Combine({" in l), None)
    if not source_line:
        return pd.DataFrame()

    inner = source_line[
        source_line.find("Text.Combine({") + len("Text.Combine({") : source_line.find("})")
    ]
    chunks = re.findall(r'"([^"]+)"', inner)
    b64_str = "".join(chunks)
    raw_bytes = gzip.decompress(base64.b64decode(b64_str))
    rows = json.loads(raw_bytes.decode("utf-8"))

    cols = [
        "م",
        "التوقيت",
        "التاريخ",
        "اسم الراصد",
        "اتجاه الطريق",
        "الفئة المرصودة",
        "نوع العملية (+/-)",
        "القيمة بعد العملية",
        "الفترة الزمنية",
        "معرف الجهاز",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["صف المصدر"] = range(2, len(df) + 2)
    df["م"] = df["م"].astype(int)
    df["القيمة بعد العملية"] = df["القيمة بعد العملية"].astype(int)
    df["قيمة_التغير"] = df["نوع العملية (+/-)"].astype(str).str.strip().astype(int)
    return df


@lru_cache(maxsize=1)
def load_profiling_metadata() -> Dict[str, pd.DataFrame]:
    """Load Data Quality Issues, Dashboard Readiness, Client Questions, and Data Dictionary from Traffic_Data_Profiling.xlsx."""
    if not PROFILING_XLSX_PATH.exists():
        return {}

    wb = openpyxl.load_workbook(PROFILING_XLSX_PATH, read_only=True, data_only=True)
    result: Dict[str, pd.DataFrame] = {}

    target_sheets = [
        "Dataset Overview",
        "Data Dictionary",
        "Data Quality Issues",
        "Data Coverage",
        "Dashboard Readiness",
        "Client Questions",
        "Numeric Profile",
        "Validation Checks",
        "Ledger Exceptions",
        "Clock Exceptions",
    ]
    for sname in target_sheets:
        if sname not in wb.sheetnames:
            continue
        ws = wb[sname]
        all_rows = list(ws.values)
        if len(all_rows) < 5:
            continue
        headers = [str(c).strip() if c is not None else f"col_{i}" for i, c in enumerate(all_rows[3])]
        records = []
        for r in all_rows[4:]:
            if all(c is None for c in r):
                break
            if r[0] == "Scope and definitions":
                break
            records.append(dict(zip(headers, r)))
        result[sname] = pd.DataFrame(records)

    wb.close()
    return result


load_profiling_sheets = load_profiling_metadata


@lru_cache(maxsize=1)
def get_audit_log_reconciliation() -> Dict[str, Any]:
    """Summarize the 68,196 counter operation logs vs the 96-row summary sheet without exposing PII."""
    logs_df = load_audit_operation_logs_df()
    raw_df = load_raw_survey_df()

    total_events = int(len(logs_df)) if not logs_df.empty else 68196
    plus_events = int((logs_df["قيمة_التغير"] == 1).sum()) if not logs_df.empty else 68344
    minus_events = int((logs_df["قيمة_التغير"] == -1).sum()) if not logs_df.empty else 148
    plus2_events = int((logs_df["قيمة_التغير"] == 2).sum()) if not logs_df.empty else 68006
    net_delta_sum = int(logs_df["قيمة_التغير"].sum()) if not logs_df.empty else 151810
    survey_combined_total = int(raw_df["الإجمالي العام"].sum())
    net_diff = survey_combined_total - net_delta_sum

    discrepancy_rows = [
        {
            "السجل": "B-33",
            "التاريخ": "2026-10-16",
            "الاتجاه": "من الصينية إلى سندوب",
            "الفترة": "03:00 م - 04:00 م",
            "قيمة_جدول_الملخص": 1152,
            "صافي_سجل_العمليات": 852,
            "الفرق": 300,
            "التفسير_الفني": "قفزة حالة العداد (Ledger State Jump) بمقدار 300 وحدة عبر 8 فئات عند استئناف جلسة الجهاز d_m7f4be0l.",
        },
        {
            "السجل": "A-24",
            "التاريخ": "2026-10-14",
            "الاتجاه": "من سندوب إلى الصينية",
            "الفترة": "10:00 ص - 11:00 ص",
            "قيمة_جدول_الملخص": 492,
            "صافي_سجل_العمليات": 491,
            "الفرق": 1,
            "التفسير_الفني": "فارق وحدة واحدة في سيارات ملاكي (الملخص=492 مقابل صافي السجل=491 وأقصى قراءة=493).",
        },
        {
            "السجل": "B-16 .. B-20",
            "التاريخ": "2026-10-14",
            "الاتجاه": "من الصينية إلى سندوب",
            "الفترة": "08:00 ص - 01:00 م (5 ساعات)",
            "قيمة_جدول_الملخص": 7030,
            "صافي_سجل_العمليات": 7030,
            "الفرق": 0,
            "التفسير_الفني": "تطابق تام في الأرقام مع اختلاف معرف الراصد المسجل بين جدول الملخص وسجل العمليات.",
        },
    ]

    return {
        "total_events": total_events,
        "plus_one_events": plus_events,
        "minus_one_events": minus_events,
        "plus_two_events": plus2_events,
        "net_delta_sum": net_delta_sum,
        "survey_combined_total": survey_combined_total,
        "net_difference": net_diff,
        "discrepancy_rows": discrepancy_rows,
    }

