"""Star-schema semantic model matching the Power BI reference model (Traffic.SemanticModel)."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List

import pandas as pd

from src.config import (
    DIRECTION_KEY_TO_AR,
    DIRECTION_KEY_TO_EN,
    DIRECTION_KEY_TO_SHORT_AR,
    SOURCE_VEHICLE_COLS,
    VEHICLE_CATEGORIES_SPEC,
)
from src.data_loader import load_raw_survey_df


def format_arabic_period_display(start_minute: int) -> str:
    """Replicate Power BI DimPeriod[فترة الرصد] DAX formula exactly."""
    h = start_minute // 60
    end_h = (h + 1) % 24

    def band(hour_24: int) -> str:
        if hour_24 < 12:
            return "صباحًا"
        if hour_24 < 16:
            return "ظهرًا"
        return "مساءً"

    start_band = band(h)
    end_band = band(end_h)
    h_12 = (h + 11) % 12 + 1
    end_12 = (end_h + 11) % 12 + 1
    m = start_minute % 60
    start_part = f"{h_12}:{m:02d}" + (f" {start_band}" if start_band != end_band else "")
    end_part = f"{end_12}:{m:02d} {end_band}"
    return f"{start_part}–{end_part}"


ARABIC_WEEKDAYS: List[str] = [
    "الاثنين",
    "الثلاثاء",
    "الأربعاء",
    "الخميس",
    "الجمعة",
    "السبت",
    "الأحد",
]

ENGLISH_WEEKDAYS: List[str] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


@dataclass(frozen=True)
class SemanticModel:
    """Immutable Star Schema container reconciled with Traffic.SemanticModel."""

    survey_raw: pd.DataFrame
    dim_date: pd.DataFrame
    dim_direction: pd.DataFrame
    dim_period: pd.DataFrame
    dim_vehicle_category: pd.DataFrame
    fact_vehicle_count: pd.DataFrame
    fact_pedestrian_count: pd.DataFrame
    fact_coverage: pd.DataFrame
    fact_period_summary: pd.DataFrame


@lru_cache(maxsize=1)
def build_semantic_model() -> SemanticModel:
    """Build and validate all semantic tables from the primary survey DataFrame."""
    raw = load_raw_survey_df()

    # 1. DimDate (6 rows: 2026-10-12 .. 2026-10-17)
    min_d: dt.date = raw["التاريخ"].min()
    max_d: dt.date = raw["التاريخ"].max()
    num_days = (max_d - min_d).days + 1
    date_records = []
    for i in range(num_days):
        d = min_d + dt.timedelta(days=i)
        # Power BI Date.DayOfWeek(d, Day.Saturday) + 1
        sat_order = ((d.weekday() + 2) % 7) + 1
        date_records.append(
            {
                "التاريخ": d,
                "تاريخ_نص": d.isoformat(),
                "السنة": d.year,
                "الشهر": "أكتوبر",
                "رقم الشهر": d.month,
                "الشهر والسنة": f"أكتوبر {d.year}",
                "ترتيب الشهر والسنة": d.year * 100 + d.month,
                "اليوم": ARABIC_WEEKDAYS[d.weekday()],
                "day_en": ENGLISH_WEEKDAYS[d.weekday()],
                "ترتيب اليوم": sat_order,
                "يوم الشهر": d.day,
                "وصف_اليوم": f"{d.day} أكتوبر ({ARABIC_WEEKDAYS[d.weekday()]})",
            }
        )
    dim_date = pd.DataFrame(date_records)

    # 2. DimDirection (2 rows)
    dim_direction = pd.DataFrame(
        [
            {
                "مفتاح الاتجاه": 1,
                "اتجاه الحصر": DIRECTION_KEY_TO_AR[1],
                "اتجاه_مختصر": DIRECTION_KEY_TO_SHORT_AR[1],
                "اتجاه_المصدر": "سندوب - الصينية",
                "direction_en": DIRECTION_KEY_TO_EN[1],
            },
            {
                "مفتاح الاتجاه": 2,
                "اتجاه الحصر": DIRECTION_KEY_TO_AR[2],
                "اتجاه_مختصر": DIRECTION_KEY_TO_SHORT_AR[2],
                "اتجاه_المصدر": "الصينية - سندوب",
                "direction_en": DIRECTION_KEY_TO_EN[2],
            },
        ]
    )

    # 3. DimPeriod (10 distinct periods ordered by start minute)
    period_pairs = (
        raw[["ترتيب الفترة", "الفترة الزمنية"]]
        .drop_duplicates()
        .sort_values("ترتيب الفترة")
        .reset_index(drop=True)
    )
    if len(period_pairs) != period_pairs["ترتيب الفترة"].nunique():
        raise ValueError("Multiple labels found for the same period start minute.")

    origin = dt.datetime(1899, 12, 30, 0, 0, 0)
    period_records = []
    for _, row in period_pairs.iterrows():
        p_min = int(row["ترتيب الفترة"])
        p_label = str(row["الفترة الزمنية"])
        start_dt = origin + dt.timedelta(minutes=p_min)
        end_dt = start_dt + dt.timedelta(hours=1)
        h24 = p_min // 60
        end24 = (h24 + 1) % 24
        period_records.append(
            {
                "ترتيب الفترة": p_min,
                "الفترة الزمنية": p_label,
                "بداية الفترة": start_dt,
                "نهاية الفترة": end_dt,
                "فترة الرصد": format_arabic_period_display(p_min),
                "ساعة_24": f"{h24:02d}:00–{end24:02d}:00",
                "ساعة_البدء": h24,
            }
        )
    dim_period = pd.DataFrame(period_records)

    # 4. DimVehicleCategory (8 rows)
    cat_records = []
    for spec in VEHICLE_CATEGORIES_SPEC:
        cat_records.append(
            {
                "ترتيب الفئة": int(spec["order"]),
                "اسم الفئة في المصدر": str(spec["source_col"]),
                "فئة المركبة": str(spec["name_ar"]),
                "category_en": str(spec["name_en"]),
            }
        )
    dim_vehicle_category = pd.DataFrame(cat_records)

    # 5. FactVehicleCount (96 * 8 = 768 rows, unpivoted, preserving zeros)
    vc_records = []
    cat_col_to_order: Dict[str, int] = {
        str(s["source_col"]): int(s["order"]) for s in VEHICLE_CATEGORIES_SPEC
    }
    cat_order_to_ar: Dict[int, str] = {
        int(s["order"]): str(s["name_ar"]) for s in VEHICLE_CATEGORIES_SPEC
    }
    for _, r in raw.iterrows():
        for src_col, c_ord in cat_col_to_order.items():
            vc_records.append(
                {
                    "SourceRowKey": r["SourceRowKey"],
                    "التاريخ": r["التاريخ"],
                    "مفتاح الاتجاه": int(r["مفتاح الاتجاه"]),
                    "ترتيب الفترة": int(r["ترتيب الفترة"]),
                    "مفتاح الموعد": int(r["مفتاح الموعد"]),
                    "ترتيب الفئة": c_ord,
                    "فئة المركبة": cat_order_to_ar[c_ord],
                    "اسم الفئة في المصدر": src_col,
                    "عدد المركبات": int(r[src_col]),
                }
            )
    fact_vehicle_count = pd.DataFrame(vc_records)

    # 6. FactPedestrianCount (96 rows)
    fact_pedestrian_count = raw[
        [
            "SourceRowKey",
            "التاريخ",
            "مفتاح الاتجاه",
            "ترتيب الفترة",
            "مفتاح الموعد",
            "مشاة (عبور المشاة)",
            "م",
            "اسم الراصد",
            "معرف_الراصد_المعمى",
            "مجموع المركبات",
            "مجموع المشاة",
            "الإجمالي العام",
        ]
    ].rename(columns={"مشاة (عبور المشاة)": "عدد مرات العبور"}).copy()

    # 7. FactCoverage (81 distinct date × direction × period slots)
    period_disp_map = dict(zip(dim_period["ترتيب الفترة"], dim_period["فترة الرصد"]))
    period_raw_map = dict(zip(dim_period["ترتيب الفترة"], dim_period["الفترة الزمنية"]))
    period_24_map = dict(zip(dim_period["ترتيب الفترة"], dim_period["ساعة_24"]))
    day_name_map = dict(zip(dim_date["التاريخ"], dim_date["اليوم"]))

    cov_groups = (
        raw.groupby(["التاريخ", "مفتاح الاتجاه", "ترتيب الفترة", "مفتاح الموعد"], as_index=False)
        .agg(
            عدد_السجلات=("SourceRowKey", "count"),
            مجموع_المركبات=("مجموع المركبات", "sum"),
            عدد_مرات_العبور=("مشاة (عبور المشاة)", "sum"),
            مفاتيح_الصفوف=("SourceRowKey", lambda s: "، ".join(sorted(s))),
            الراصدون_المعمون=("معرف_الراصد_المعمى", lambda s: " + ".join(sorted(set(s)))),
        )
        .sort_values(["التاريخ", "ترتيب الفترة", "مفتاح الاتجاه"])
        .reset_index(drop=True)
    )

    cov_records = []
    for _, r in cov_groups.iterrows():
        d: dt.date = r["التاريخ"]
        di = int(r["مفتاح الاتجاه"])
        p_min = int(r["ترتيب الفترة"])
        p_disp = period_disp_map[p_min]
        dir_disp = DIRECTION_KEY_TO_AR[di]
        hour_caption = f"{d.day} أكتوبر {d.year} | {p_disp} | {dir_disp}"
        cov_records.append(
            {
                "التاريخ": d,
                "تاريخ_نص": d.isoformat(),
                "اليوم": day_name_map[d],
                "مفتاح الاتجاه": di,
                "اتجاه الحصر": dir_disp,
                "اتجاه_مختصر": DIRECTION_KEY_TO_SHORT_AR[di],
                "ترتيب الفترة": p_min,
                "الفترة الزمنية": period_raw_map[p_min],
                "فترة الرصد": p_disp,
                "ساعة_24": period_24_map[p_min],
                "مفتاح الموعد": int(r["مفتاح الموعد"]),
                "عدد السجلات": int(r["عدد_السجلات"]),
                "دقائق الرصد": 60,
                "ساعة الحصر": hour_caption,
                "إجمالي_مركبات_الفترة": int(r["مجموع_المركبات"]),
                "عبور_مشاة_الفترة": int(r["عدد_مرات_العبور"]),
                "تعدد_الراصدين": int(r["عدد_السجلات"]) > 1,
                "مفاتيح_الصفوف": r["مفاتيح_الصفوف"],
                "الراصدون_المعمون": r["الراصدون_المعمون"],
            }
        )
    fact_coverage_full = pd.DataFrame(cov_records)

    # Check shared slots (observed in both directions)
    slots_dir1 = set(
        fact_coverage_full.loc[fact_coverage_full["مفتاح الاتجاه"] == 1, "مفتاح الموعد"]
    )
    slots_dir2 = set(
        fact_coverage_full.loc[fact_coverage_full["مفتاح الاتجاه"] == 2, "مفتاح الموعد"]
    )
    shared_slots = slots_dir1 & slots_dir2
    fact_coverage_full["فترة_مشتركة"] = fact_coverage_full["مفتاح الموعد"].isin(shared_slots)

    # Also add per-category sums to fact_period_summary for fast analytical queries
    cat_by_period = (
        raw.groupby(["التاريخ", "مفتاح الاتجاه", "ترتيب الفترة", "مفتاح الموعد"], as_index=False)[
            SOURCE_VEHICLE_COLS
        ].sum()
    )
    fact_period_summary = fact_coverage_full.merge(
        cat_by_period,
        on=["التاريخ", "مفتاح الاتجاه", "ترتيب الفترة", "مفتاح الموعد"],
        how="left",
    )

    fact_coverage = fact_coverage_full[
        [
            "التاريخ",
            "مفتاح الاتجاه",
            "ترتيب الفترة",
            "مفتاح الموعد",
            "عدد السجلات",
            "دقائق الرصد",
            "ساعة الحصر",
        ]
    ].copy()

    return SemanticModel(
        survey_raw=raw,
        dim_date=dim_date,
        dim_direction=dim_direction,
        dim_period=dim_period,
        dim_vehicle_category=dim_vehicle_category,
        fact_vehicle_count=fact_vehicle_count,
        fact_pedestrian_count=fact_pedestrian_count,
        fact_coverage=fact_coverage,
        fact_period_summary=fact_period_summary,
    )


get_semantic_model = build_semantic_model

