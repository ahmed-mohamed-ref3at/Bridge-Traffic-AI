"""Deterministic Python metric engine reconciled with Power BI (مؤشرات المرور.tmdl)."""
from __future__ import annotations

import datetime as dt
import math
import statistics
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd

from src.config import (
    DIRECTION_KEY_TO_AR,
    DIRECTION_KEY_TO_SHORT_AR,
    DIRECTION_SOURCE_TO_KEY,
    DISPLAY_VEHICLE_CATS_AR,
    SOURCE_VEHICLE_COLS,
    VEHICLE_CATEGORIES_SPEC,
)
from src.semantic_model import SemanticModel, build_semantic_model


@dataclass(frozen=True)
class FilterScope:
    """Normalized filter context corresponding to Power BI slicers."""

    dates: Optional[Tuple[dt.date, ...]] = None          # None = all dates
    directions: Optional[Tuple[int, ...]] = None         # None = both directions (1, 2)
    periods: Optional[Tuple[int, ...]] = None            # None = all period start minutes
    categories: Optional[Tuple[int, ...]] = None         # None = all 8 vehicle categories (1..8)
    matched_only: bool = False                           # True = restrict to common (date, period) slots

    @staticmethod
    def from_inputs(
        dates: Optional[Sequence[Union[dt.date, str]]] = None,
        directions: Optional[Sequence[Union[int, str]]] = None,
        periods: Optional[Sequence[Union[int, str]]] = None,
        categories: Optional[Sequence[Union[int, str]]] = None,
        matched_only: bool = False,
        model: Optional[SemanticModel] = None,
    ) -> "FilterScope":
        model = model or build_semantic_model()

        # Normalize dates
        norm_dates: Optional[Tuple[dt.date, ...]] = None
        if dates is not None and len(dates) > 0:
            parsed_d: List[dt.date] = []
            for d in dates:
                if isinstance(d, dt.datetime):
                    parsed_d.append(d.date())
                elif isinstance(d, dt.date):
                    parsed_d.append(d)
                else:
                    s = str(d).strip()
                    if s.isdigit() and int(s) in (12, 13, 14, 15, 16, 17):
                        parsed_d.append(dt.date(2026, 10, int(s)))
                    else:
                        parsed_d.append(dt.date.fromisoformat(s[:10]))
            norm_dates = tuple(sorted(set(parsed_d)))

        # Normalize directions
        norm_dirs: Optional[Tuple[int, ...]] = None
        if directions is not None and len(directions) > 0:
            parsed_dir: List[int] = []
            for di in directions:
                if isinstance(di, int) and di in (1, 2):
                    parsed_dir.append(di)
                else:
                    s = str(di).strip()
                    if s in ("1", "2"):
                        parsed_dir.append(int(s))
                    elif s in DIRECTION_SOURCE_TO_KEY:
                        parsed_dir.append(DIRECTION_SOURCE_TO_KEY[s])
                    elif "سندوب" in s and "الصينية" in s:
                        if s.index("سندوب") < s.index("الصينية"):
                            parsed_dir.append(1)
                        else:
                            parsed_dir.append(2)
            if parsed_dir:
                norm_dirs = tuple(sorted(set(parsed_dir)))

        # Normalize periods
        norm_periods: Optional[Tuple[int, ...]] = None
        if periods is not None and len(periods) > 0:
            p_df = model.dim_period
            label_to_min: Dict[str, int] = {}
            for _, r in p_df.iterrows():
                m_val = int(r["ترتيب الفترة"])
                label_to_min[str(r["الفترة الزمنية"]).strip()] = m_val
                label_to_min[str(r["فترة الرصد"]).strip()] = m_val
                label_to_min[str(r["ساعة_24"]).strip()] = m_val
                label_to_min[f"{r['ساعة_البدء']}:00"] = m_val
                label_to_min[f"{r['ساعة_البدء']:02d}:00"] = m_val
            parsed_p: List[int] = []
            valid_mins = set(p_df["ترتيب الفترة"].astype(int))
            for p in periods:
                if isinstance(p, int):
                    if p in valid_mins:
                        parsed_p.append(p)
                    elif 7 <= p <= 16 and (p * 60) in valid_mins:
                        parsed_p.append(p * 60)
                else:
                    s = str(p).strip()
                    if s.isdigit():
                        iv = int(s)
                        if iv in valid_mins:
                            parsed_p.append(iv)
                        elif 7 <= iv <= 16 and (iv * 60) in valid_mins:
                            parsed_p.append(iv * 60)
                    elif s in label_to_min:
                        parsed_p.append(label_to_min[s])
            if parsed_p:
                norm_periods = tuple(sorted(set(parsed_p)))

        # Normalize categories
        norm_cats: Optional[Tuple[int, ...]] = None
        if categories is not None and len(categories) > 0:
            cat_map: Dict[str, int] = {}
            for spec in VEHICLE_CATEGORIES_SPEC:
                ord_i = int(spec["order"])
                cat_map[str(spec["source_col"]).strip()] = ord_i
                cat_map[str(spec["name_ar"]).strip()] = ord_i
                cat_map[str(spec["name_en"]).strip().lower()] = ord_i
            cat_map["فئة غير مسماة"] = 8
            cat_map["غير مسماة"] = 8
            cat_map["custom"] = 8
            cat_map["ملاكي"] = 1
            cat_map["ميكروباص"] = 2
            cat_map["أتوبيسات"] = 3
            cat_map["اتوبيسات"] = 3
            cat_map["نقل خفيف"] = 4
            cat_map["سوزوكي"] = 4
            cat_map["نقل ثقيل"] = 5
            cat_map["تريلات"] = 5
            cat_map["موتوسيكلات"] = 6
            cat_map["دراجات"] = 6
            cat_map["توك توك"] = 7
            cat_map["تروسيكل"] = 7

            parsed_c: List[int] = []
            for c in categories:
                if isinstance(c, int) and 1 <= c <= 8:
                    parsed_c.append(c)
                else:
                    s = str(c).strip()
                    if s.isdigit() and 1 <= int(s) <= 8:
                        parsed_c.append(int(s))
                    elif s in cat_map:
                        parsed_c.append(cat_map[s])
                    elif s.lower() in cat_map:
                        parsed_c.append(cat_map[s.lower()])
            if parsed_c:
                norm_cats = tuple(sorted(set(parsed_c)))

        return FilterScope(
            dates=norm_dates,
            directions=norm_dirs,
            periods=norm_periods,
            categories=norm_cats,
            matched_only=matched_only,
        )

    def describe_ar(self) -> str:
        parts = []
        if self.dates:
            parts.append("التواريخ: " + "، ".join(d.isoformat() for d in self.dates))
        else:
            parts.append("التواريخ: كل الأيام (12–17 أكتوبر 2026)")

        if self.directions and len(self.directions) == 1:
            parts.append(f"الاتجاه: {DIRECTION_KEY_TO_AR[self.directions[0]]}")
        else:
            parts.append("الاتجاه: كلا الاتجاهين")

        if self.periods:
            model = build_semantic_model()
            p_map = dict(zip(model.dim_period["ترتيب الفترة"], model.dim_period["فترة الرصد"]))
            parts.append("الفترات: " + "، ".join(p_map.get(p, str(p)) for p in self.periods))
        else:
            parts.append("الفترات: جميع فترات الرصد المتاحة")

        if self.categories and len(self.categories) < 8:
            model = build_semantic_model()
            c_map = dict(
                zip(
                    model.dim_vehicle_category["ترتيب الفئة"],
                    model.dim_vehicle_category["فئة المركبة"],
                )
            )
            parts.append("فئات المركبات: " + "، ".join(c_map.get(c, str(c)) for c in self.categories))
        else:
            parts.append("فئات المركبات: جميع الفئات الثماني")

        if self.matched_only:
            parts.append("نطاق المقارنة: الفترات المشتركة بين الاتجاهين فقط")
        return " | ".join(parts)

    def describe_en(self) -> str:
        parts = []
        if self.dates:
            parts.append("Dates: " + ", ".join(d.isoformat() for d in self.dates))
        else:
            parts.append("Dates: All study days (12–17 Oct 2026)")

        if self.directions and len(self.directions) == 1:
            d_en = "Sandoub → El-Siniya" if self.directions[0] == 1 else "El-Siniya → Sandoub"
            parts.append(f"Direction: {d_en}")
        else:
            parts.append("Direction: Both directions")

        if self.periods:
            model = build_semantic_model()
            p_map = dict(zip(model.dim_period["ترتيب الفترة"], model.dim_period["الفترة الزمنية"]))
            parts.append("Periods: " + ", ".join(p_map.get(p, str(p)) for p in self.periods))
        else:
            parts.append("Periods: All observed periods")

        if self.categories and len(self.categories) < 8:
            model = build_semantic_model()
            c_map = dict(
                zip(
                    model.dim_vehicle_category["ترتيب الفئة"],
                    model.dim_vehicle_category["category_en"],
                )
            )
            parts.append("Categories: " + ", ".join(c_map.get(c, str(c)) for c in self.categories))
        else:
            parts.append("Categories: All 8 vehicle categories")

        if self.matched_only:
            parts.append("Window mode: Matched common periods only")
        return " | ".join(parts)

    def filter_fact_vehicle(self, model: Optional[SemanticModel] = None) -> pd.DataFrame:
        model = model or build_semantic_model()
        df = model.fact_vehicle_count
        if self.dates is not None:
            df = df[df["التاريخ"].isin(self.dates)]
        if self.directions is not None:
            df = df[df["مفتاح الاتجاه"].isin(self.directions)]
        if self.periods is not None:
            df = df[df["ترتيب الفترة"].isin(self.periods)]
        if self.categories is not None:
            df = df[df["ترتيب الفئة"].isin(self.categories)]
        if self.matched_only:
            ps = self.filter_period_summary(model)
            df = df[df["مفتاح الموعد"].isin(set(ps["مفتاح الموعد"]))]
        return df

    def filter_fact_coverage(self, model: Optional[SemanticModel] = None) -> pd.DataFrame:
        model = model or build_semantic_model()
        df = model.fact_coverage
        if self.dates is not None:
            df = df[df["التاريخ"].isin(self.dates)]
        if self.directions is not None:
            df = df[df["مفتاح الاتجاه"].isin(self.directions)]
        if self.periods is not None:
            df = df[df["ترتيب الفترة"].isin(self.periods)]
        return df

    def filter_period_summary(self, model: Optional[SemanticModel] = None) -> pd.DataFrame:
        return get_filtered_period_summary(self)



@lru_cache(maxsize=1)
def get_full_study_coactivity_thresholds() -> Dict[int, Tuple[float, float]]:
    """Compute Power BI PERCENTILEX.INC(..., 0.75) thresholds per direction across full study."""
    model = build_semantic_model()
    ps = model.fact_period_summary
    thresholds: Dict[int, Tuple[float, float]] = {}
    for d in (1, 2):
        sub = ps[ps["مفتاح الاتجاه"] == d]
        v_cut = statistics.quantiles(sub["إجمالي_مركبات_الفترة"].tolist(), n=4, method="inclusive")[2]
        p_cut = statistics.quantiles(sub["عبور_مشاة_الفترة"].tolist(), n=4, method="inclusive")[2]
        thresholds[d] = (float(v_cut), float(p_cut))
    return thresholds


def _filter_survey_rows(
    raw: pd.DataFrame,
    scope: FilterScope,
    ignore_category: bool = False,
    override_direction: Optional[int] = None,
) -> pd.DataFrame:
    df = raw
    if scope.dates is not None:
        df = df[df["التاريخ"].isin(scope.dates)]
    if override_direction is not None:
        df = df[df["مفتاح الاتجاه"] == override_direction]
    elif scope.directions is not None:
        df = df[df["مفتاح الاتجاه"].isin(scope.directions)]
    if scope.periods is not None:
        df = df[df["ترتيب الفترة"].isin(scope.periods)]
    if scope.matched_only:
        # Restrict to slots observed in both directions within date/period filters
        base_dp = raw
        if scope.dates is not None:
            base_dp = base_dp[base_dp["التاريخ"].isin(scope.dates)]
        if scope.periods is not None:
            base_dp = base_dp[base_dp["ترتيب الفترة"].isin(scope.periods)]
        s1 = set(base_dp.loc[base_dp["مفتاح الاتجاه"] == 1, "مفتاح الموعد"])
        s2 = set(base_dp.loc[base_dp["مفتاح الاتجاه"] == 2, "مفتاح الموعد"])
        common = s1 & s2
        df = df[df["مفتاح الموعد"].isin(common)]
    return df


def _sum_vehicles_for_rows(
    df: pd.DataFrame,
    categories: Optional[Tuple[int, ...]] = None,
) -> Optional[int]:
    if df.empty:
        return None
    if categories is None or len(categories) == 8:
        return int(df["مجموع المركبات"].sum())
    cols = [SOURCE_VEHICLE_COLS[c - 1] for c in categories if 1 <= c <= 8]
    if not cols:
        return None
    return int(df[cols].sum().sum())


def _compute_peak_for_rows(
    df: pd.DataFrame,
    categories: Optional[Tuple[int, ...]] = None,
) -> Tuple[Optional[int], Optional[str]]:
    if df.empty:
        return None, None
    grouped = df.groupby(["التاريخ", "ترتيب الفترة", "الفترة الزمنية"], as_index=False)
    slot_totals: List[Tuple[dt.date, int, str, int]] = []
    for (day, p_min, p_label), grp in grouped:
        v = _sum_vehicles_for_rows(grp, categories)
        if v is not None:
            slot_totals.append((day, int(p_min), str(p_label), v))
    if not slot_totals:
        return None, None
    max_val = max(x[3] for x in slot_totals)
    winners = sorted(
        [(x[0], x[1], x[2]) for x in slot_totals if x[3] == max_val],
        key=lambda item: (item[0], item[1]),
    )
    label_str = " ، ".join(f"{w[0].isoformat()} | {w[2]}" for w in winners)
    return max_val, label_str


def compute_all_powerbi_measures(scope: Optional[FilterScope] = None) -> Dict[str, Any]:
    """Compute all Power BI measures in مؤشرات المرور.tmdl under the given FilterScope."""
    if scope is None:
        scope = FilterScope()
    model = build_semantic_model()
    raw = model.survey_raw

    active = _filter_survey_rows(raw, scope)
    has_one_dir = (
        active["مفتاح الاتجاه"].nunique() == 1
        if not active.empty
        else (scope.directions is not None and len(scope.directions) == 1)
    )
    # Note: In Power BI, HASONEVALUE('DimDirection'[مفتاح الاتجاه]) checks DimDirection filter context
    dim_one_dir = (scope.directions is not None and len(scope.directions) == 1)
    dim_one_period = (scope.periods is not None and len(scope.periods) == 1)

    total_vehicles = _sum_vehicles_for_rows(active, scope.categories)
    all_cat_vehicles = _sum_vehicles_for_rows(active, None)

    cov_count = (
        int(active[["التاريخ", "مفتاح الاتجاه", "ترتيب الفترة"]].drop_duplicates().shape[0])
        if not active.empty
        else None
    )
    rec_count = int(len(active)) if not active.empty else None
    day_count = int(active["التاريخ"].nunique()) if not active.empty else None
    slot_count = int(active["مفتاح الموعد"].nunique()) if not active.empty else None

    ped_crossings = (
        int(active["مشاة (عبور المشاة)"].sum())
        if (not active.empty and dim_one_dir)
        else None
    )
    ped_crossings_any_scope = int(active["مشاة (عبور المشاة)"].sum()) if not active.empty else 0

    def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is not None and b is not None and b != 0:
            return float(a) / float(b)
        return None

    hourly_mean_one_dir = safe_div(total_vehicles, cov_count) if dim_one_dir else None
    hourly_mean_per_dir_period = safe_div(total_vehicles, cov_count)
    ped_hourly_mean = safe_div(ped_crossings, cov_count)
    daily_mean = safe_div(total_vehicles, day_count)
    period_across_days_mean = (
        safe_div(total_vehicles, day_count) if (dim_one_period and dim_one_dir) else None
    )
    category_share = safe_div(total_vehicles, all_cat_vehicles)

    peak_val, peak_label = _compute_peak_for_rows(active, scope.categories) if dim_one_dir else (None, None)

    # Direction-specific measures
    rs1 = active[active["مفتاح الاتجاه"] == 1]
    rs2 = active[active["مفتاح الاتجاه"] == 2]

    s1_slots = set(rs1["مفتاح الموعد"])
    s2_slots = set(rs2["مفتاح الموعد"])
    shared_slots = s1_slots & s2_slots
    shared_count = len(shared_slots) if shared_slots else None

    v1 = _sum_vehicles_for_rows(rs1, scope.categories)
    v2 = _sum_vehicles_for_rows(rs2, scope.categories)
    n1 = int(rs1[["التاريخ", "مفتاح الاتجاه", "ترتيب الفترة"]].drop_duplicates().shape[0]) or None
    n2 = int(rs2[["التاريخ", "مفتاح الاتجاه", "ترتيب الفترة"]].drop_duplicates().shape[0]) or None
    ped1 = int(rs1["مشاة (عبور المشاة)"].sum()) if not rs1.empty else None
    ped2 = int(rs2["مشاة (عبور المشاة)"].sum()) if not rs2.empty else None
    p1, l1 = _compute_peak_for_rows(rs1, scope.categories)
    p2, l2 = _compute_peak_for_rows(rs2, scope.categories)

    m1 = _sum_vehicles_for_rows(rs1[rs1["مفتاح الموعد"].isin(shared_slots)], scope.categories)
    m2 = _sum_vehicles_for_rows(rs2[rs2["مفتاح الموعد"].isin(shared_slots)], scope.categories)
    matched_diff = (m1 - m2) if (m1 is not None and m2 is not None) else None
    matched_diff_pct = safe_div(matched_diff, m2)

    # Matched pedestrians as well for fair direction comparison when requested
    mped1 = (
        int(rs1[rs1["مفتاح الموعد"].isin(shared_slots)]["مشاة (عبور المشاة)"].sum())
        if shared_slots and not rs1.empty
        else None
    )
    mped2 = (
        int(rs2[rs2["مفتاح الموعد"].isin(shared_slots)]["مشاة (عبور المشاة)"].sum())
        if shared_slots and not rs2.empty
        else None
    )

    # Top vehicle category & share within selection
    cat_indices = list(scope.categories) if scope.categories else list(range(1, 9))
    cat_totals: Dict[str, int] = {}
    cat_shares: List[float] = []
    if not active.empty and all_cat_vehicles:
        for ci in cat_indices:
            c_col = SOURCE_VEHICLE_COLS[ci - 1]
            c_name = DISPLAY_VEHICLE_CATS_AR[ci - 1]
            c_sum = int(active[c_col].sum())
            cat_totals[c_name] = c_sum
            cat_shares.append(c_sum / float(all_cat_vehicles))

    if cat_totals:
        max_cat_v = max(cat_totals.values())
        top_cats = sorted([k for k, val in cat_totals.items() if val == max_cat_v])
        top_cat_label: Optional[str] = "، ".join(top_cats)
        top_cat_share: Optional[float] = max(cat_shares) if cat_shares else None
    else:
        top_cat_label = None
        top_cat_share = None

    selection_status = (
        "لا توجد مشاهدات ضمن الاختيار"
        if not cov_count
        else f"فترات الرصد ضمن الاختيار  ·  {cov_count:,d}"
    )

    return {
        "إجمالي المركبات المسجلة": total_vehicles,
        "عدد مرات عبور المشاة": ped_crossings,
        "مرات_العبور_ضمن_النطاق": ped_crossings_any_scope,
        "عدد فترات الرصد حسب الاتجاه": cov_count,
        "عدد سجلات الحصر": rec_count,
        "عدد أيام الرصد": day_count,
        "عدد الساعات الزمنية المرصودة": slot_count,
        "متوسط عدد المركبات في الساعة": hourly_mean_one_dir,
        "متوسط المركبات لكل فترة اتجاهية": hourly_mean_per_dir_period,
        "متوسط مرات العبور في الساعة": ped_hourly_mean,
        "عدد المركبات حسب الفئة": total_vehicles,
        "نسبة فئة المركبات": category_share,
        "متوسط المركبات في يوم رصد": daily_mean,
        "متوسط المركبات للفترة عبر الأيام": period_across_days_mean,
        "أعلى عدد مركبات في ساعة مرصودة": peak_val,
        "تاريخ وفترة الذروة المرصودة": peak_label,
        "إجمالي المركبات من سندوب إلى الصينية": v1,
        "مرات عبور المشاة من سندوب إلى الصينية": ped1,
        "متوسط عدد المركبات في الساعة من سندوب إلى الصينية": safe_div(v1, n1),
        "عدد فترات الرصد من سندوب إلى الصينية": n1,
        "عدد سجلات الحصر من سندوب إلى الصينية": int(len(rs1)) if not rs1.empty else 0,
        "أعلى عدد مركبات في ساعة من سندوب إلى الصينية": p1,
        "تاريخ وفترة الذروة من سندوب إلى الصينية": l1,
        "المركبات في الفترات المشتركة من سندوب إلى الصينية": m1,
        "إجمالي المركبات من الصينية إلى سندوب": v2,
        "مرات عبور المشاة من الصينية إلى سندوب": ped2,
        "متوسط عدد المركبات في الساعة من الصينية إلى سندوب": safe_div(v2, n2),
        "عدد فترات الرصد من الصينية إلى سندوب": n2,
        "عدد سجلات الحصر من الصينية إلى سندوب": int(len(rs2)) if not rs2.empty else 0,
        "أعلى عدد مركبات في ساعة من الصينية إلى سندوب": p2,
        "تاريخ وفترة الذروة من الصينية إلى سندوب": l2,
        "المركبات في الفترات المشتركة من الصينية إلى سندوب": m2,
        "عدد الفترات المشتركة بين الاتجاهين": shared_count,
        "فرق المركبات بين الاتجاهين في الفترات المشتركة": matched_diff,
        "نسبة فرق المركبات في الفترات المشتركة": matched_diff_pct,
        "متوسط المركبات في الفترات المشتركة من سندوب إلى الصينية": safe_div(m1, shared_count),
        "متوسط المركبات في الفترات المشتركة من الصينية إلى سندوب": safe_div(m2, shared_count),
        "عبور المشاة في الفترات المشتركة من سندوب إلى الصينية": mped1,
        "عبور المشاة في الفترات المشتركة من الصينية إلى سندوب": mped2,
        "فئة المركبات الأعلى تسجيلًا": top_cat_label,
        "حصة الفئة الأعلى تسجيلًا": top_cat_share,
        "حالة اختيار الرصد": selection_status,
    }


def get_filtered_period_summary(scope: Optional[FilterScope] = None) -> pd.DataFrame:
    """Return the period-grain DataFrame (up to 81 rows) filtered by scope, with category-filtered vehicle counts."""
    if scope is None:
        scope = FilterScope()
    model = build_semantic_model()
    ps = model.fact_period_summary.copy()

    if scope.dates is not None:
        ps = ps[ps["التاريخ"].isin(scope.dates)]
    if scope.directions is not None:
        ps = ps[ps["مفتاح الاتجاه"].isin(scope.directions)]
    if scope.periods is not None:
        ps = ps[ps["ترتيب الفترة"].isin(scope.periods)]
    if scope.matched_only:
        s1 = set(ps.loc[ps["مفتاح الاتجاه"] == 1, "مفتاح الموعد"])
        s2 = set(ps.loc[ps["مفتاح الاتجاه"] == 2, "مفتاح الموعد"])
        ps = ps[ps["مفتاح الموعد"].isin(s1 & s2)]

    if scope.categories is not None and len(scope.categories) < 8:
        cols = [SOURCE_VEHICLE_COLS[c - 1] for c in scope.categories if 1 <= c <= 8]
        ps["المركبات_المختارة"] = ps[cols].sum(axis=1) if cols else 0
    else:
        ps["المركبات_المختارة"] = ps["إجمالي_مركبات_الفترة"]

    # Add co-activity flag based on full-study directional thresholds (Power BI rule)
    thresholds = get_full_study_coactivity_thresholds()

    def is_coactive(row: pd.Series) -> bool:
        d = int(row["مفتاح الاتجاه"])
        v_cut, p_cut = thresholds[d]
        return bool(
            row["إجمالي_مركبات_الفترة"] >= v_cut and row["عبور_مشاة_الفترة"] >= p_cut
        )

    if not ps.empty:
        ps["تزامن_مرتفع"] = ps.apply(is_coactive, axis=1)
    else:
        ps["تزامن_مرتفع"] = pd.Series(dtype=bool)

    return ps.reset_index(drop=True)


def get_top_peak_hours(
    scope: Optional[FilterScope] = None,
    top_n: int = 5,
    by_metric: str = "vehicles",
) -> pd.DataFrame:
    """Return top-N observed direction-hours with tie-breaking matching Power BI (Date ASC, Period ASC, Direction ASC)."""
    ps = get_filtered_period_summary(scope)
    if ps.empty:
        return pd.DataFrame()

    metric_col = "المركبات_المختارة" if by_metric == "vehicles" else "عبور_مشاة_الفترة"
    sorted_df = (
        ps.sort_values(
            by=[metric_col, "التاريخ", "ترتيب الفترة", "مفتاح الاتجاه"],
            ascending=[False, True, True, True],
        )
        .head(top_n)
        .copy()
    )
    sorted_df["الرتبة"] = range(1, len(sorted_df) + 1)
    return sorted_df[
        [
            "الرتبة",
            "التاريخ",
            "تاريخ_نص",
            "اليوم",
            "فترة الرصد",
            "الفترة الزمنية",
            "اتجاه الحصر",
            "اتجاه_مختصر",
            "المركبات_المختارة",
            "إجمالي_مركبات_الفترة",
            "عبور_مشاة_الفترة",
            "عدد السجلات",
            "فترة_مشتركة",
            "تزامن_مرتفع",
        ]
    ].reset_index(drop=True)


def get_daily_summary_table(scope: Optional[FilterScope] = None) -> pd.DataFrame:
    """Aggregate metrics by Date and Direction, showing daily totals, observed hours, and hourly averages."""
    ps = get_filtered_period_summary(scope)
    if ps.empty:
        return pd.DataFrame()

    grp = (
        ps.groupby(["التاريخ", "تاريخ_نص", "اليوم", "مفتاح الاتجاه", "اتجاه الحصر", "اتجاه_مختصر"], as_index=False)
        .agg(
            إجمالي_المركبات=("المركبات_المختارة", "sum"),
            عبور_المشاة=("عبور_مشاة_الفترة", "sum"),
            ساعات_الرصد=("ترتيب الفترة", "count"),
            سجلات_الحصر=("عدد السجلات", "sum"),
            أعلى_ساعة_مركبات=("المركبات_المختارة", "max"),
        )
        .sort_values(["التاريخ", "مفتاح الاتجاه"])
        .reset_index(drop=True)
    )
    grp["متوسط_المركبات_في_الساعة"] = grp["إجمالي_المركبات"] / grp["ساعات_الرصد"]
    grp["متوسط_عبور_المشاة_في_الساعة"] = grp["عبور_المشاة"] / grp["ساعات_الرصد"]
    return grp


def get_period_summary_table(scope: Optional[FilterScope] = None) -> pd.DataFrame:
    """Aggregate metrics by Time Period and Direction across observed days."""
    ps = get_filtered_period_summary(scope)
    if ps.empty:
        return pd.DataFrame()

    grp = (
        ps.groupby(
            ["ترتيب الفترة", "الفترة الزمنية", "فترة الرصد", "ساعة_24", "مفتاح الاتجاه", "اتجاه الحصر", "اتجاه_مختصر"],
            as_index=False,
        )
        .agg(
            إجمالي_المركبات=("المركبات_المختارة", "sum"),
            عبور_المشاة=("عبور_مشاة_الفترة", "sum"),
            عدد_أيام_الرصد=("التاريخ", "nunique"),
            فترات_الرصد=("التاريخ", "count"),
            سجلات_الحصر=("عدد السجلات", "sum"),
            أعلى_عد_في_الفترة=("المركبات_المختارة", "max"),
        )
        .sort_values(["ترتيب الفترة", "مفتاح الاتجاه"])
        .reset_index(drop=True)
    )
    grp["متوسط_المركبات_في_الساعة"] = grp["إجمالي_المركبات"] / grp["فترات_الرصد"]
    grp["متوسط_عبور_المشاة_في_الساعة"] = grp["عبور_المشاة"] / grp["فترات_الرصد"]
    return grp


def get_vehicle_category_breakdown(
    scope: Optional[FilterScope] = None,
    ignore_category_filter: bool = False,
) -> pd.DataFrame:
    """Compute counts, shares, and hourly rates by vehicle category (overall and by direction)."""
    if scope is None:
        scope = FilterScope()
    model = build_semantic_model()
    raw = model.survey_raw
    active_all_cats = _filter_survey_rows(raw, scope, ignore_category=True)
    if active_all_cats.empty:
        return pd.DataFrame()

    total_all = int(active_all_cats["مجموع المركبات"].sum())
    rs1 = active_all_cats[active_all_cats["مفتاح الاتجاه"] == 1]
    rs2 = active_all_cats[active_all_cats["مفتاح الاتجاه"] == 2]
    tot1 = int(rs1["مجموع المركبات"].sum()) if not rs1.empty else 0
    tot2 = int(rs2["مجموع المركبات"].sum()) if not rs2.empty else 0

    n_all = int(active_all_cats[["التاريخ", "مفتاح الاتجاه", "ترتيب الفترة"]].drop_duplicates().shape[0])
    n1 = int(rs1[["التاريخ", "مفتاح الاتجاه", "ترتيب الفترة"]].drop_duplicates().shape[0])
    n2 = int(rs2[["التاريخ", "مفتاح الاتجاه", "ترتيب الفترة"]].drop_duplicates().shape[0])

    selected_orders = (
        list(range(1, 9))
        if (ignore_category_filter or not scope.categories)
        else list(scope.categories)
    )

    records = []
    for spec in VEHICLE_CATEGORIES_SPEC:
        c_ord = int(spec["order"])
        if c_ord not in selected_orders:
            continue
        col = str(spec["source_col"])
        c_all = int(active_all_cats[col].sum())
        c_1 = int(rs1[col].sum()) if not rs1.empty else 0
        c_2 = int(rs2[col].sum()) if not rs2.empty else 0
        records.append(
            {
                "ترتيب الفئة": c_ord,
                "فئة المركبة": str(spec["name_ar"]),
                "اسم الفئة في المصدر": col,
                "category_en": str(spec["name_en"]),
                "إجمالي_المركبات": c_all,
                "النسبة_من_إجمالي_المركبات": (c_all / total_all * 100.0) if total_all else 0.0,
                "متوسط_الساعة_الاتجاهية": (c_all / n_all) if n_all else 0.0,
                "سندوب_إلى_الصينية": c_1,
                "حصة_سندوب_إلى_الصينية_%": (c_1 / tot1 * 100.0) if tot1 else 0.0,
                "متوسط_ساعة_سندوب_إلى_الصينية": (c_1 / n1) if n1 else 0.0,
                "الصينية_إلى_سندوب": c_2,
                "حصة_الصينية_إلى_سندوب_%": (c_2 / tot2 * 100.0) if tot2 else 0.0,
                "متوسط_ساعة_الصينية_إلى_سندوب": (c_2 / n2) if n2 else 0.0,
            }
        )
    return pd.DataFrame(records)


def get_period_category_matrix(scope: Optional[FilterScope] = None) -> pd.DataFrame:
    """Compute share of each vehicle category within each time period (matching Power BI matrix)."""
    if scope is None:
        scope = FilterScope()
    ps = get_filtered_period_summary(
        FilterScope(
            dates=scope.dates,
            directions=scope.directions,
            periods=scope.periods,
            categories=None,
            matched_only=scope.matched_only,
        )
    )
    if ps.empty:
        return pd.DataFrame()

    grp = (
        ps.groupby(["ترتيب الفترة", "فترة الرصد", "الفترة الزمنية"], as_index=False)[
            SOURCE_VEHICLE_COLS + ["إجمالي_مركبات_الفترة", "التاريخ"]
        ]
        .agg({**{c: "sum" for c in SOURCE_VEHICLE_COLS}, "إجمالي_مركبات_الفترة": "sum", "التاريخ": "count"})
        .rename(columns={"التاريخ": "عدد_الفترات_المرصودة"})
        .sort_values("ترتيب الفترة")
        .reset_index(drop=True)
    )

    records = []
    for _, r in grp.iterrows():
        p_tot = float(r["إجمالي_مركبات_الفترة"])
        n_per = int(r["عدد_الفترات_المرصودة"])
        for spec in VEHICLE_CATEGORIES_SPEC:
            c_ord = int(spec["order"])
            if scope.categories and c_ord not in scope.categories:
                continue
            col = str(spec["source_col"])
            c_val = int(r[col])
            records.append(
                {
                    "ترتيب الفترة": int(r["ترتيب الفترة"]),
                    "فترة الرصد": r["فترة الرصد"],
                    "الفترة الزمنية": r["الفترة الزمنية"],
                    "ترتيب الفئة": c_ord,
                    "فئة المركبة": str(spec["name_ar"]),
                    "عدد_المركبات": c_val,
                    "متوسط_الساعة_الاتجاهية": c_val / n_per if n_per else 0.0,
                    "الحصة_داخل_الفترة_%": (c_val / p_tot * 100.0) if p_tot else 0.0,
                    "عدد_الفترات_المرصودة": n_per,
                }
            )
    return pd.DataFrame(records)


def get_coactivity_windows_table(scope: Optional[FilterScope] = None) -> pd.DataFrame:
    """Return the simultaneous high-activity windows (both vehicles and pedestrian crossings >= P75 of direction)."""
    ps = get_filtered_period_summary(scope)
    if ps.empty:
        return pd.DataFrame()
    co = ps[ps["تزامن_مرتفع"]].sort_values(["التاريخ", "ترتيب الفترة", "مفتاح الاتجاه"]).copy()
    thresholds = get_full_study_coactivity_thresholds()
    co["حد_المركبات_مئين75"] = co["مفتاح الاتجاه"].map(lambda d: thresholds[int(d)][0])
    co["حد_المشاة_مئين75"] = co["مفتاح الاتجاه"].map(lambda d: thresholds[int(d)][1])
    return co[
        [
            "التاريخ",
            "تاريخ_نص",
            "اليوم",
            "فترة الرصد",
            "الفترة الزمنية",
            "اتجاه الحصر",
            "اتجاه_مختصر",
            "إجمالي_مركبات_الفترة",
            "عبور_مشاة_الفترة",
            "حد_المركبات_مئين75",
            "حد_المشاة_مئين75",
            "عدد السجلات",
        ]
    ].reset_index(drop=True)


def run_baseline_qa_checks() -> Dict[str, Any]:
    """Run all fixed baseline QA expectations from AGENT_TASK.md and Power BI validation.json."""
    model = build_semantic_model()
    full_m = compute_all_powerbi_measures(FilterScope())
    cat_df = get_vehicle_category_breakdown(FilterScope(), ignore_category_filter=True)
    co_df = get_coactivity_windows_table(FilterScope())
    top5_df = get_top_peak_hours(FilterScope(), top_n=5, by_metric="vehicles")
    thresholds = get_full_study_coactivity_thresholds()

    checks = [
        {
            "id": "QA-01",
            "name_ar": "إجمالي سجلات الحصر (52 اتجاه أول + 44 اتجاه عكسي)",
            "expected": "96 (52 + 44)",
            "actual": f"{full_m['عدد سجلات الحصر']} ({full_m['عدد سجلات الحصر من سندوب إلى الصينية']} + {full_m['عدد سجلات الحصر من الصينية إلى سندوب']})",
            "passed": (
                full_m["عدد سجلات الحصر"] == 96
                and full_m["عدد سجلات الحصر من سندوب إلى الصينية"] == 52
                and full_m["عدد سجلات الحصر من الصينية إلى سندوب"] == 44
            ),
        },
        {
            "id": "QA-02",
            "name_ar": "إجمالي المركبات المسجلة",
            "expected": 127769,
            "actual": full_m["إجمالي المركبات المسجلة"],
            "passed": full_m["إجمالي المركبات المسجلة"] == 127769,
        },
        {
            "id": "QA-03",
            "name_ar": "مركبات اتجاه سندوب → الصينية",
            "expected": 74248,
            "actual": full_m["إجمالي المركبات من سندوب إلى الصينية"],
            "passed": full_m["إجمالي المركبات من سندوب إلى الصينية"] == 74248,
        },
        {
            "id": "QA-04",
            "name_ar": "مركبات اتجاه الصينية → سندوب",
            "expected": 53521,
            "actual": full_m["إجمالي المركبات من الصينية إلى سندوب"],
            "passed": full_m["إجمالي المركبات من الصينية إلى سندوب"] == 53521,
        },
        {
            "id": "QA-05",
            "name_ar": "عدد فترات الرصد الاتجاهية المميزة (43 + 38)",
            "expected": "81 (43 + 38)",
            "actual": f"{full_m['عدد فترات الرصد حسب الاتجاه']} ({full_m['عدد فترات الرصد من سندوب إلى الصينية']} + {full_m['عدد فترات الرصد من الصينية إلى سندوب']})",
            "passed": (
                full_m["عدد فترات الرصد حسب الاتجاه"] == 81
                and full_m["عدد فترات الرصد من سندوب إلى الصينية"] == 43
                and full_m["عدد فترات الرصد من الصينية إلى سندوب"] == 38
            ),
        },
        {
            "id": "QA-06",
            "name_ar": "عدد الفترات المشتركة بين الاتجاهين والفرق المتطابق",
            "expected": "38 ساعة | 65,279 مقابل 53,521 | +11,758 (+21.97%)",
            "actual": (
                f"{full_m['عدد الفترات المشتركة بين الاتجاهين']} ساعة | "
                f"{full_m['المركبات في الفترات المشتركة من سندوب إلى الصينية']:,} مقابل "
                f"{full_m['المركبات في الفترات المشتركة من الصينية إلى سندوب']:,} | "
                f"+{full_m['فرق المركبات بين الاتجاهين في الفترات المشتركة']:,} "
                f"(+{full_m['نسبة فرق المركبات في الفترات المشتركة']*100:.2f}%)"
            ),
            "passed": (
                full_m["عدد الفترات المشتركة بين الاتجاهين"] == 38
                and full_m["المركبات في الفترات المشتركة من سندوب إلى الصينية"] == 65279
                and full_m["المركبات في الفترات المشتركة من الصينية إلى سندوب"] == 53521
                and full_m["فرق المركبات بين الاتجاهين في الفترات المشتركة"] == 11758
                and math.isclose(full_m["نسبة فرق المركبات في الفترات المشتركة"], 11758 / 53521, rel_tol=1e-9)
            ),
        },
        {
            "id": "QA-07",
            "name_ar": "متوسط المركبات في الساعة لكل اتجاه",
            "expected": "1,726.70 مقابل 1,408.45",
            "actual": (
                f"{full_m['متوسط عدد المركبات في الساعة من سندوب إلى الصينية']:,.2f} مقابل "
                f"{full_m['متوسط عدد المركبات في الساعة من الصينية إلى سندوب']:,.2f}"
            ),
            "passed": (
                math.isclose(full_m["متوسط عدد المركبات في الساعة من سندوب إلى الصينية"], 74248 / 43, rel_tol=1e-9)
                and math.isclose(full_m["متوسط عدد المركبات في الساعة من الصينية إلى سندوب"], 53521 / 38, rel_tol=1e-9)
            ),
        },
        {
            "id": "QA-08",
            "name_ar": "مرات عبور المشاة المسجلة لكل اتجاه (منفصلة عن المركبات)",
            "expected": "17,478 و 6,864 (ومقياس الاتجاهين معًا = None)",
            "actual": (
                f"{full_m['مرات عبور المشاة من سندوب إلى الصينية']:,} و "
                f"{full_m['مرات عبور المشاة من الصينية إلى سندوب']:,} "
                f"(معًا = {full_m['عدد مرات عبور المشاة']})"
            ),
            "passed": (
                full_m["مرات عبور المشاة من سندوب إلى الصينية"] == 17478
                and full_m["مرات عبور المشاة من الصينية إلى سندوب"] == 6864
                and full_m["عدد مرات عبور المشاة"] is None
            ),
        },
        {
            "id": "QA-09",
            "name_ar": "ساعة الذروة في كل اتجاه وأعلى 5 ساعات",
            "expected": "4,425 (14 أكتوبر 12–1 م) و 2,687 (17 أكتوبر 12–1 م)",
            "actual": (
                f"{full_m['أعلى عدد مركبات في ساعة من سندوب إلى الصينية']:,} ({full_m['تاريخ وفترة الذروة من سندوب إلى الصينية']}) و "
                f"{full_m['أعلى عدد مركبات في ساعة من الصينية إلى سندوب']:,} ({full_m['تاريخ وفترة الذروة من الصينية إلى سندوب']})"
            ),
            "passed": (
                full_m["أعلى عدد مركبات في ساعة من سندوب إلى الصينية"] == 4425
                and full_m["أعلى عدد مركبات في ساعة من الصينية إلى سندوب"] == 2687
                and top5_df["المركبات_المختارة"].tolist() == [4425, 3636, 2687, 2546, 2508]
            ),
        },
        {
            "id": "QA-10",
            "name_ar": "الفئة غير المسماة custom_1792135578939 ومجموع حصص الفئات الثماني",
            "expected": "30 مركبة (0.0235%) ومجموع الحصص = 100.00%",
            "actual": (
                f"{int(cat_df.loc[cat_df['ترتيب الفئة']==8, 'إجمالي_المركبات'].iloc[0])} مركبة "
                f"({cat_df.loc[cat_df['ترتيب الفئة']==8, 'النسبة_من_إجمالي_المركبات'].iloc[0]:.4f}%) | "
                f"المجموع = {cat_df['النسبة_من_إجمالي_المركبات'].sum():.2f}%"
            ),
            "passed": (
                int(cat_df.loc[cat_df["ترتيب الفئة"] == 8, "إجمالي_المركبات"].iloc[0]) == 30
                and math.isclose(cat_df["النسبة_من_إجمالي_المركبات"].sum(), 100.0, rel_tol=1e-9)
            ),
        },
        {
            "id": "QA-11",
            "name_ar": "ساعات التزامن المرتفع (المئين 75 للمركبات والمشاة داخل كل اتجاه)",
            "expected": "9 ساعات (4 سندوب→الصينية + 5 الصينية→سندوب)",
            "actual": (
                f"{len(co_df)} ساعات "
                f"({int((co_df['اتجاه الحصر']==DIRECTION_KEY_TO_AR[1]).sum())} + "
                f"{int((co_df['اتجاه الحصر']==DIRECTION_KEY_TO_AR[2]).sum())})"
            ),
            "passed": (
                len(co_df) == 9
                and int((co_df["اتجاه الحصر"] == DIRECTION_KEY_TO_AR[1]).sum()) == 4
                and int((co_df["اتجاه الحصر"] == DIRECTION_KEY_TO_AR[2]).sum()) == 5
                and thresholds[1] == (2113.0, 563.0)
                and thresholds[2] == (1730.25, 281.0)
            ),
        },
        {
            "id": "QA-12",
            "name_ar": "تطابق أحجام جداول النموذج الدلالي (Star Schema)",
            "expected": "FactVehicleCount=768, FactPedestrianCount=96, FactCoverage=81, DimDate=6, DimPeriod=10",
            "actual": (
                f"FactVehicleCount={len(model.fact_vehicle_count)}, "
                f"FactPedestrianCount={len(model.fact_pedestrian_count)}, "
                f"FactCoverage={len(model.fact_coverage)}, "
                f"DimDate={len(model.dim_date)}, DimPeriod={len(model.dim_period)}"
            ),
            "passed": (
                len(model.fact_vehicle_count) == 768
                and len(model.fact_pedestrian_count) == 96
                and len(model.fact_coverage) == 81
                and len(model.dim_date) == 6
                and len(model.dim_period) == 10
                and len(model.dim_direction) == 2
                and len(model.dim_vehicle_category) == 8
            ),
        },
    ]

    return {
        "all_passed": all(c["passed"] for c in checks),
        "checks": checks,
    }


def compute_kpis(
    scope: Optional[FilterScope] = None,
    model: Optional[SemanticModel] = None,
) -> Dict[str, Any]:
    """Compute standardized KPI dictionary (combining Power BI measures and shorthand keys)."""
    scope = scope or FilterScope()
    m = compute_all_powerbi_measures(scope)
    ps = get_filtered_period_summary(scope)

    total_v = int(m["إجمالي المركبات المسجلة"] or 0)
    obs_h = int(m["عدد فترات الرصد حسب الاتجاه"] or 0)
    mean_v = float(m["متوسط المركبات لكل فترة اتجاهية"] or 0.0)
    total_ped = int(m["مرات_العبور_ضمن_النطاق"] or 0)
    mean_ped = (total_ped / obs_h) if obs_h > 0 else 0.0

    # Overall peak within current scope
    if not ps.empty:
        top_row = ps.sort_values(
            ["المركبات_المختارة", "التاريخ", "ترتيب الفترة", "مفتاح الاتجاه"],
            ascending=[False, True, True, True],
        ).iloc[0]
        peak_any = int(top_row["المركبات_المختارة"])
        peak_caption_any = str(top_row["ساعة الحصر"])
    else:
        peak_any = 0
        peak_caption_any = "—"

    d1_ped = int(m["مرات عبور المشاة من سندوب إلى الصينية"] or 0)
    d2_ped = int(m["مرات عبور المشاة من الصينية إلى سندوب"] or 0)
    d1_h = int(m["عدد فترات الرصد من سندوب إلى الصينية"] or 0)
    d2_h = int(m["عدد فترات الرصد من الصينية إلى سندوب"] or 0)

    return {
        **m,
        "إجمالي_المركبات": total_v,
        "ساعات_الرصد_الفعلي": obs_h,
        "متوسط_المركبات_لكل_ساعة_رصد": mean_v,
        "عدد_صفوف_الرصد": int(m["عدد سجلات الحصر"] or 0),
        "مركبات_سندوب_إلى_الصينية": int(m["إجمالي المركبات من سندوب إلى الصينية"] or 0),
        "مركبات_الصينية_إلى_سندوب": int(m["إجمالي المركبات من الصينية إلى سندوب"] or 0),
        "ساعات_سندوب_إلى_الصينية": d1_h,
        "ساعات_الصينية_إلى_سندوب": d2_h,
        "متوسط_سندوب_إلى_الصينية": float(m["متوسط عدد المركبات في الساعة من سندوب إلى الصينية"] or 0.0),
        "متوسط_الصينية_إلى_سندوب": float(m["متوسط عدد المركبات في الساعة من الصينية إلى سندوب"] or 0.0),
        "عدد_الساعات_المشتركة": int(m["عدد الفترات المشتركة بين الاتجاهين"] or 0),
        "مركبات_سندوب_المشتركة": int(m["المركبات في الفترات المشتركة من سندوب إلى الصينية"] or 0),
        "مركبات_الصينية_المشتركة": int(m["المركبات في الفترات المشتركة من الصينية إلى سندوب"] or 0),
        "فرق_الساعات_المشتركة": int(m["فرق المركبات بين الاتجاهين في الفترات المشتركة"] or 0),
        "نسبة_فرق_الساعات_المشتركة": float((m["نسبة فرق المركبات في الفترات المشتركة"] or 0.0) * 100.0),
        "متوسط_سندوب_المشترك_لكل_ساعة": float(
            m["متوسط المركبات في الفترات المشتركة من سندوب إلى الصينية"] or 0.0
        ),
        "متوسط_الصينية_المشترك_لكل_ساعة": float(
            m["متوسط المركبات في الفترات المشتركة من الصينية إلى سندوب"] or 0.0
        ),
        "أعلى_ساعة_في_النطاق": peak_any,
        "وصف_أعلى_ساعة_في_النطاق": peak_caption_any,
        "إجمالي_عبور_المشاة": total_ped,
        "متوسط_العبور_لكل_ساعة_رصد": mean_ped,
        "عبور_سندوب_إلى_الصينية": d1_ped,
        "عبور_الصينية_إلى_سندوب": d2_ped,
        "متوسط_عبور_سندوب_لكل_ساعة": (d1_ped / d1_h) if d1_h > 0 else 0.0,
        "متوسط_عبور_الصينية_لكل_ساعة": (d2_ped / d2_h) if d2_h > 0 else 0.0,
    }


def compute_direction_comparison(
    scope: Optional[FilterScope] = None,
    model: Optional[SemanticModel] = None,
) -> pd.DataFrame:
    """Return a 2-row DataFrame comparing Direction 1 vs Direction 2 across all metrics."""
    k = compute_kpis(scope, model)
    rows = [
        {
            "مفتاح الاتجاه": 1,
            "اتجاه الحصر": DIRECTION_KEY_TO_AR[1],
            "اتجاه_مختصر": "سندوب → الصينية",
            "عدد_السجلات": int(k["عدد سجلات الحصر من سندوب إلى الصينية"] or 0),
            "ساعات_الرصد": k["ساعات_سندوب_إلى_الصينية"],
            "إجمالي_المركبات": k["مركبات_سندوب_إلى_الصينية"],
            "متوسط_المركبات_لكل_ساعة": round(k["متوسط_سندوب_إلى_الصينية"], 2),
            "المركبات_المشتركة": k["مركبات_سندوب_المشتركة"],
            "متوسط_المشتركة_لكل_ساعة": round(k["متوسط_سندوب_المشترك_لكل_ساعة"], 2),
            "ذروة_الساعة": int(k["أعلى عدد مركبات في ساعة من سندوب إلى الصينية"] or 0),
            "توقيت_الذروة": str(k["تاريخ وفترة الذروة من سندوب إلى الصينية"] or "—"),
            "إجمالي_عبور_المشاة": k["عبور_سندوب_إلى_الصينية"],
            "متوسط_المشاة_لكل_ساعة": round(k["متوسط_عبور_سندوب_لكل_ساعة"], 2),
        },
        {
            "مفتاح الاتجاه": 2,
            "اتجاه الحصر": DIRECTION_KEY_TO_AR[2],
            "اتجاه_مختصر": "الصينية → سندوب",
            "عدد_السجلات": int(k["عدد سجلات الحصر من الصينية إلى سندوب"] or 0),
            "ساعات_الرصد": k["ساعات_الصينية_إلى_سندوب"],
            "إجمالي_المركبات": k["مركبات_الصينية_إلى_سندوب"],
            "متوسط_المركبات_لكل_ساعة": round(k["متوسط_الصينية_إلى_سندوب"], 2),
            "المركبات_المشتركة": k["مركبات_الصينية_المشتركة"],
            "متوسط_المشتركة_لكل_ساعة": round(k["متوسط_الصينية_المشترك_لكل_ساعة"], 2),
            "ذروة_الساعة": int(k["أعلى عدد مركبات في ساعة من الصينية إلى سندوب"] or 0),
            "توقيت_الذروة": str(k["تاريخ وفترة الذروة من الصينية إلى سندوب"] or "—"),
            "إجمالي_عبور_المشاة": k["عبور_الصينية_إلى_سندوب"],
            "متوسط_المشاة_لكل_ساعة": round(k["متوسط_عبور_الصينية_لكل_ساعة"], 2),
        },
    ]
    return pd.DataFrame(rows)


def compute_category_breakdown(
    scope: Optional[FilterScope] = None,
    model: Optional[SemanticModel] = None,
) -> pd.DataFrame:
    return get_vehicle_category_breakdown(scope, ignore_category_filter=False)


def compute_coactivity_windows(
    scope: Optional[FilterScope] = None,
    model: Optional[SemanticModel] = None,
) -> Tuple[pd.DataFrame, Dict[int, Tuple[float, float]]]:
    return get_coactivity_windows_table(scope), get_full_study_coactivity_thresholds()


def compute_daily_summary(
    scope: Optional[FilterScope] = None,
    model: Optional[SemanticModel] = None,
) -> pd.DataFrame:
    return get_daily_summary_table(scope)


def compute_period_profile(
    scope: Optional[FilterScope] = None,
    model: Optional[SemanticModel] = None,
) -> pd.DataFrame:
    return get_period_summary_table(scope)


def compute_ranked_hours(
    scope: Optional[FilterScope] = None,
    top_n: int = 10,
    ascending: bool = False,
    model: Optional[SemanticModel] = None,
) -> pd.DataFrame:
    ps = get_filtered_period_summary(scope)
    if ps.empty:
        return pd.DataFrame()
    sorted_df = (
        ps.sort_values(
            by=["المركبات_المختارة", "التاريخ", "ترتيب الفترة", "مفتاح الاتجاه"],
            ascending=[ascending, True, True, True],
        )
        .head(top_n)
        .copy()
    )
    sorted_df["الرتبة"] = range(1, len(sorted_df) + 1)
    sorted_df["إجمالي_المركبات"] = sorted_df["المركبات_المختارة"]
    return sorted_df[
        [
            "الرتبة",
            "تاريخ_نص",
            "اليوم",
            "اتجاه الحصر",
            "فترة الرصد",
            "الفترة الزمنية",
            "إجمالي_المركبات",
            "عبور_مشاة_الفترة",
            "عدد السجلات",
            "ساعة الحصر",
            "فترة_مشتركة",
            "تزامن_مرتفع",
        ]
    ].reset_index(drop=True)

