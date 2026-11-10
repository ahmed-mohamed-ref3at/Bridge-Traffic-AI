"""Guarded, schema-validated analytical tools and NL query resolver for the AI Traffic Analyst.

Every tool validates its parameters with Pydantic against strict whitelists, executes
deterministic calculations on the validated Star Schema / MetricsEngine, and returns
structured evidence tables, chart specifications, and calculation lineage.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from src.config import (
    DIRECTION_KEY_TO_AR,
    DIRECTION_KEY_TO_EN,
    DIRECTION_KEY_TO_SHORT_AR,
    STUDY_DATES_ISO,
    VEHICLE_CATEGORIES_SPEC,
    VEHICLE_COL_TO_DISPLAY_AR,
)
from src.data_loader import (
    get_audit_log_reconciliation,
    load_profiling_sheets,
)
from src.metrics_engine import (
    FilterScope,
    compute_category_breakdown,
    compute_coactivity_windows,
    compute_daily_summary,
    compute_direction_comparison,
    compute_kpis,
    compute_period_profile,
    compute_ranked_hours,
    run_baseline_qa_checks,
)
from src.semantic_model import SemanticModel, build_semantic_model


# ============================================================================
# Whitelists & Pydantic Tool Input Schemas
# ============================================================================

VALID_GROUP_DIMENSIONS = {
    "date",
    "day_name",
    "direction",
    "period",
    "vehicle_category",
}

VALID_METRICS = {
    "total_vehicles",
    "mean_vehicles_per_hour",
    "peak_hour_vehicles",
    "min_hour_vehicles",
    "share_pct",
    "pedestrian_crossings",
    "mean_pedestrians_per_hour",
    "observed_hours",
    "survey_records",
}

FORBIDDEN_TOKENS_PATTERN = re.compile(
    r"(__import__|eval\s*\(|exec\s*\(|os\.system|subprocess|open\s*\(|DROP\s+TABLE|DELETE\s+FROM|INSERT\s+INTO)",
    re.IGNORECASE,
)


def _check_safe_string(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    if FORBIDDEN_TOKENS_PATTERN.search(val):
        raise ValueError("Forbidden expression detected in parameter.")
    return val.strip()


class ScopeInputModel(BaseModel):
    """Validated filter scope parameters."""

    dates: Optional[List[str]] = Field(
        default=None,
        description="Optional list of ISO dates within ['2026-10-12'..'2026-10-17'].",
    )
    directions: Optional[List[Union[int, str]]] = Field(
        default=None,
        description="Optional list of direction keys (1='سندوب → الصينية', 2='الصينية → سندوب') or Arabic names.",
    )
    periods: Optional[List[Union[int, str]]] = Field(
        default=None,
        description="Optional list of period start minutes (420..960) or hour labels like '12:00 PM - 01:00 PM' or '12:00'.",
    )
    categories: Optional[List[Union[int, str]]] = Field(
        default=None,
        description="Optional list of vehicle category orders (1..8) or Arabic/English category names.",
    )
    matched_only: bool = Field(
        default=False,
        description="If True, restrict analysis to common (date, period) slots observed in both directions.",
    )

    @field_validator("dates")
    @classmethod
    def validate_dates(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if not v:
            return None
        cleaned: List[str] = []
        for item in v:
            _check_safe_string(str(item))
            s = str(item).strip()
            if s not in STUDY_DATES_ISO:
                raise ValueError(
                    f"Unsupported date '{s}'. Valid study dates are: {', '.join(STUDY_DATES_ISO)}"
                )
            cleaned.append(s)
        return cleaned

    def to_filter_scope(self, model: Optional[SemanticModel] = None) -> FilterScope:
        return FilterScope.from_inputs(
            dates=self.dates,
            directions=self.directions,
            periods=self.periods,
            categories=self.categories,
            matched_only=self.matched_only,
            model=model,
        )


class AggregationQueryInput(ScopeInputModel):
    """Validated parameters for flexible traffic/pedestrian aggregations."""

    group_by: List[str] = Field(
        default_factory=lambda: ["direction"],
        description="Dimensions to group by. Allowed: 'date', 'day_name', 'direction', 'period', 'vehicle_category'.",
    )
    metric: str = Field(
        default="total_vehicles",
        description="Primary metric to sort/highlight. Allowed: 'total_vehicles', 'mean_vehicles_per_hour', 'peak_hour_vehicles', 'min_hour_vehicles', 'share_pct', 'pedestrian_crossings', 'mean_pedestrians_per_hour', 'observed_hours', 'survey_records'.",
    )
    sort_order: Literal["desc", "asc", "chronological"] = Field(
        default="desc",
        description="Sort order for the result table.",
    )
    top_n: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum rows to highlight in summary.",
    )

    @field_validator("group_by")
    @classmethod
    def validate_group_by(cls, v: List[str]) -> List[str]:
        if not v:
            return ["direction"]
        cleaned: List[str] = []
        for dim in v:
            _check_safe_string(dim)
            d = dim.strip().lower()
            if d not in VALID_GROUP_DIMENSIONS:
                raise ValueError(
                    f"Invalid group_by dimension '{dim}'. Allowed: {sorted(VALID_GROUP_DIMENSIONS)}"
                )
            if d not in cleaned:
                cleaned.append(d)
        return cleaned

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, v: str) -> str:
        _check_safe_string(v)
        m = v.strip().lower()
        if m not in VALID_METRICS:
            raise ValueError(f"Invalid metric '{v}'. Allowed: {sorted(VALID_METRICS)}")
        return m


class DirectionComparisonInput(ScopeInputModel):
    """Validated parameters for fair cross-direction comparison."""

    target: Literal["vehicles", "pedestrians", "category"] = Field(
        default="vehicles",
        description="Target subject to compare across the two directions.",
    )
    category_filter: Optional[Union[int, str]] = Field(
        default=None,
        description="Optional specific vehicle category (1..8 or name) when comparing a class across directions.",
    )


class PeaksAndAnomaliesInput(ScopeInputModel):
    """Validated parameters for peak hours, lowest hours, and statistical outlier analysis."""

    rank_order: Literal["top", "bottom"] = Field(
        default="top",
        description="Whether to rank highest ('top') or lowest ('bottom') traffic hours.",
    )
    top_n: int = Field(
        default=10,
        ge=1,
        le=81,
        description="Number of ranked direction-period hours to return.",
    )
    include_outliers: bool = Field(
        default=True,
        description="Whether to compute Tukey 1.5*IQR outlier thresholds and flagged hours.",
    )


class PedestrianCoactivityInput(ScopeInputModel):
    """Validated parameters for pedestrian crossing and vehicle co-activity analysis."""

    focus: Literal["overview", "coactivity_windows", "by_period", "by_date"] = Field(
        default="overview",
        description="Focus area of the pedestrian & co-activity analysis.",
    )


class CoverageAndQualityInput(ScopeInputModel):
    """Validated parameters for coverage matrix, multi-observer, custom class, and audit log checks."""

    aspect: Literal[
        "all",
        "coverage_gaps",
        "multi_observer",
        "unnamed_category",
        "zero_values",
        "audit_logs",
    ] = Field(
        default="all",
        description="Specific data quality or coverage aspect to inspect.",
    )


class WhatIfScenarioInput(ScopeInputModel):
    """Validated parameters for explicitly labeled hypothetical what-if scenarios."""

    scenario_type: Literal[
        "vehicle_category_change",
        "pedestrian_bridge_diversion",
        "peak_hour_shift",
    ] = Field(
        default="pedestrian_bridge_diversion",
        description="Type of hypothetical scenario to simulate.",
    )
    change_pct: float = Field(
        default=70.0,
        ge=-100.0,
        le=200.0,
        description="Percentage change or diversion rate (e.g. 70.0 means 70% diversion/reduction).",
    )
    target_category: Optional[Union[int, str]] = Field(
        default=None,
        description="Target vehicle category when scenario_type='vehicle_category_change'.",
    )


class EvidenceBoundaryInput(ScopeInputModel):
    """Validated parameters for checking unsupported questions or false premises."""

    topic: str = Field(
        default="general",
        description="Topic requested by user (e.g. 'accidents', 'speed', 'map_coordinates', 'bridge_impact', 'wrong_year', 'false_premise_direction', 'false_premise_oct16_vs_oct14', 'mixed_units').",
    )


# ============================================================================
# Standardized Tool Result Container
# ============================================================================


@dataclass
class AnalysisToolResult:
    """Standardized output from any guarded analytical tool."""

    tool_name: str
    parameters: Dict[str, Any]
    scope_used: FilterScope
    inherited_context_notes_ar: List[str] = field(default_factory=list)
    inherited_context_notes_en: List[str] = field(default_factory=list)
    headline_metrics: Dict[str, Any] = field(default_factory=dict)
    evidence_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    secondary_df: Optional[pd.DataFrame] = None
    chart_spec: Dict[str, Any] = field(default_factory=dict)
    lineage_ar: Dict[str, Any] = field(default_factory=dict)
    lineage_en: Dict[str, Any] = field(default_factory=dict)
    warnings_ar: List[str] = field(default_factory=list)
    warnings_en: List[str] = field(default_factory=list)
    is_boundary_or_refusal: bool = False
    boundary_topic: Optional[str] = None


# ============================================================================
# Tool Implementations
# ============================================================================


def tool_get_overview_and_kpis(
    params: ScopeInputModel,
    model: Optional[SemanticModel] = None,
) -> AnalysisToolResult:
    """Compute comprehensive KPIs, direction summary, top categories, and peak hours."""
    model = model or build_semantic_model()
    scope = params.to_filter_scope(model)
    kpis = compute_kpis(scope, model)
    dir_comp = compute_direction_comparison(scope, model)
    cat_df = compute_category_breakdown(scope, model)
    top_hours = compute_ranked_hours(scope, top_n=5, ascending=False, model=model)

    # Build a clean summary evidence table
    rows = [
        {
            "المؤشر": "إجمالي المركبات المرصودة",
            "Indicator": "Total Observed Vehicles",
            "سندوب → الصينية": kpis["مركبات_سندوب_إلى_الصينية"],
            "الصينية → سندوب": kpis["مركبات_الصينية_إلى_سندوب"],
            "الإجمالي / العام": kpis["إجمالي_المركبات"],
            "الوحدة": "مركبة (Vehicle)",
        },
        {
            "المؤشر": "ساعات الرصد الفعلية (فترات اتجاه-ساعة)",
            "Indicator": "Observed Direction-Hours",
            "سندوب → الصينية": kpis["ساعات_سندوب_إلى_الصينية"],
            "الصينية → سندوب": kpis["ساعات_الصينية_إلى_سندوب"],
            "الإجمالي / العام": kpis["ساعات_الرصد_الفعلي"],
            "الوحدة": "ساعة رصد (Dir-Hour)",
        },
        {
            "المؤشر": "متوسط المركبات لكل ساعة رصد",
            "Indicator": "Mean Vehicles per Observed Hour",
            "سندوب → الصينية": round(kpis["متوسط_سندوب_إلى_الصينية"], 2),
            "الصينية → سندوب": round(kpis["متوسط_الصينية_إلى_سندوب"], 2),
            "الإجمالي / العام": round(kpis["متوسط_المركبات_لكل_ساعة_رصد"], 2),
            "الوحدة": "مركبة/ساعة (Veh/Hr)",
        },
        {
            "المؤشر": "مركبات الساعات المشتركة (المتناظرة)",
            "Indicator": "Matched-Window Vehicles (Common Slots)",
            "سندوب → الصينية": kpis["مركبات_سندوب_المشتركة"],
            "الصينية → سندوب": kpis["مركبات_الصينية_المشتركة"],
            "الإجمالي / العام": kpis["مركبات_سندوب_المشتركة"] + kpis["مركبات_الصينية_المشتركة"],
            "الوحدة": f"مركبة في {kpis['عدد_الساعات_المشتركة']} ساعة مشتركة",
        },
        {
            "المؤشر": "أعلى ساعة رصد (ذروة المركبات)",
            "Indicator": "Peak Observed Hour Volume",
            "سندوب → الصينية": int(dir_comp.loc[dir_comp["مفتاح الاتجاه"] == 1, "ذروة_الساعة"].iloc[0])
            if 1 in dir_comp["مفتاح الاتجاه"].values
            else 0,
            "الصينية → سندوب": int(dir_comp.loc[dir_comp["مفتاح الاتجاه"] == 2, "ذروة_الساعة"].iloc[0])
            if 2 in dir_comp["مفتاح الاتجاه"].values
            else 0,
            "الإجمالي / العام": kpis["أعلى_ساعة_في_النطاق"],
            "الوحدة": "مركبة/ساعة (Veh/Hr)",
        },
        {
            "المؤشر": "إجمالي أحداث عبور المشاة (مؤشر مستقل)",
            "Indicator": "Total Pedestrian Crossings (Independent Metric)",
            "سندوب → الصينية": kpis["عبور_سندوب_إلى_الصينية"],
            "الصينية → سندوب": kpis["عبور_الصينية_إلى_سندوب"],
            "الإجمالي / العام": kpis["إجمالي_عبور_المشاة"],
            "الوحدة": "حدث عبور (Crossing Event)",
        },
        {
            "المؤشر": "متوسط عبور المشاة لكل ساعة رصد",
            "Indicator": "Mean Pedestrian Crossings per Hour",
            "سندوب → الصينية": round(kpis["متوسط_عبور_سندوب_لكل_ساعة"], 2),
            "الصينية → سندوب": round(kpis["متوسط_عبور_الصينية_لكل_ساعة"], 2),
            "الإجمالي / العام": round(kpis["متوسط_العبور_لكل_ساعة_رصد"], 2),
            "الوحدة": "عبور/ساعة (Crossings/Hr)",
        },
        {
            "المؤشر": "عدد سجلات الرصد الخام",
            "Indicator": "Raw Survey Records",
            "سندوب → الصينية": int(dir_comp.loc[dir_comp["مفتاح الاتجاه"] == 1, "عدد_السجلات"].iloc[0])
            if 1 in dir_comp["مفتاح الاتجاه"].values
            else 0,
            "الصينية → سندوب": int(dir_comp.loc[dir_comp["مفتاح الاتجاه"] == 2, "عدد_السجلات"].iloc[0])
            if 2 in dir_comp["مفتاح الاتجاه"].values
            else 0,
            "الإجمالي / العام": kpis["عدد_صفوف_الرصد"],
            "الوحدة": "سجل (Row)",
        },
    ]
    evidence_df = pd.DataFrame(rows)

    lineage_ar = {
        "الأداة": "get_overview_and_kpis (ملخص المؤشرات المعتمدة)",
        "النطاق_المطبق": scope.describe_ar(),
        "الجداول_المستخدمة": "FactVehicleCount (768) + FactPedestrianCount (96) + FactCoverage (81)",
        "وحدة_التحليل": "فترة اتجاه-ساعة (Direction-Period = 60 دقيقة) مع جمع سجلات تعدد الراصدين لنفس الساعة والاتجاه.",
        "المعادلة_الأساسية": "متوسط المركبات/ساعة = إجمالي المركبات ÷ عدد ساعات الرصد الفعلية (وليس عدد الصفوف الخام).",
        "قاعدة_عدم_الخلط": "يُحسب عبور المشاة كمؤشر مستقل تمامًا ولا يُجمع مع المركبات في مؤشر واحد.",
    }
    lineage_en = {
        "tool": "get_overview_and_kpis",
        "scope": scope.describe_en(),
        "tables": "FactVehicleCount (768), FactPedestrianCount (96), FactCoverage (81)",
        "grain": "Direction-Period (60-minute window); multi-observer records within the same date+direction+period are summed for counts while coverage counts once.",
        "formula": "Mean Vehicles/Hour = Total Vehicles / Observed Direction-Hours.",
    }

    return AnalysisToolResult(
        tool_name="get_overview_and_kpis",
        parameters=params.model_dump(),
        scope_used=scope,
        headline_metrics=kpis,
        evidence_df=evidence_df,
        secondary_df=top_hours,
        chart_spec={
            "chart_type": "direction_overview",
            "title_ar": "مقارنة الاتجاهين: المتوسط لكل ساعة رصد مقابل الساعات المشتركة",
            "title_en": "Direction Comparison: Hourly Mean vs Matched Windows",
            "dir_comp_df": dir_comp,
            "cat_df": cat_df,
        },
        lineage_ar=lineage_ar,
        lineage_en=lineage_en,
    )


def tool_query_traffic_aggregations(
    params: AggregationQueryInput,
    model: Optional[SemanticModel] = None,
) -> AnalysisToolResult:
    """Execute flexible multi-dimensional group-by aggregations on the semantic model."""
    model = model or build_semantic_model()
    scope = params.to_filter_scope(model)
    kpis = compute_kpis(scope, model)
    dims = params.group_by

    # Case 1: Grouping includes vehicle_category
    if "vehicle_category" in dims:
        fv = scope.filter_fact_vehicle(model).copy()
        fc = scope.filter_fact_coverage(model).copy()
        if fv.empty:
            return AnalysisToolResult(
                tool_name="query_traffic_aggregations",
                parameters=params.model_dump(),
                scope_used=scope,
                headline_metrics=kpis,
                evidence_df=pd.DataFrame(),
                warnings_ar=["لا توجد بيانات مطابقة للفلاتر المحددة."],
                warnings_en=["No records match the selected filter scope."],
            )

        # Merge dimension attributes onto FactVehicleCount
        fv = fv.merge(
            model.dim_date[["التاريخ", "تاريخ_نص", "اليوم", "day_en"]],
            on="التاريخ",
            how="left",
        )
        fv = fv.merge(
            model.dim_direction[["مفتاح الاتجاه", "اتجاه الحصر", "direction_en"]],
            on="مفتاح الاتجاه",
            how="left",
        )
        fv = fv.merge(
            model.dim_period[["ترتيب الفترة", "الفترة الزمنية", "فترة الرصد"]],
            on="ترتيب الفترة",
            how="left",
        )
        fv = fv.merge(
            model.dim_vehicle_category[
                ["ترتيب الفئة", "category_en"]
            ],
            on="ترتيب الفئة",
            how="left",
        )

        # First aggregate to direction-period-category grain so multi-observer rows are summed per hour
        period_cat = (
            fv.groupby(
                [
                    "التاريخ",
                    "تاريخ_نص",
                    "اليوم",
                    "مفتاح الاتجاه",
                    "اتجاه الحصر",
                    "ترتيب الفترة",
                    "الفترة الزمنية",
                    "فترة الرصد",
                    "ترتيب الفئة",
                    "فئة المركبة",
                    "category_en",
                    "اسم الفئة في المصدر",
                ],
                as_index=False,
            )
            .agg(
                عدد_المركبات=("عدد المركبات", "sum"),
                عدد_السجلات=("عدد المركبات", "count"),
            )
        )

        group_cols_ar: List[str] = []
        sort_cols_chrono: List[str] = []
        if "date" in dims:
            group_cols_ar.extend(["تاريخ_نص", "اليوم"])
            sort_cols_chrono.append("تاريخ_نص")
        elif "day_name" in dims:
            group_cols_ar.extend(["تاريخ_نص", "اليوم"])
            sort_cols_chrono.append("تاريخ_نص")
        if "direction" in dims:
            group_cols_ar.extend(["مفتاح الاتجاه", "اتجاه الحصر"])
            sort_cols_chrono.append("مفتاح الاتجاه")
        if "period" in dims:
            group_cols_ar.extend(["ترتيب الفترة", "فترة الرصد", "الفترة الزمنية"])
            sort_cols_chrono.append("ترتيب الفترة")
        if "vehicle_category" in dims:
            group_cols_ar.extend(["ترتيب الفئة", "فئة المركبة", "category_en"])
            sort_cols_chrono.append("ترتيب الفئة")

        grouped = (
            period_cat.groupby(group_cols_ar, as_index=False)
            .agg(
                إجمالي_المركبات=("عدد_المركبات", "sum"),
                ساعات_الرصد=("عدد_المركبات", "count"),
                متوسط_المركبات_لكل_ساعة=("عدد_المركبات", "mean"),
                أعلى_ساعة=("عدد_المركبات", "max"),
                أدنى_ساعة=("عدد_المركبات", "min"),
            )
        )
        total_v = int(grouped["إجمالي_المركبات"].sum())
        grouped["النسبة_المئوية"] = (
            grouped["إجمالي_المركبات"] / total_v * 100.0 if total_v > 0 else 0.0
        ).round(2)
        grouped["متوسط_المركبات_لكل_ساعة"] = grouped["متوسط_المركبات_لكل_ساعة"].round(2)

        metric_col_map = {
            "total_vehicles": "إجمالي_المركبات",
            "mean_vehicles_per_hour": "متوسط_المركبات_لكل_ساعة",
            "peak_hour_vehicles": "أعلى_ساعة",
            "min_hour_vehicles": "أدنى_ساعة",
            "share_pct": "النسبة_المئوية",
            "observed_hours": "ساعات_الرصد",
        }
        sort_col = metric_col_map.get(params.metric, "إجمالي_المركبات")
        if params.sort_order == "chronological" and sort_cols_chrono:
            grouped = grouped.sort_values(sort_cols_chrono).reset_index(drop=True)
        else:
            asc = params.sort_order == "asc"
            grouped = grouped.sort_values(
                [sort_col] + sort_cols_chrono, ascending=[asc] + [True] * len(sort_cols_chrono)
            ).reset_index(drop=True)

        chart_type = "category_breakdown"
        if "period" in dims and "vehicle_category" in dims:
            chart_type = "period_category_heatmap"
        elif "direction" in dims and "vehicle_category" in dims:
            chart_type = "direction_category_stacked"

        return AnalysisToolResult(
            tool_name="query_traffic_aggregations",
            parameters=params.model_dump(),
            scope_used=scope,
            headline_metrics={
                **kpis,
                "عدد_مجموعات_التجميع": len(grouped),
                "أعلى_قيمة_في_التجميع": float(grouped[sort_col].iloc[0]) if not grouped.empty else 0,
            },
            evidence_df=grouped,
            chart_spec={
                "chart_type": chart_type,
                "df": grouped,
                "dims": dims,
                "metric_col": sort_col,
                "title_ar": f"تجميع المركبات حسب ({' × '.join(dims)})",
                "title_en": f"Vehicle Aggregation by ({' × '.join(dims)})",
            },
            lineage_ar={
                "الأداة": "query_traffic_aggregations",
                "الأبعاد": " × ".join(dims),
                "المؤشر_الرئيسي": sort_col,
                "النطاق_المطبق": scope.describe_ar(),
                "الجداول_المستخدمة": "FactVehicleCount + FactCoverage + أبعاد النموذج الدلالي",
                "ملاحظة_منهجية": "تُجمع سجلات تعدد الراصدين أولاً على مستوى (التاريخ × الاتجاه × الفترة × الفئة) قبل حساب المتوسطات والقمم.",
            },
            lineage_en={
                "tool": "query_traffic_aggregations",
                "dimensions": " × ".join(dims),
                "primary_metric": sort_col,
                "scope": scope.describe_en(),
            },
        )

    # Case 2: Grouping on date / day_name / direction / period using FactPeriodSummary
    ps = scope.filter_period_summary(model).copy()
    if ps.empty:
        return AnalysisToolResult(
            tool_name="query_traffic_aggregations",
            parameters=params.model_dump(),
            scope_used=scope,
            headline_metrics=kpis,
            evidence_df=pd.DataFrame(),
            warnings_ar=["لا توجد فترات رصد مطابقة للفلاتر المحددة."],
            warnings_en=["No observed periods match the selected filter scope."],
        )

    group_cols: List[str] = []
    sort_cols_chrono = []
    if "date" in dims or "day_name" in dims:
        group_cols.extend(["تاريخ_نص", "اليوم"])
        sort_cols_chrono.append("تاريخ_نص")
    if "direction" in dims:
        group_cols.extend(["مفتاح الاتجاه", "اتجاه الحصر"])
        sort_cols_chrono.append("مفتاح الاتجاه")
    if "period" in dims:
        group_cols.extend(["ترتيب الفترة", "فترة الرصد", "الفترة الزمنية"])
        sort_cols_chrono.append("ترتيب الفترة")

    if not group_cols:
        group_cols = ["مفتاح الاتجاه", "اتجاه الحصر"]
        sort_cols_chrono = ["مفتاح الاتجاه"]

    grouped = (
        ps.groupby(group_cols, as_index=False)
        .agg(
            ساعات_الرصد=("دقائق الرصد", "count"),
            عدد_السجلات=("عدد السجلات", "sum"),
            إجمالي_المركبات=("إجمالي_مركبات_الفترة", "sum"),
            متوسط_المركبات_لكل_ساعة=("إجمالي_مركبات_الفترة", "mean"),
            أعلى_ساعة_مركبات=("إجمالي_مركبات_الفترة", "max"),
            أدنى_ساعة_مركبات=("إجمالي_مركبات_الفترة", "min"),
            إجمالي_عبور_المشاة=("عبور_مشاة_الفترة", "sum"),
            متوسط_المشاة_لكل_ساعة=("عبور_مشاة_الفترة", "mean"),
            أعلى_ساعة_مشاة=("عبور_مشاة_الفترة", "max"),
        )
    )
    total_v = int(grouped["إجمالي_المركبات"].sum())
    grouped["النسبة_من_المركبات"] = (
        grouped["إجمالي_المركبات"] / total_v * 100.0 if total_v > 0 else 0.0
    ).round(2)
    grouped["متوسط_المركبات_لكل_ساعة"] = grouped["متوسط_المركبات_لكل_ساعة"].round(2)
    grouped["متوسط_المشاة_لكل_ساعة"] = grouped["متوسط_المشاة_لكل_ساعة"].round(2)

    metric_col_map = {
        "total_vehicles": "إجمالي_المركبات",
        "mean_vehicles_per_hour": "متوسط_المركبات_لكل_ساعة",
        "peak_hour_vehicles": "أعلى_ساعة_مركبات",
        "min_hour_vehicles": "أدنى_ساعة_مركبات",
        "share_pct": "النسبة_من_المركبات",
        "pedestrian_crossings": "إجمالي_عبور_المشاة",
        "mean_pedestrians_per_hour": "متوسط_المشاة_لكل_ساعة",
        "observed_hours": "ساعات_الرصد",
        "survey_records": "عدد_السجلات",
    }
    sort_col = metric_col_map.get(params.metric, "إجمالي_المركبات")

    if params.sort_order == "chronological" and sort_cols_chrono:
        grouped = grouped.sort_values(sort_cols_chrono).reset_index(drop=True)
    else:
        asc = params.sort_order == "asc"
        grouped = grouped.sort_values(
            [sort_col] + sort_cols_chrono,
            ascending=[asc] + [True] * len(sort_cols_chrono),
        ).reset_index(drop=True)

    # Pick suitable dynamic chart type
    if dims == ["date"] or dims == ["day_name"] or set(dims) == {"date", "direction"}:
        chart_type = "daily_trend"
    elif dims == ["period"] or set(dims) == {"period", "direction"}:
        chart_type = "period_profile"
    elif set(dims) == {"date", "period"} or set(dims) == {"date", "period", "direction"}:
        chart_type = "date_period_heatmap"
    else:
        chart_type = "generic_bar"

    warnings_ar: List[str] = []
    warnings_en: List[str] = []
    if "ساعات_الرصد" in grouped.columns and grouped["ساعات_الرصد"].nunique() > 1:
        warnings_ar.append(
            "تنبيه منهجي: عدد ساعات الرصد الفعلية غير متساوٍ بين المجموعات؛ لذا يجب اعتماد «متوسط المركبات لكل ساعة رصد» عند مقارنة الكثافة المرورية."
        )
        warnings_en.append(
            "Methodological note: Observed hours vary across groups; use 'Mean Vehicles per Observed Hour' for fair intensity comparisons."
        )

    return AnalysisToolResult(
        tool_name="query_traffic_aggregations",
        parameters=params.model_dump(),
        scope_used=scope,
        headline_metrics={
            **kpis,
            "عدد_صفوف_النتيجة": len(grouped),
            "المؤشر_المختار": sort_col,
        },
        evidence_df=grouped,
        chart_spec={
            "chart_type": chart_type,
            "df": grouped,
            "dims": dims,
            "metric_col": sort_col,
            "title_ar": f"تحليل ({sort_col.replace('_', ' ')}) حسب ({' × '.join(dims)})",
            "title_en": f"Analysis of ({params.metric}) by ({' × '.join(dims)})",
        },
        lineage_ar={
            "الأداة": "query_traffic_aggregations",
            "الأبعاد": " × ".join(dims),
            "المؤشر_الأساسي": sort_col,
            "النطاق_المطبق": scope.describe_ar(),
            "الجداول_المستخدمة": "FactPeriodSummary (81 فترة اتجاه-ساعة) المشتق من FactVehicleCount + FactPedestrianCount + FactCoverage",
            "طريقة_الحساب": "يتم تجميع القيم على مستوى فترة الرصد (60 دقيقة) أولاً لضمان عدم تضخيم الساعات التي شارك فيها راصدان، ثم يُحسب المجموع والمتوسط لكل ساعة رصد فعلية.",
        },
        lineage_en={
            "tool": "query_traffic_aggregations",
            "dimensions": " × ".join(dims),
            "metric": params.metric,
            "scope": scope.describe_en(),
        },
        warnings_ar=warnings_ar,
        warnings_en=warnings_en,
    )


def tool_compare_directions_fairly(
    params: DirectionComparisonInput,
    model: Optional[SemanticModel] = None,
) -> AnalysisToolResult:
    """Compare Sandoub -> Al-Siniya vs Al-Siniya -> Sandoub fairly (raw, hourly mean, matched windows)."""
    model = model or build_semantic_model()
    # Force both directions in scope so comparison is always bilateral
    base_scope = FilterScope.from_inputs(
        dates=params.dates,
        directions=[1, 2],
        periods=params.periods,
        categories=[params.category_filter] if params.category_filter else params.categories,
        matched_only=False,
        model=model,
    )
    matched_scope = FilterScope(
        dates=base_scope.dates,
        directions=(1, 2),
        periods=base_scope.periods,
        categories=base_scope.categories,
        matched_only=True,
    )

    kpis_all = compute_kpis(base_scope, model)
    kpis_matched = compute_kpis(matched_scope, model)
    dir_df = compute_direction_comparison(base_scope, model)

    d1_hours = kpis_all["ساعات_سندوب_إلى_الصينية"]
    d2_hours = kpis_all["ساعات_الصينية_إلى_سندوب"]
    common_hours = kpis_all["عدد_الساعات_المشتركة"]

    if params.target == "pedestrians":
        d1_raw = kpis_all["عبور_سندوب_إلى_الصينية"]
        d2_raw = kpis_all["عبور_الصينية_إلى_سندوب"]
        d1_mean = kpis_all["متوسط_عبور_سندوب_لكل_ساعة"]
        d2_mean = kpis_all["متوسط_عبور_الصينية_لكل_ساعة"]
        d1_match = kpis_matched["عبور_سندوب_إلى_الصينية"]
        d2_match = kpis_matched["عبور_الصينية_إلى_سندوب"]
        d1_match_mean = d1_match / common_hours if common_hours else 0.0
        d2_match_mean = d2_match / common_hours if common_hours else 0.0
        unit_label = "حدث عبور مشاة"
    else:
        d1_raw = kpis_all["مركبات_سندوب_إلى_الصينية"]
        d2_raw = kpis_all["مركبات_الصينية_إلى_سندوب"]
        d1_mean = kpis_all["متوسط_سندوب_إلى_الصينية"]
        d2_mean = kpis_all["متوسط_الصينية_إلى_سندوب"]
        d1_match = kpis_all["مركبات_سندوب_المشتركة"]
        d2_match = kpis_all["مركبات_الصينية_المشتركة"]
        d1_match_mean = kpis_all["متوسط_سندوب_المشترك_لكل_ساعة"]
        d2_match_mean = kpis_all["متوسط_الصينية_المشترك_لكل_ساعة"]
        unit_label = "مركبة"

    raw_diff = d1_raw - d2_raw
    raw_pct = (raw_diff / d2_raw * 100.0) if d2_raw else 0.0
    mean_diff = d1_mean - d2_mean
    mean_pct = (mean_diff / d2_mean * 100.0) if d2_mean else 0.0
    match_diff = d1_match - d2_match
    match_pct = (match_diff / d2_match * 100.0) if d2_match else 0.0

    # Identify extra hours observed only in one direction
    ps_all = base_scope.filter_period_summary(model)
    slots_1 = set(ps_all.loc[ps_all["مفتاح الاتجاه"] == 1, "مفتاح الموعد"])
    slots_2 = set(ps_all.loc[ps_all["مفتاح الاتجاه"] == 2, "مفتاح الموعد"])
    only_1_slots = sorted(slots_1 - slots_2)
    only_2_slots = sorted(slots_2 - slots_1)

    unmatched_df = ps_all[ps_all["مفتاح الموعد"].isin(set(only_1_slots) | set(only_2_slots))][
        [
            "تاريخ_نص",
            "اليوم",
            "اتجاه الحصر",
            "فترة الرصد",
            "إجمالي_مركبات_الفترة",
            "عبور_مشاة_الفترة",
            "مفاتيح_الصفوف",
        ]
    ].reset_index(drop=True)

    comp_rows = [
        {
            "معيار المقارنة": "1. الإجمالي الخام (كل الساعات المرصودة)",
            "Comparison_Lens": "1. Raw Observed Total (All Hours)",
            "سندوب → الصينية": d1_raw,
            "ساعات (سندوب → الصينية)": d1_hours,
            "الصينية → سندوب": d2_raw,
            "ساعات (الصينية → سندوب)": d2_hours,
            "الفارق (سندوب - الصينية)": round(raw_diff, 2),
            "نسبة الفارق %": round(raw_pct, 2),
            "تقييم العدالة المنهجية": "يتأثر باختلاف ساعات الرصد" if d1_hours != d2_hours else "متكافئ الساعات",
        },
        {
            "معيار المقارنة": "2. المتوسط لكل ساعة رصد فعلية (توحيد المقام)",
            "Comparison_Lens": "2. Mean per Observed Hour (Normalized Rate)",
            "سندوب → الصينية": round(d1_mean, 2),
            "ساعات (سندوب → الصينية)": d1_hours,
            "الصينية → سندوب": round(d2_mean, 2),
            "ساعات (الصينية → سندوب)": d2_hours,
            "الفارق (سندوب - الصينية)": round(mean_diff, 2),
            "نسبة الفارق %": round(mean_pct, 2),
            "تقييم العدالة المنهجية": "معيار معتمد لتوحيد المقام الزمني",
        },
        {
            "معيار المقارنة": f"3. إجمالي الساعات المشتركة المتناظرة ({common_hours} ساعة)",
            "Comparison_Lens": f"3. Matched Common Windows Total ({common_hours} hrs)",
            "سندوب → الصينية": d1_match,
            "ساعات (سندوب → الصينية)": common_hours,
            "الصينية → سندوب": d2_match,
            "ساعات (الصينية → سندوب)": common_hours,
            "الفارق (سندوب - الصينية)": round(match_diff, 2),
            "نسبة الفارق %": round(match_pct, 2),
            "تقييم العدالة المنهجية": "المعيار الأدق للمقارنة المباشرة لنفس المواعيد",
        },
        {
            "معيار المقارنة": f"4. متوسط الساعة داخل الفترات المشتركة ({common_hours} ساعة)",
            "Comparison_Lens": f"4. Matched Common Windows Hourly Mean ({common_hours} hrs)",
            "سندوب → الصينية": round(d1_match_mean, 2),
            "ساعات (سندوب → الصينية)": common_hours,
            "الصينية → سندوب": round(d2_match_mean, 2),
            "ساعات (الصينية → سندوب)": common_hours,
            "الفارق (سندوب - الصينية)": round(d1_match_mean - d2_match_mean, 2),
            "نسبة الفارق %": round(match_pct, 2),
            "تقييم العدالة المنهجية": "مقارنة متطابقة المقام والتوقيت",
        },
    ]
    evidence_df = pd.DataFrame(comp_rows)

    headline = {
        **kpis_all,
        "target_unit": unit_label,
        "d1_raw": d1_raw,
        "d2_raw": d2_raw,
        "d1_hours": d1_hours,
        "d2_hours": d2_hours,
        "d1_mean": round(d1_mean, 2),
        "d2_mean": round(d2_mean, 2),
        "common_hours": common_hours,
        "d1_matched": d1_match,
        "d2_matched": d2_match,
        "matched_diff": match_diff,
        "matched_diff_pct": round(match_pct, 2),
        "unmatched_d1_hours": len(only_1_slots),
        "unmatched_d1_vehicles": int(unmatched_df["إجمالي_مركبات_الفترة"].sum())
        if not unmatched_df.empty
        else 0,
    }

    return AnalysisToolResult(
        tool_name="compare_directions_fairly",
        parameters=params.model_dump(),
        scope_used=base_scope,
        headline_metrics=headline,
        evidence_df=evidence_df,
        secondary_df=unmatched_df,
        chart_spec={
            "chart_type": "direction_overview",
            "dir_comp_df": dir_df,
            "evidence_df": evidence_df,
            "title_ar": "المقارنة المنهجية العادلة بين الاتجاهين (المتوسط الساعي والساعات الـ 38 المشتركة)",
            "title_en": "Fair Direction Comparison (Hourly Mean & Matched Windows)",
        },
        lineage_ar={
            "الأداة": "compare_directions_fairly (المقارنة العادلة بين الاتجاهين)",
            "النطاق_المطبق": base_scope.describe_ar(),
            "سبب_عدم_كفاية_الإجمالي_الخام": f"اتجاه سندوب ← الصينية رُصد في {d1_hours} ساعة مقابل {d2_hours} ساعة لاتجاه الصينية ← سندوب.",
            "الحل_المنهجي_المزدوج": (
                f"1) القسمة على ساعات الرصد الفعلية ({d1_hours} مقابل {d2_hours})، "
                f"2) الاقتصار على {common_hours} ساعة مشتركة رُصد فيها الاتجاهان معًا في نفس التاريخ ونفس الفترة."
            ),
            "الفترات_غير_المشتركة": (
                f"توجد {len(only_1_slots)} ساعات رُصدت في اتجاه سندوب ← الصينية فقط "
                "(14 أكتوبر 12–2 م [ساعتان: 4,425 + 3,636 = 8,061 مركبة] و16 أكتوبر 7–9 ص و4–5 م [3 ساعات: 908 مركبات])."
            ),
        },
        lineage_en={
            "tool": "compare_directions_fairly",
            "scope": base_scope.describe_en(),
            "methodology": f"Normalizes by observed hours ({d1_hours} vs {d2_hours}) and evaluates the {common_hours} matched (date, period) slots.",
        },
        warnings_ar=[
            f"لا يُعتمد الإجمالي الخام وحده للمقارنة لأن تغطية الرصد تختلف ({d1_hours} ساعة مقابل {d2_hours} ساعة)."
        ]
        if d1_hours != d2_hours
        else [],
        warnings_en=[
            f"Raw totals alone are not directly comparable due to unequal coverage ({d1_hours}h vs {d2_hours}h)."
        ]
        if d1_hours != d2_hours
        else [],
    )


def tool_analyze_peaks_and_anomalies(
    params: PeaksAndAnomaliesInput,
    model: Optional[SemanticModel] = None,
) -> AnalysisToolResult:
    """Analyze top/bottom hours, recurring midday peak patterns (12-1 vs 1-2 PM), and IQR outliers."""
    model = model or build_semantic_model()
    scope = params.to_filter_scope(model)
    kpis = compute_kpis(scope, model)
    ascending = params.rank_order == "bottom"
    ranked_df = compute_ranked_hours(scope, top_n=params.top_n, ascending=ascending, model=model)
    period_profile = compute_period_profile(scope, model)

    # Compute Tukey 1.5*IQR outliers on observed direction-period vehicle totals per direction
    ps = scope.filter_period_summary(model).copy()
    outlier_rows: List[Dict[str, Any]] = []
    iqr_summary: Dict[int, Dict[str, float]] = {}
    if not ps.empty and params.include_outliers:
        for di, grp in ps.groupby("مفتاح الاتجاه"):
            vals = grp["إجمالي_مركبات_الفترة"].astype(float)
            q25 = float(np.percentile(vals, 25))
            q75 = float(np.percentile(vals, 75))
            iqr = q75 - q25
            upper_fence = q75 + 1.5 * iqr
            lower_fence = max(0.0, q25 - 1.5 * iqr)
            iqr_summary[int(di)] = {
                "q25": round(q25, 2),
                "q75": round(q75, 2),
                "iqr": round(iqr, 2),
                "upper_fence": round(upper_fence, 2),
                "lower_fence": round(lower_fence, 2),
            }
            flagged = grp[
                (grp["إجمالي_مركبات_الفترة"] > upper_fence)
                | (grp["إجمالي_مركبات_الفترة"] < lower_fence)
            ]
            for _, r in flagged.iterrows():
                outlier_rows.append(
                    {
                        "التاريخ": r["تاريخ_نص"],
                        "اليوم": r["اليوم"],
                        "اتجاه الحصر": r["اتجاه الحصر"],
                        "فترة الرصد": r["فترة الرصد"],
                        "إجمالي_مركبات_الفترة": int(r["إجمالي_مركبات_الفترة"]),
                        "الحد_الأعلى_IQR": round(upper_fence, 2),
                        "نوع_القيمة": "قيمة متطرفة عليا (ذروة استثنائية)"
                        if r["إجمالي_مركبات_الفترة"] > upper_fence
                        else "قيمة منخفضة متطرفة",
                        "مفاتيح_الصفوف": r["مفاتيح_الصفوف"],
                    }
                )
    outlier_df = pd.DataFrame(outlier_rows)

    # Check 12:00-13:00 vs 13:00-14:00 pattern per direction
    p720_d1 = ps[(ps["مفتاح الاتجاه"] == 1) & (ps["ترتيب الفترة"] == 720)]
    p780_d1 = ps[(ps["مفتاح الاتجاه"] == 1) & (ps["ترتيب الفترة"] == 780)]
    p720_d2 = ps[(ps["مفتاح الاتجاه"] == 2) & (ps["ترتيب الفترة"] == 720)]
    p780_d2 = ps[(ps["مفتاح الاتجاه"] == 2) & (ps["ترتيب الفترة"] == 780)]

    headline = {
        **kpis,
        "rank_order": params.rank_order,
        "top_1_vehicles": int(ranked_df["إجمالي_المركبات"].iloc[0]) if not ranked_df.empty else 0,
        "top_1_caption": str(ranked_df["ساعة الحصر"].iloc[0]) if not ranked_df.empty else "—",
        "d1_12_13_hours": len(p720_d1),
        "d1_12_13_mean": round(float(p720_d1["إجمالي_مركبات_الفترة"].mean()), 2)
        if not p720_d1.empty
        else 0.0,
        "d1_13_14_hours": len(p780_d1),
        "d1_13_14_mean": round(float(p780_d1["إجمالي_مركبات_الفترة"].mean()), 2)
        if not p780_d1.empty
        else 0.0,
        "d2_12_13_hours": len(p720_d2),
        "d2_12_13_mean": round(float(p720_d2["إجمالي_مركبات_الفترة"].mean()), 2)
        if not p720_d2.empty
        else 0.0,
        "d2_13_14_hours": len(p780_d2),
        "d2_13_14_mean": round(float(p780_d2["إجمالي_مركبات_الفترة"].mean()), 2)
        if not p780_d2.empty
        else 0.0,
        "iqr_summary": iqr_summary,
        "outlier_count": len(outlier_df),
    }

    return AnalysisToolResult(
        tool_name="analyze_peaks_and_anomalies",
        parameters=params.model_dump(),
        scope_used=scope,
        headline_metrics=headline,
        evidence_df=ranked_df,
        secondary_df=period_profile if outlier_df.empty else outlier_df,
        chart_spec={
            "chart_type": "peaks_and_profile",
            "ranked_df": ranked_df,
            "period_profile_df": period_profile,
            "rank_order": params.rank_order,
            "title_ar": "ترتيب ساعات الذروة والنمط الزمني على مدار اليوم"
            if params.rank_order == "top"
            else "أدنى ساعات الرصد والنمط الزمني",
            "title_en": "Peak Hours Ranking & Time-of-Day Profile",
        },
        lineage_ar={
            "الأداة": "analyze_peaks_and_anomalies (تحليل القمم والأنماط والقيم المتطرفة)",
            "النطاق_المطبق": scope.describe_ar(),
            "وحدة_الترتيب": "فترة اتجاه-ساعة (81 فترة إجمالاً) بعد جمع سجلات تعدد الراصدين داخل الساعة الواحدة.",
            "التمييز_بين_القفزة_والنمط": (
                "الساعة 12:00–1:00 ظهرًا في 14 أكتوبر (سندوب ← الصينية) سجلت أعلى ساعة منفردة (4,425 مركبة) لكنها رُصدت مرة واحدة فقط في ذلك الاتجاه، "
                "بينما الساعة 1:00–2:00 ظهرًا تمثل النمط اليومي المتكرر الأعلى بمتوسط 2,141.60 مركبة/ساعة عبر 5 أيام في سندوب ← الصينية و1,663.75 عبر 4 أيام في الصينية ← سندوب."
            ),
            "معيار_القيم_المتطرفة": "قاعدة Tukey (Q3 + 1.5 × IQR) على إجمالي مركبات الساعة داخل كل اتجاه.",
        },
        lineage_en={
            "tool": "analyze_peaks_and_anomalies",
            "scope": scope.describe_en(),
            "methodology": "Ranks 60-minute direction-periods and distinguishes single-observation spikes (12-1 PM on Oct 14 = 4,425) from sustained recurring patterns (1-2 PM across 5 days = 2,141.60/hr).",
        },
    )


def tool_analyze_pedestrian_and_coactivity(
    params: PedestrianCoactivityInput,
    model: Optional[SemanticModel] = None,
) -> AnalysisToolResult:
    """Analyze pedestrian crossings independently and identify P75 high vehicle+pedestrian co-activity windows."""
    model = model or build_semantic_model()
    scope = params.to_filter_scope(model)
    kpis = compute_kpis(scope, model)
    co_df, thresholds = compute_coactivity_windows(scope, model)
    ps = scope.filter_period_summary(model).copy()

    # Descriptive Pearson and Spearman correlation per direction & overall
    correlations: Dict[str, float] = {}
    if len(ps) >= 2:
        correlations["overall_pearson"] = round(
            float(ps["إجمالي_مركبات_الفترة"].corr(ps["عبور_مشاة_الفترة"])), 3
        )
        for di in (1, 2):
            sub = ps[ps["مفتاح الاتجاه"] == di]
            if len(sub) >= 2:
                correlations[f"dir_{di}_pearson"] = round(
                    float(sub["إجمالي_مركبات_الفترة"].corr(sub["عبور_مشاة_الفترة"])), 3
                )

    # Top pedestrian hours
    top_ped = (
        ps.sort_values(
            ["عبور_مشاة_الفترة", "إجمالي_مركبات_الفترة"], ascending=[False, False]
        )
        .head(10)[
            [
                "تاريخ_نص",
                "اليوم",
                "اتجاه الحصر",
                "فترة الرصد",
                "عبور_مشاة_الفترة",
                "إجمالي_مركبات_الفترة",
                "تعدد_الراصدين",
                "مفاتيح_الصفوف",
            ]
        ]
        .reset_index(drop=True)
    )
    top_ped["الرتبة"] = range(1, len(top_ped) + 1)

    evidence_df = co_df if params.focus == "coactivity_windows" else top_ped
    secondary_df = top_ped if params.focus == "coactivity_windows" else co_df

    headline = {
        **kpis,
        "p75_dir1_vehicles": thresholds.get(1, (0.0, 0.0))[0],
        "p75_dir1_pedestrians": thresholds.get(1, (0.0, 0.0))[1],
        "p75_dir2_vehicles": thresholds.get(2, (0.0, 0.0))[0],
        "p75_dir2_pedestrians": thresholds.get(2, (0.0, 0.0))[1],
        "coactivity_windows_count": len(co_df),
        "coactivity_dir1_count": int(
            (co_df["اتجاه الحصر"] == DIRECTION_KEY_TO_AR[1]).sum()
        )
        if not co_df.empty
        else 0,
        "coactivity_dir2_count": int(
            (co_df["اتجاه الحصر"] == DIRECTION_KEY_TO_AR[2]).sum()
        )
        if not co_df.empty
        else 0,
        "correlations": correlations,
    }

    return AnalysisToolResult(
        tool_name="analyze_pedestrian_and_coactivity",
        parameters=params.model_dump(),
        scope_used=scope,
        headline_metrics=headline,
        evidence_df=evidence_df,
        secondary_df=secondary_df,
        chart_spec={
            "chart_type": "pedestrian_coactivity_scatter",
            "ps_df": ps,
            "co_df": co_df,
            "thresholds": thresholds,
            "title_ar": "مصفوفة التزامن بين كثافة المركبات وعبور المشاة (تحديد ساعات الأولوية التشغيلية)",
            "title_en": "Vehicle & Pedestrian Co-Activity Matrix (P75 Priority Windows)",
        },
        lineage_ar={
            "الأداة": "analyze_pedestrian_and_coactivity (تحليل المشاة والتزامن مع المركبات)",
            "النطاق_المطبق": scope.describe_ar(),
            "فصل_الوحدات": "المركبات والمشاة يقاسان بوحدتين مختلفتين ولا يُجمعان أبدًا في رقم واحد.",
            "تعريف_ساعات_التزامن_المرتفع": (
                f"الفترة التي تحقق أو تتجاوز المئين 75 (P75) للمركبات والمشاة معًا داخل نفس الاتجاه: "
                f"في سندوب ← الصينية (مركبات >= {thresholds.get(1, (0, 0))[0]:,.2f} ومشاة >= {thresholds.get(1, (0, 0))[1]:,.2f})، "
                f"وفي الصينية ← سندوب (مركبات >= {thresholds.get(2, (0, 0))[0]:,.2f} ومشاة >= {thresholds.get(2, (0, 0))[1]:,.2f})."
            ),
            "حدود_الدليل_المكاني": (
                "هذه المؤشرات تحدد «متى» و«في أي اتجاه» يتزامن النشاط المرتفع (التعرض الزمني)، "
                "لكنها لا تحدد نقطة العبور الجغرافية الدقيقة على الطريق ولا تقيس عدد الحوادث."
            ),
        },
        lineage_en={
            "tool": "analyze_pedestrian_and_coactivity",
            "scope": scope.describe_en(),
            "methodology": "Evaluates pedestrian crossings independently and flags direction-periods >= P75 for both vehicles and pedestrians within each direction.",
        },
    )


def tool_inspect_coverage_and_data_quality(
    params: CoverageAndQualityInput,
    model: Optional[SemanticModel] = None,
) -> AnalysisToolResult:
    """Inspect coverage gaps, 15 multi-observer periods, 30 custom_1792135578939 vehicles, zero preservation, and 68,196 audit logs."""
    model = model or build_semantic_model()
    scope = params.to_filter_scope(model)
    kpis = compute_kpis(scope, model)
    audit_rec = get_audit_log_reconciliation()
    profiling = load_profiling_sheets()

    raw = model.survey_raw.copy()
    ps = model.fact_period_summary.copy()

    # 1. Multi-observer intervals (15 direction-periods = 30 rows)
    multi_df = ps[ps["تعدد_الراصدين"]].copy()[
        [
            "تاريخ_نص",
            "اليوم",
            "اتجاه الحصر",
            "فترة الرصد",
            "عدد السجلات",
            "إجمالي_مركبات_الفترة",
            "عبور_مشاة_الفترة",
            "مفاتيح_الصفوف",
            "الراصدون_المعمون",
        ]
    ].reset_index(drop=True)

    # 2. Unnamed category custom_1792135578939 rows
    unnamed_rows = raw[raw["custom_1792135578939"] > 0][
        [
            "SourceRowKey",
            "التاريخ",
            "اتجاه_الحصر",
            "الفترة الزمنية",
            "custom_1792135578939",
            "مجموع المركبات",
            "معرف_الراصد_المعمى",
        ]
    ].copy()
    unnamed_rows["التاريخ"] = unnamed_rows["التاريخ"].astype(str)
    unnamed_rows = unnamed_rows.reset_index(drop=True)

    # 3. Coverage by date and direction
    daily_cov = compute_daily_summary(scope, model)

    # 4. Zero counts per column
    count_cols = [str(c["source_col"]) for c in VEHICLE_CATEGORIES_SPEC] + [
        "مشاة (عبور المشاة)"
    ]
    zero_summary_rows = []
    for c in count_cols:
        z_cnt = int((raw[c] == 0).sum())
        zero_summary_rows.append(
            {
                "العمود": VEHICLE_COL_TO_DISPLAY_AR.get(c, c),
                "العمود_المصدري": c,
                "عدد_الخلايا_الصفرية_من_96": z_cnt,
                "عدد_الخلايا_الموجبة": int(96 - z_cnt),
                "إجمالي_العمود": int(raw[c].sum()),
                "الحالة": "محفوظة كأصفار رصد صحيحة (لم تُحذف ولم تُستبدل)",
            }
        )
    zero_df = pd.DataFrame(zero_summary_rows)

    # Select primary evidence_df based on requested aspect
    if params.aspect == "multi_observer":
        evidence_df = multi_df
        secondary_df = daily_cov
    elif params.aspect == "unnamed_category":
        evidence_df = unnamed_rows
        secondary_df = zero_df
    elif params.aspect == "zero_values":
        evidence_df = zero_df
        secondary_df = unnamed_rows
    elif params.aspect == "audit_logs":
        evidence_df = pd.DataFrame(audit_rec.get("discrepancy_rows", []))
        secondary_df = multi_df
    else:
        evidence_df = daily_cov
        secondary_df = multi_df

    headline = {
        **kpis,
        "total_possible_slots_6x10x2": 120,
        "observed_direction_periods": 81,
        "unobserved_direction_periods": 39,
        "distinct_calendar_date_periods": 43,
        "matched_common_periods": 38,
        "multi_observer_periods": len(multi_df),
        "multi_observer_raw_rows": int(multi_df["عدد السجلات"].sum()),
        "unnamed_category_total": int(raw["custom_1792135578939"].sum()),
        "unnamed_category_rows_count": len(unnamed_rows),
        "total_zero_cells_across_9_cols": int(sum(r["عدد_الخلايا_الصفرية_من_96"] for r in zero_summary_rows)),
        "audit_logs_total_events": audit_rec.get("total_events", 68196),
        "audit_logs_net_delta_sum": audit_rec.get("net_delta_sum", 151810),
        "combined_sheet_grand_total": audit_rec.get("survey_combined_total", 152111),
        "audit_vs_sheet_diff": audit_rec.get("net_difference", 301),
    }

    return AnalysisToolResult(
        tool_name="inspect_coverage_and_data_quality",
        parameters=params.model_dump(),
        scope_used=scope,
        headline_metrics=headline,
        evidence_df=evidence_df,
        secondary_df=secondary_df,
        chart_spec={
            "chart_type": "coverage_matrix",
            "daily_cov_df": daily_cov,
            "ps_df": ps,
            "title_ar": "خريطة التغطية الزمنية وجودة البيانات (81 ساعة مرصودة من أصل 120 ساعة محتملة)",
            "title_en": "Temporal Coverage & Data Quality Matrix (81 of 120 Possible Slots)",
        },
        lineage_ar={
            "الأداة": "inspect_coverage_and_data_quality (فحص التغطية وجودة البيانات وسجلات التدقيق)",
            "التغطية_الزمنية": "81 فترة اتجاه-ساعة مرصودة (43 في سندوب ← الصينية و38 في الصينية ← سندوب) من أصل 120 فترة محتملة (6 أيام × 10 ساعات × اتجاهين).",
            "تعدد_الراصدين": "15 فترة اتجاه-ساعة ضمت سجلين (30 سجلًا خامًا: 9 مجموعات/18 سجلًا في سندوب ← الصينية و6 مجموعات/12 سجلًا في الصينية ← سندوب)؛ تُجمع أعدادها ولا تُحذف باعتبارها تكرارًا.",
            "الفئة_غير_المسماة": "العمود custom_1792135578939 يحتوي على 30 مركبة (0.0235%) كلها يوم 16 أكتوبر 2026 في اتجاه الصينية ← سندوب عبر 5 سجلات (B-29 إلى B-33: القيم 5 و4 و3 و11 و7).",
            "سجل_عمليات_العداد": "ملف AuditOperationLogs يضم 68,196 عملية نقر بصافي مجموع (delta = 151,810) مقابل 152,111 في جدول الملخص (فارق 301 يفسره 300 في السجل B-33 نتيجة قفزة حالة العداد عبر 8 فئات + 1 في السجل A-24 حيث الملخص=492 والسجل=491).",
            "حماية_الخصوصية": "تم تعمية أسماء الراصدين إلى معرفات تحليلية (راصد #01 .. راصد #15) التزامًا بقواعد الخصوصية.",
        },
        lineage_en={
            "tool": "inspect_coverage_and_data_quality",
            "scope": scope.describe_en(),
            "methodology": "Verifies 81/120 coverage slots, 15 multi-observer periods (30 rows), 30 vehicles in custom_1792135578939, 53 zero cells preserved, and 68,196 audit log events.",
        },
    )


def tool_evaluate_what_if_scenario(
    params: WhatIfScenarioInput,
    model: Optional[SemanticModel] = None,
) -> AnalysisToolResult:
    """Evaluate a clearly-labeled hypothetical 'What-If' scenario based on observed baseline metrics."""
    model = model or build_semantic_model()
    scope = params.to_filter_scope(model)
    kpis = compute_kpis(scope, model)
    co_df, thresholds = compute_coactivity_windows(scope, model)

    if params.scenario_type == "pedestrian_bridge_diversion":
        diversion_rate = max(0.0, min(100.0, params.change_pct)) / 100.0
        base_total_ped = kpis["إجمالي_عبور_المشاة"]
        base_d1_ped = kpis["عبور_سندوب_إلى_الصينية"]
        base_d2_ped = kpis["عبور_الصينية_إلى_سندوب"]
        co_ped = int(co_df["عبور_مشاة_الفترة"].sum()) if not co_df.empty else 0

        diverted_total = round(base_total_ped * diversion_rate)
        remaining_surface = base_total_ped - diverted_total
        diverted_co = round(co_ped * diversion_rate)
        remaining_co = co_ped - diverted_co

        rows = [
            {
                "البند": "إجمالي عبور المشاة في النطاق",
                "الوضع_التاريخي_المرصود": base_total_ped,
                "المتحول_إلى_الكوبري_افتراضيًا": diverted_total,
                "المتبقي_على_السطح_افتراضيًا": remaining_surface,
                "نسبة_التحويل_الافتراضية": f"{params.change_pct:.1f}%",
                "نوع_الرقم": "محاكاة افتراضية (What-If)",
            },
            {
                "البند": "عبور المشاة في سندوب → الصينية",
                "الوضع_التاريخي_المرصود": base_d1_ped,
                "المتحول_إلى_الكوبري_افتراضيًا": round(base_d1_ped * diversion_rate),
                "المتبقي_على_السطح_افتراضيًا": base_d1_ped - round(base_d1_ped * diversion_rate),
                "نسبة_التحويل_الافتراضية": f"{params.change_pct:.1f}%",
                "نوع_الرقم": "محاكاة افتراضية (What-If)",
            },
            {
                "البند": "عبور المشاة في الصينية → سندوب",
                "الوضع_التاريخي_المرصود": base_d2_ped,
                "المتحول_إلى_الكوبري_افتراضيًا": round(base_d2_ped * diversion_rate),
                "المتبقي_على_السطح_افتراضيًا": base_d2_ped - round(base_d2_ped * diversion_rate),
                "نسبة_التحويل_الافتراضية": f"{params.change_pct:.1f}%",
                "نوع_الرقم": "محاكاة افتراضية (What-If)",
            },
            {
                "البند": f"العبور خلال ساعات التزامن المرتفع ({len(co_df)} ساعات أولوية)",
                "الوضع_التاريخي_المرصود": co_ped,
                "المتحول_إلى_الكوبري_افتراضيًا": diverted_co,
                "المتبقي_على_السطح_افتراضيًا": remaining_co,
                "نسبة_التحويل_الافتراضية": f"{params.change_pct:.1f}%",
                "نوع_الرقم": "محاكاة افتراضية (What-If)",
            },
        ]
        evidence_df = pd.DataFrame(rows)
        headline = {
            **kpis,
            "scenario_type": "pedestrian_bridge_diversion",
            "assumed_rate_pct": params.change_pct,
            "observed_pedestrians": base_total_ped,
            "hypothetical_diverted_pedestrians": diverted_total,
            "hypothetical_remaining_surface": remaining_surface,
            "priority_windows_observed_ped": co_ped,
            "priority_windows_diverted_ped": diverted_co,
        }
    else:
        # Vehicle category change scenario
        cat_df = compute_category_breakdown(scope, model)
        target_ord = 5  # Default Heavy Trucks if not specified
        if params.target_category:
            tmp_scope = FilterScope.from_inputs(categories=[params.target_category], model=model)
            if tmp_scope.categories:
                target_ord = tmp_scope.categories[0]
        row_cat = cat_df[cat_df["ترتيب الفئة"] == target_ord]
        cat_name = str(row_cat["فئة المركبة"].iloc[0]) if not row_cat.empty else "نقل ثقيل وتريلات"
        base_cat_v = int(row_cat["إجمالي_المركبات"].iloc[0]) if not row_cat.empty else 0
        base_total_v = kpis["إجمالي_المركبات"]
        hours = kpis["ساعات_الرصد_الفعلي"]

        delta_v = round(base_cat_v * (params.change_pct / 100.0))
        new_cat_v = max(0, base_cat_v + delta_v)
        new_total_v = max(0, base_total_v + (new_cat_v - base_cat_v))
        new_mean_v = new_total_v / hours if hours else 0.0

        evidence_df = pd.DataFrame(
            [
                {
                    "المؤشر": f"حجم فئة ({cat_name})",
                    "القيمة_التاريخية_المرصودة": base_cat_v,
                    "القيمة_الافتراضية_بعد_التغيير": new_cat_v,
                    "الفرق_الافتراضي": new_cat_v - base_cat_v,
                    "نسبة_التغيير_المفترضة": f"{params.change_pct:+.1f}%",
                },
                {
                    "المؤشر": "إجمالي المركبات في النطاق",
                    "القيمة_التاريخية_المرصودة": base_total_v,
                    "القيمة_الافتراضية_بعد_التغيير": new_total_v,
                    "الفرق_الافتراضي": new_total_v - base_total_v,
                    "نسبة_التغيير_المفترضة": f"{(new_total_v - base_total_v) / base_total_v * 100.0 if base_total_v else 0:+.2f}%",
                },
                {
                    "المؤشر": "متوسط المركبات لكل ساعة رصد",
                    "القيمة_التاريخية_المرصودة": round(kpis["متوسط_المركبات_لكل_ساعة_رصد"], 2),
                    "القيمة_الافتراضية_بعد_التغيير": round(new_mean_v, 2),
                    "الفرق_الافتراضي": round(new_mean_v - kpis["متوسط_المركبات_لكل_ساعة_رصد"], 2),
                    "نسبة_التغيير_المفترضة": f"{(new_total_v - base_total_v) / base_total_v * 100.0 if base_total_v else 0:+.2f}%",
                },
            ]
        )
        headline = {
            **kpis,
            "scenario_type": "vehicle_category_change",
            "target_category": cat_name,
            "assumed_change_pct": params.change_pct,
            "observed_category_vehicles": base_cat_v,
            "hypothetical_category_vehicles": new_cat_v,
            "observed_total_vehicles": base_total_v,
            "hypothetical_total_vehicles": new_total_v,
        }

    return AnalysisToolResult(
        tool_name="evaluate_what_if_scenario",
        parameters=params.model_dump(),
        scope_used=scope,
        headline_metrics=headline,
        evidence_df=evidence_df,
        chart_spec={
            "chart_type": "what_if_comparison",
            "df": evidence_df,
            "title_ar": "مقارنة الوضع التاريخي المرصود مقابل السيناريو الافتراضي (What-If)",
            "title_en": "Observed Baseline vs Hypothetical What-If Scenario",
        },
        lineage_ar={
            "الأداة": "evaluate_what_if_scenario (محاكاة سيناريو افتراضي)",
            "تنبيه_إلزامي": "الأرقام الافتراضية في هذا الجدول هي محاكاة حسابية بناءً على افتراض المستخدم وليست رصدًا ميدانيًا فعليًا.",
            "النطاق_المطبق": scope.describe_ar(),
        },
        lineage_en={
            "tool": "evaluate_what_if_scenario",
            "scope": scope.describe_en(),
            "note": "Explicitly separates observed historical counts from user-specified hypothetical simulation values.",
        },
        warnings_ar=[
            "تنبيه: النتائج المحاكاة هي سيناريو افتراضي (What-If)؛ البيانات الميدانية الحالية لا تفصل بين العبور فوق الكوبري والعبور السطحي."
        ],
        warnings_en=[
            "Note: Simulated values are hypothetical (What-If); the current dataset does not separate bridge vs surface crossings."
        ],
    )


def tool_check_evidence_boundaries(
    params: EvidenceBoundaryInput,
    model: Optional[SemanticModel] = None,
) -> AnalysisToolResult:
    """Handle unsupported questions (accidents, speed, maps, bridge impact, wrong year, mixed units, false premises) with closest verified evidence."""
    model = model or build_semantic_model()
    scope = params.to_filter_scope(model)
    kpis = compute_kpis(scope, model)
    co_df, thresholds = compute_coactivity_windows(scope, model)
    dir_df = compute_direction_comparison(scope, model)

    topic = params.topic

    # Build closest verified evidence table depending on the boundary topic
    if topic in ("false_premise_oct16_vs_oct14", "coverage_vs_intensity"):
        ps = model.fact_period_summary
        d1_14 = ps[(ps["مفتاح الاتجاه"] == 1) & (ps["تاريخ_نص"] == "2026-10-14")]
        d1_16 = ps[(ps["مفتاح الاتجاه"] == 1) & (ps["تاريخ_نص"] == "2026-10-16")]
        d2_14 = ps[(ps["مفتاح الاتجاه"] == 2) & (ps["تاريخ_نص"] == "2026-10-14")]
        d2_16 = ps[(ps["مفتاح الاتجاه"] == 2) & (ps["تاريخ_نص"] == "2026-10-16")]
        evidence_df = pd.DataFrame(
            [
                {
                    "التاريخ": "2026-10-14 (الأربعاء)",
                    "الاتجاه": DIRECTION_KEY_TO_AR[1],
                    "ساعات_الرصد_الفعلية": len(d1_14),
                    "إجمالي_المركبات": int(d1_14["إجمالي_مركبات_الفترة"].sum()),
                    "متوسط_المركبات_لكل_ساعة": round(float(d1_14["إجمالي_مركبات_الفترة"].mean()), 2),
                    "التفسير_المنهجي": "ساعات رصد أقل (7 ساعات) لكن كثافة مرورية مرتفعة جدًا (2,543.43 مركبة/ساعة)",
                },
                {
                    "التاريخ": "2026-10-16 (الجمعة)",
                    "الاتجاه": DIRECTION_KEY_TO_AR[1],
                    "ساعات_الرصد_الفعلية": len(d1_16),
                    "إجمالي_المركبات": int(d1_16["إجمالي_مركبات_الفترة"].sum()),
                    "متوسط_المركبات_لكل_ساعة": round(float(d1_16["إجمالي_مركبات_الفترة"].mean()), 2),
                    "التفسير_المنهجي": "أطول تغطية رصد (9 ساعات) لكن أقل كثافة مرورية (777.89 مركبة/ساعة)",
                },
                {
                    "التاريخ": "2026-10-14 (الأربعاء)",
                    "الاتجاه": DIRECTION_KEY_TO_AR[2],
                    "ساعات_الرصد_الفعلية": len(d2_14),
                    "إجمالي_المركبات": int(d2_14["إجمالي_مركبات_الفترة"].sum()),
                    "متوسط_المركبات_لكل_ساعة": round(float(d2_14["إجمالي_مركبات_الفترة"].mean()), 2),
                    "التفسير_المنهجي": "5 ساعات رصد بمتوسط 1,406.00 مركبة/ساعة",
                },
                {
                    "التاريخ": "2026-10-16 (الجمعة)",
                    "الاتجاه": DIRECTION_KEY_TO_AR[2],
                    "ساعات_الرصد_الفعلية": len(d2_16),
                    "إجمالي_المركبات": int(d2_16["إجمالي_مركبات_الفترة"].sum()),
                    "متوسط_المركبات_لكل_ساعة": round(float(d2_16["إجمالي_مركبات_الفترة"].mean()), 2),
                    "التفسير_المنهجي": "6 ساعات رصد بمتوسط 1,039.83 مركبة/ساعة",
                },
            ]
        )
        chart_spec = {
            "chart_type": "daily_trend",
            "df": compute_daily_summary(scope, model),
            "title_ar": "الفرق بين طول التغطية الزمنية (ساعات الرصد) والكثافة المرورية (مركبة/ساعة)",
            "title_en": "Coverage Hours vs Hourly Traffic Intensity (Oct 14 vs Oct 16)",
        }
    elif topic in ("false_premise_direction",):
        res_comp = tool_compare_directions_fairly(
            DirectionComparisonInput(
                dates=params.dates,
                periods=params.periods,
                categories=params.categories,
            ),
            model=model,
        )
        evidence_df = res_comp.evidence_df
        chart_spec = res_comp.chart_spec
    else:
        # For accidents, speeds, spatial maps, bridge impact, mixed units, out-of-range dates:
        # Provide the 9 priority co-activity windows and verified baseline summary as the closest actionable evidence
        evidence_df = co_df
        chart_spec = {
            "chart_type": "pedestrian_coactivity_scatter",
            "ps_df": scope.filter_period_summary(model),
            "co_df": co_df,
            "thresholds": thresholds,
            "title_ar": "البديل الموثق في البيانات: مصفوفة التزامن الزمني بين المركبات والمشاة (9 ساعات أولوية)",
            "title_en": "Verified Alternative in Dataset: Vehicle & Pedestrian Temporal Co-Activity (9 Priority Windows)",
        }

    return AnalysisToolResult(
        tool_name="check_evidence_boundaries",
        parameters=params.model_dump(),
        scope_used=scope,
        headline_metrics={
            **kpis,
            "boundary_topic": topic,
            "coactivity_windows_count": len(co_df),
            "p75_dir1": thresholds.get(1, (2113.0, 563.0)),
            "p75_dir2": thresholds.get(2, (1730.25, 281.0)),
        },
        evidence_df=evidence_df,
        secondary_df=dir_df,
        chart_spec=chart_spec,
        lineage_ar={
            "الأداة": "check_evidence_boundaries (فحص حدود الأدلة وتصحيح الافتراضات)",
            "الموضوع_المفحوص": topic,
            "النطاق_المطبق": scope.describe_ar(),
            "القاعدة_المنهجية": (
                "الالتزام الصارم بحدود البيانات المرصودة (12–17 أكتوبر 2026): عدم اختلاق أرقام للحوادث أو السرعات أو الإحداثيات الجغرافية أو نسب التحول الفعلي للكوبري، "
                "وعدم جمع المركبات والمشاة في رقم واحد، مع تقديم أقرب مؤشر كمي موثق (التعرض الزمني المشترك ومقارنات الساعات المتناظرة)."
            ),
        },
        lineage_en={
            "tool": "check_evidence_boundaries",
            "topic": topic,
            "scope": scope.describe_en(),
            "rule": "Strict adherence to observed dataset boundaries; zero fabrication of crashes, speeds, GIS coordinates, or causal bridge impact.",
        },
        is_boundary_or_refusal=True,
        boundary_topic=topic,
    )


# ============================================================================
# Deterministic Natural-Language Query & Context Resolver
# ============================================================================


ARABIC_DIGITS_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_arabic_text(text: str) -> str:
    """Normalize Arabic text for reliable entity & intent extraction."""
    s = text.translate(ARABIC_DIGITS_MAP).lower().strip()
    s = re.sub(r"[أإآ]", "ا", s)
    s = re.sub(r"ة", "ه", s)
    s = re.sub(r"ى", "ي", s)
    s = re.sub(r"[\u064B-\u065F\u0670]", "", s)  # strip diacritics
    return s


def detect_query_language(text: str, preferred_lang: str = "auto") -> Literal["ar", "en"]:
    """Detect whether the user question is primarily Arabic or English."""
    if preferred_lang in ("ar", "en"):
        return preferred_lang  # type: ignore[return-value]
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    latin_chars = len(re.findall(r"[A-Za-z]", text))
    if arabic_chars == 0 and latin_chars > 0:
        return "en"
    return "ar"


def extract_entities_from_question(
    question: str,
    model: Optional[SemanticModel] = None,
) -> Dict[str, Any]:
    """Extract dates, directions, periods, categories, boundary flags, and follow-up signals from natural language."""
    model = model or build_semantic_model()
    norm = normalize_arabic_text(question)
    raw_lower = question.translate(ARABIC_DIGITS_MAP).lower()

    # 1. Check out-of-range years/months first
    wrong_years = re.findall(r"\b(201\d|202[0-5]|202[7-9])\b", raw_lower)
    wrong_months_ar = any(
        m in norm
        for m in [
            "يناير",
            "فبراير",
            "مارس",
            "ابريل",
            "مايو",
            "يونيو",
            "يوليو",
            "اغسطس",
            "سبتمبر",
            "نوفمبر",
            "ديسمبر",
            "رمضان",
        ]
    )
    wrong_months_en = any(
        m in raw_lower
        for m in [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "november",
            "december",
        ]
    )

    # 2. Extract valid study dates (Oct 12 .. Oct 17, 2026) or weekday names
    dates: List[str] = []
    for day_num in range(12, 18):
        patterns = [
            rf"\b2026-10-{day_num}\b",
            rf"\b{day_num}\s*[/\-]\s*10\b",
            rf"\b10\s*[/\-]\s*{day_num}\b",
            rf"\b{day_num}\s*اكتوبر\b",
            rf"\b{day_num}\s*oct",
            rf"\boct(?:ober)?\s*{day_num}\b",
            rf"(?:يوم|بتاريخ|ليوم|في)\s*{day_num}\b",
            rf"\bday\s*{day_num}\b",
        ]
        if any(re.search(p, norm) or re.search(p, raw_lower) for p in patterns):
            iso_d = f"2026-10-{day_num}"
            if iso_d not in dates:
                dates.append(iso_d)

    # Also check short follow-up like "طب 14؟" or "و 16؟"
    if not dates:
        m_short = re.search(r"\b(1[2-7])\b", norm)
        if m_short and any(
            w in norm
            for w in ["طب", "طيب", "وماذا عن", "وعن", "ويوم", "يوم", "اكتوبر", "oct", "what about", "and on"]
        ):
            dates.append(f"2026-10-{m_short.group(1)}")

    # Check out-of-range days in October (e.g., Oct 10, Oct 11, Oct 18, Oct 20)
    out_of_range_day = False
    for m_day in re.finditer(r"(?:يوم|اكتوبر|oct(?:ober)?)\s*(\d{1,2})\b|\b(\d{1,2})\s*(?:اكتوبر|oct)", norm):
        d_val = int(m_day.group(1) or m_day.group(2))
        if d_val < 12 or d_val > 17:
            out_of_range_day = True

    weekday_to_iso = {
        "الاثنين": "2026-10-12",
        "الثلاثاء": "2026-10-13",
        "الاربعاء": "2026-10-14",
        "الخميس": "2026-10-15",
        "الجمعه": "2026-10-16",
        "السبت": "2026-10-17",
        "monday": "2026-10-12",
        "tuesday": "2026-10-13",
        "wednesday": "2026-10-14",
        "thursday": "2026-10-15",
        "friday": "2026-10-16",
        "saturday": "2026-10-17",
    }
    for w_name, iso_d in weekday_to_iso.items():
        if w_name in norm or w_name in raw_lower:
            if iso_d not in dates:
                dates.append(iso_d)

    if "الاحد" in norm or "sunday" in raw_lower:
        out_of_range_day = True

    # 3. Extract directions
    directions: List[int] = []
    # Check if both directions or comparison is mentioned
    is_both_or_compare = any(
        w in norm
        for w in [
            "قارن",
            "مقارنه",
            "الاتجاهين",
            "كلا الاتجاهين",
            "بين الاتجاهين",
            "الفرق بين",
            "اي اتجاه",
            "ايهما اعلي",
            "ايهما اكبر",
            "compare",
            "both directions",
            "between directions",
            "which direction",
        ]
    )
    if not is_both_or_compare:
        # Check specific direction order
        d1_patterns = [
            r"سندوب\s*(?:->|←|→|الي|الى|لـ|ل)\s*الصينيه",
            r"من\s*سندوب",
            r"اتجاه\s*سندوب\s*الصينيه",
            r"sandoub\s*(?:->|to)\s*(?:al[\-\s]*)?siniya",
            r"from\s*sandoub",
            r"direction\s*1\b",
        ]
        d2_patterns = [
            r"الصينيه\s*(?:->|←|→|الي|الى|لـ|ل)\s*سندوب",
            r"من\s*الصينيه",
            r"اتجاه\s*الصينيه\s*سندوب",
            r"siniya\s*(?:->|to)\s*sandoub",
            r"from\s*(?:al[\-\s]*)?siniya",
            r"direction\s*2\b",
        ]
        if any(re.search(p, norm) or re.search(p, raw_lower) for p in d1_patterns):
            directions.append(1)
        if any(re.search(p, norm) or re.search(p, raw_lower) for p in d2_patterns):
            directions.append(2)

    # 4. Extract vehicle categories
    categories: List[int] = []
    cat_keywords = [
        (1, ["ملاكي", "سيارات خاصه", "private car", "passenger car", "cars"]),
        (2, ["ميكروباص", "microbus"]),
        (3, ["اتوبيس", "باص", "حافلات", "bus", "buses"]),
        (4, ["نقل خفيف", "سوزوكي", "رباعيه", "light truck", "suzuki", "pickup"]),
        (5, ["نقل ثقيل", "تريلا", "تريلات", "شاحنات", "heavy truck", "trailer"]),
        (6, ["موتوسيكل", "دراجات", "دراجه", "motorcycle", "bike", "two-wheel"]),
        (7, ["توك توك", "توكتوك", "تروسيكل", "tuk", "tuktuk", "tricycle"]),
        (8, ["غير مسماه", "غير مصنفه", "custom_1792135578939", "custom", "unnamed"]),
    ]
    for cid, kw_list in cat_keywords:
        if any(kw in norm or kw in raw_lower for kw in kw_list):
            categories.append(cid)

    # 5. Extract specific periods
    periods: List[int] = []
    period_patterns = [
        (420, [r"7\s*[:\-–]\s*8", r"07:00\s*ص", r"7\s*صباحا", r"7\s*am"]),
        (480, [r"8\s*[:\-–]\s*9", r"08:00\s*ص", r"8\s*صباحا", r"8\s*am"]),
        (540, [r"9\s*[:\-–]\s*10", r"09:00\s*ص", r"9\s*صباحا", r"9\s*am"]),
        (600, [r"10\s*[:\-–]\s*11", r"10:00\s*ص", r"10\s*صباحا", r"10\s*am"]),
        (660, [r"11\s*[:\-–]\s*12", r"11:00\s*ص", r"11\s*صباحا", r"11\s*am"]),
        (720, [r"12\s*[:\-–]\s*(?:1|01|13)", r"12:00\s*م", r"12\s*ظهرا", r"12\s*pm"]),
        (780, [r"(?:1|01|13)\s*[:\-–]\s*(?:2|02|14)", r"01:00\s*م", r"1\s*ظهرا", r"1\s*pm", r"واحده\s*ظهرا"]),
        (840, [r"(?:2|02|14)\s*[:\-–]\s*(?:3|03|15)", r"02:00\s*م", r"2\s*ظهرا", r"2\s*pm"]),
        (900, [r"(?:3|03|15)\s*[:\-–]\s*(?:4|04|16)", r"03:00\s*م", r"3\s*عصرا", r"3\s*مساء", r"3\s*pm"]),
        (960, [r"(?:4|04|16)\s*[:\-–]\s*(?:5|05|17)", r"04:00\s*م", r"4\s*عصرا", r"4\s*مساء", r"4\s*pm"]),
    ]
    for p_min, p_list in period_patterns:
        if any(re.search(p, norm) or re.search(p, raw_lower) for p in p_list):
            periods.append(p_min)

    # Detect follow-up signal
    is_followup = (
        len(question.split()) <= 7
        and any(
            w in norm or w in raw_lower
            for w in [
                "طب",
                "طيب",
                "وماذا عن",
                "وعن",
                "وفي",
                "والاتجاه",
                "والعكس",
                "العكسي",
                "الاخر",
                "الثاني",
                "what about",
                "how about",
                "and for",
                "and on",
                "and in",
                "reverse direction",
                "other direction",
            ]
        )
    )
    wants_opposite_direction = any(
        w in norm or w in raw_lower
        for w in ["الاتجاه العكسي", "الاتجاه الاخر", "الاتجاه الثاني", "والعكس", "other direction", "opposite direction", "reverse direction"]
    )

    return {
        "dates": dates,
        "directions": directions,
        "categories": categories,
        "periods": periods,
        "wrong_years": wrong_years,
        "wrong_month": wrong_months_ar or wrong_months_en,
        "out_of_range_day": out_of_range_day,
        "is_followup": is_followup,
        "wants_opposite_direction": wants_opposite_direction,
        "is_both_or_compare": is_both_or_compare,
        "norm": norm,
        "raw_lower": raw_lower,
    }


def resolve_question_and_execute(
    question: str,
    session_context: Optional[Dict[str, Any]] = None,
    ui_scope: Optional[FilterScope] = None,
    model: Optional[SemanticModel] = None,
) -> Tuple[AnalysisToolResult, Dict[str, Any]]:
    """Deterministically resolve any Arabic/English question into a guarded tool call, execute it, and return updated session context."""
    model = model or build_semantic_model()
    ctx = dict(session_context or {})
    ent = extract_entities_from_question(question, model)
    norm = ent["norm"]
    raw_lower = ent["raw_lower"]

    inherited_ar: List[str] = []
    inherited_en: List[str] = []

    # Resolve scope parameters with context inheritance for follow-ups
    dates = ent["dates"] or None
    directions = ent["directions"] or None
    categories = ent["categories"] or None
    periods = ent["periods"] or None

    if ent["wants_opposite_direction"] and ctx.get("last_directions"):
        prev_d = ctx["last_directions"][0]
        directions = [2 if prev_d == 1 else 1]
        inherited_ar.append(f"تم التبديل إلى الاتجاه المعاكس ({DIRECTION_KEY_TO_AR[directions[0]]}) بناءً على سياق السؤال السابق.")
        inherited_en.append(f"Switched to opposite direction ({DIRECTION_KEY_TO_EN[directions[0]]}) from previous context.")

    if ent["is_followup"]:
        if not dates and ctx.get("last_dates") and not any(w in norm for w in ["كل الايام", "جميع الايام", "all days"]):
            # If user changed direction or category in follow-up, keep previous date
            if directions or categories or periods:
                dates = ctx["last_dates"]
                inherited_ar.append(f"تم وراثة التاريخ ({'، '.join(dates)}) من السؤال السابق.")
                inherited_en.append(f"Inherited date filter ({', '.join(dates)}) from previous question.")
        if not directions and ctx.get("last_directions") and not ent["is_both_or_compare"]:
            if dates or categories or periods:
                directions = ctx["last_directions"]
                inherited_ar.append(
                    f"تم وراثة الاتجاه ({'، '.join(DIRECTION_KEY_TO_AR[d] for d in directions)}) من السؤال السابق."
                )
                inherited_en.append(
                    f"Inherited direction ({', '.join(DIRECTION_KEY_TO_EN[d] for d in directions)}) from previous question."
                )
        if not categories and ctx.get("last_categories"):
            if dates or directions or periods:
                categories = ctx["last_categories"]
                inherited_ar.append("تم وراثة فئة المركبة من السؤال السابق.")
                inherited_en.append("Inherited vehicle category filter from previous question.")

    # Fall back to active UI slicers if no explicit filter was stated in the question
    if ui_scope is not None:
        if not dates and ui_scope.dates:
            dates = [d.isoformat() for d in ui_scope.dates]
            inherited_ar.append(f"تم تطبيق فلتر التاريخ النشط في الواجهة ({'، '.join(dates)}).")
            inherited_en.append(f"Applied active UI date filter ({', '.join(dates)}).")
        if not directions and ui_scope.directions and not ent["is_both_or_compare"]:
            directions = list(ui_scope.directions)
            inherited_ar.append(
                f"تم تطبيق فلتر الاتجاه النشط في الواجهة ({'، '.join(DIRECTION_KEY_TO_AR[d] for d in directions)})."
            )
            inherited_en.append(
                f"Applied active UI direction filter ({', '.join(DIRECTION_KEY_TO_EN[d] for d in directions)})."
            )
        if not periods and ui_scope.periods:
            periods = list(ui_scope.periods)
            inherited_ar.append("تم تطبيق فلتر الفترات الزمنية النشط في الواجهة.")
            inherited_en.append("Applied active UI period filter.")
        if not categories and ui_scope.categories:
            categories = list(ui_scope.categories)
            inherited_ar.append("تم تطبيق فلتر فئة المركبات النشط في الواجهة.")
            inherited_en.append("Applied active UI category filter.")

    # ------------------------------------------------------------------------
    # 1. Check Evidence Boundaries & False Premises First
    # ------------------------------------------------------------------------
    # 1a. Out-of-range year or month or day
    if ent["wrong_years"] or ent["wrong_month"] or ent["out_of_range_day"]:
        res = tool_check_evidence_boundaries(
            EvidenceBoundaryInput(topic="wrong_date_or_year"),
            model=model,
        )
        res.inherited_context_notes_ar = inherited_ar
        res.inherited_context_notes_en = inherited_en
        return res, ctx

    # 1b. Explicit request to sum/add vehicles + pedestrians together as one total
    if any(
        p in norm or p in raw_lower
        for p in [
            "اجمع المركبات والمشاه",
            "مجموع المركبات والمشاه",
            "حاصل جمع المركبات والمشاه",
            "اجمالي المركبات مع المشاه",
            "sum vehicles and pedestrians",
            "add vehicles and pedestrians",
            "combined total of vehicles and pedestrians",
            "total traffic including pedestrians",
        ]
    ):
        res = tool_check_evidence_boundaries(
            EvidenceBoundaryInput(
                dates=dates,
                directions=directions,
                periods=periods,
                topic="mixed_units",
            ),
            model=model,
        )
        res.inherited_context_notes_ar = inherited_ar
        res.inherited_context_notes_en = inherited_en
        return res, ctx

    # 1c. False premise: "Why is Al-Siniya -> Sandoub higher than Sandoub -> Al-Siniya?"
    if (
        any(w in norm or w in raw_lower for w in ["لماذا", "ليه", "كيف", "ازاي", "why", "how come"])
        and any(
            p in norm or p in raw_lower
            for p in [
                "الصينيه الي سندوب اعلي",
                "الصينيه الي سندوب اكبر",
                "الصينيه -> سندوب اعلي",
                "الصينيه ← سندوب اعلي",
                "الصينيه الي سندوب اكثر",
                "siniya to sandoub higher",
                "siniya -> sandoub higher",
                "siniya to sandoub more traffic",
            ]
        )
        and not categories  # Note: for Heavy Trucks % share, Dir 2 IS higher (6.88% vs 5.11%), so only flag for general traffic
    ):
        res = tool_check_evidence_boundaries(
            EvidenceBoundaryInput(dates=dates, periods=periods, topic="false_premise_direction"),
            model=model,
        )
        res.inherited_context_notes_ar = inherited_ar
        res.inherited_context_notes_en = inherited_en
        return res, ctx

    # 1d. False premise: Oct 16 vs Oct 14 intensity vs coverage hours
    if ("16" in norm and "14" in norm) and any(
        w in norm or w in raw_lower
        for w in ["اعلي", "اكثر", "اكبر", "ازحم", "كثافه", "تغطيه", "ساعات", "higher", "more", "busier", "coverage", "hours"]
    ):
        res = tool_check_evidence_boundaries(
            EvidenceBoundaryInput(topic="false_premise_oct16_vs_oct14"),
            model=model,
        )
        res.inherited_context_notes_ar = inherited_ar
        res.inherited_context_notes_en = inherited_en
        return res, ctx

    # 1e. Hypothetical What-If Scenario vs Unobserved Bridge/Safety Boundary
    is_what_if = any(
        w in norm or w in raw_lower
        for w in [
            "ماذا لو",
            "لو افترضنا",
            "بافتراض",
            "سيناريو",
            "نفترض",
            "تخيل لو",
            "what if",
            "suppose",
            "assuming",
            "hypothetical",
            "scenario",
            "if we reduce",
            "if bridge",
        ]
    )
    if is_what_if:
        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", raw_lower)
        pct_val = float(pct_match.group(1)) if pct_match else 70.0
        if any(w in norm or w in raw_lower for w in ["كوبري", "مشاه", "عبور", "bridge", "pedestrian", "crossing"]):
            scen_type: Literal["vehicle_category_change", "pedestrian_bridge_diversion", "peak_hour_shift"] = (
                "pedestrian_bridge_diversion"
            )
        else:
            scen_type = "vehicle_category_change"
            if any(w in norm or w in raw_lower for w in ["انخفض", "قل", "تخفيض", "نقص", "decrease", "reduce", "drop"]):
                pct_val = -abs(pct_val)
        res = tool_evaluate_what_if_scenario(
            WhatIfScenarioInput(
                dates=dates,
                directions=directions,
                periods=periods,
                categories=categories,
                scenario_type=scen_type,
                change_pct=pct_val,
                target_category=categories[0] if categories else None,
            ),
            model=model,
        )
        res.inherited_context_notes_ar = inherited_ar
        res.inherited_context_notes_en = inherited_en
        ctx.update(
            {
                "last_tool": "evaluate_what_if_scenario",
                "last_dates": dates,
                "last_directions": directions,
                "last_categories": categories,
                "last_periods": periods,
            }
        )
        return res, ctx

    # 1f. Unsupported physical/safety/spatial/causal topics
    if any(
        w in norm or w in raw_lower
        for w in [
            "حادث",
            "حوادث",
            "تصادم",
            "اصابات",
            "وفيات",
            "قتلي",
            "دهس",
            "accident",
            "crash",
            "collision",
            "fatality",
            "fatalities",
            "injury",
            "injuries",
        ]
    ):
        res = tool_check_evidence_boundaries(
            EvidenceBoundaryInput(dates=dates, directions=directions, periods=periods, topic="accidents"),
            model=model,
        )
        res.inherited_context_notes_ar = inherited_ar
        res.inherited_context_notes_en = inherited_en
        return res, ctx

    if any(
        w in norm or w in raw_lower
        for w in [
            "سرعه",
            "سرعات",
            "كم/س",
            "كيلومتر في الساعه",
            "زمن الانتظار",
            "طابور",
            "طوابير",
            "مستوي الخدمه",
            "سعه الحاره",
            "عدد الركاب",
            "اشغال المركبه",
            "speed",
            "km/h",
            "mph",
            "wait time",
            "delay",
            "queue",
            "level of service",
            "lane capacity",
            "occupancy",
        ]
    ):
        res = tool_check_evidence_boundaries(
            EvidenceBoundaryInput(dates=dates, directions=directions, periods=periods, topic="speed_or_delay"),
            model=model,
        )
        res.inherited_context_notes_ar = inherited_ar
        res.inherited_context_notes_en = inherited_en
        return res, ctx

    if any(
        w in norm or w in raw_lower
        for w in [
            "احداثيات",
            "خريطه جغرافيه",
            "خريطه حراريه مكانيه",
            "موقع النقطه",
            "اي نقطه علي الطريق",
            "اين يقع اخطر مكان",
            "اخطر نقطه عبور",
            "gps",
            "coordinates",
            "latitude",
            "longitude",
            "spatial map",
            "exact location on the road",
            "hotspot map",
        ]
    ):
        res = tool_check_evidence_boundaries(
            EvidenceBoundaryInput(dates=dates, directions=directions, periods=periods, topic="map_coordinates"),
            model=model,
        )
        res.inherited_context_notes_ar = inherited_ar
        res.inherited_context_notes_en = inherited_en
        return res, ctx

    if any(
        w in norm or w in raw_lower
        for w in [
            "كم شخص استخدم الكوبري",
            "نسبه استخدام الكوبري",
            "فوق الكوبري وتحت الكوبري",
            "فوق الكوبري مقابل",
            "اثر الكوبري الفعلي",
            "هل قلل الكوبري",
            "قبل وبعد الكوبري",
            "how many used the bridge",
            "bridge usage rate",
            "above bridge vs below",
            "bridge vs street",
            "did the bridge reduce",
            "actual bridge impact",
        ]
    ):
        res = tool_check_evidence_boundaries(
            EvidenceBoundaryInput(dates=dates, directions=directions, periods=periods, topic="bridge_impact"),
            model=model,
        )
        res.inherited_context_notes_ar = inherited_ar
        res.inherited_context_notes_en = inherited_en
        return res, ctx

    # ------------------------------------------------------------------------
    # 2. Route to the Best Analytical Tool Based on Intent
    # ------------------------------------------------------------------------
    # Inherit previous tool if this is a short follow-up that only supplied a filter
    last_tool = ctx.get("last_tool", "get_overview_and_kpis")

    # 2a. Data quality, coverage gaps, multi-observer, custom_1792135578939, audit logs, zero values
    if any(
        w in norm or w in raw_lower
        for w in [
            "جوده البيانات",
            "تغطيه",
            "فجوات",
            "ساعات غائبه",
            "ساعات غير مرصوده",
            "تعدد الراصدين",
            "راصدين",
            "التكرار",
            "مكرره",
            "فئه غير مسماه",
            "custom_1792135578939",
            "الاصفار",
            "صفريه",
            "الخلايا الصفريه",
            "قيم صفريه",
            "سجل العمليات",
            "سجل التدقيق",
            "68196",
            "68,196",
            "151810",
            "301",
            "audit",
            "operation log",
            "data quality",
            "coverage gap",
            "missing hours",
            "multi-observer",
            "duplicate",
            "unnamed category",
            "zero values",
            "zero cells",
        ]
    ):
        aspect: Literal[
            "all", "coverage_gaps", "multi_observer", "unnamed_category", "zero_values", "audit_logs"
        ] = "all"
        if any(w in norm or w in raw_lower for w in ["تعدد الراصدين", "راصدين", "مكرر", "multi-observer", "duplicate"]):
            aspect = "multi_observer"
        elif any(w in norm or w in raw_lower for w in ["غير مسماه", "custom", "unnamed"]):
            aspect = "unnamed_category"
        elif any(w in norm or w in raw_lower for w in ["اصفار", "صفريه", "zero"]):
            aspect = "zero_values"
        elif any(w in norm or w in raw_lower for w in ["سجل العمليات", "التدقيق", "68196", "68,196", "301", "audit", "log"]):
            aspect = "audit_logs"
        elif any(w in norm or w in raw_lower for w in ["تغطيه", "فجوات", "غائبه", "coverage", "missing"]):
            aspect = "coverage_gaps"

        res = tool_inspect_coverage_and_data_quality(
            CoverageAndQualityInput(
                dates=dates,
                directions=directions,
                periods=periods,
                categories=categories,
                aspect=aspect,
            ),
            model=model,
        )
        res.inherited_context_notes_ar = inherited_ar
        res.inherited_context_notes_en = inherited_en
        ctx.update(
            {
                "last_tool": "inspect_coverage_and_data_quality",
                "last_dates": dates,
                "last_directions": directions,
                "last_categories": categories,
                "last_periods": periods,
            }
        )
        return res, ctx

    # 2b. Pedestrian crossings & co-activity windows
    if any(
        w in norm or w in raw_lower
        for w in [
            "مشاه",
            "عبور",
            "تزامن",
            "المئين 75",
            "اولويه",
            "كوبري المشاه",
            "pedestrian",
            "crossing",
            "co-activity",
            "coactivity",
            "simultaneous",
            "p75",
        ]
    ):
        focus: Literal["overview", "coactivity_windows", "by_period", "by_date"] = "overview"
        if any(w in norm or w in raw_lower for w in ["تزامن", "75", "اولويه", "مشتركه", "co-activity", "coactivity", "p75", "simultaneous"]):
            focus = "coactivity_windows"
        res = tool_analyze_pedestrian_and_coactivity(
            PedestrianCoactivityInput(
                dates=dates,
                directions=directions,
                periods=periods,
                focus=focus,
            ),
            model=model,
        )
        res.inherited_context_notes_ar = inherited_ar
        res.inherited_context_notes_en = inherited_en
        ctx.update(
            {
                "last_tool": "analyze_pedestrian_and_coactivity",
                "last_dates": dates,
                "last_directions": directions,
                "last_categories": categories,
                "last_periods": periods,
            }
        )
        return res, ctx

    # 2c. Explicit fair comparison between directions (when not asking for a category/period/date table)
    is_explicit_compare = any(
        w in norm or w in raw_lower
        for w in [
            "قارن",
            "مقارنه",
            "الفرق بين الاتجاهين",
            "ايهما اعلي",
            "ايهما اكبر",
            "الساعات المشتركه",
            "الفترات المتناظره",
            "الفترات المشتركه",
            "38 ساعه",
            "compare",
            "matched hours",
            "matched windows",
            "common hours",
            "fair comparison",
        ]
    )
    if is_explicit_compare:
        target: Literal["vehicles", "pedestrians", "category"] = (
            "category" if categories else "vehicles"
        )
        res = tool_compare_directions_fairly(
            DirectionComparisonInput(
                dates=dates,
                periods=periods,
                categories=categories,
                target=target,
                category_filter=categories[0] if categories else None,
            ),
            model=model,
        )
        res.inherited_context_notes_ar = inherited_ar
        res.inherited_context_notes_en = inherited_en
        ctx.update(
            {
                "last_tool": "compare_directions_fairly",
                "last_dates": dates,
                "last_directions": [1, 2],
                "last_categories": categories,
                "last_periods": periods,
            }
        )
        return res, ctx

    # 2d. Peak hours, lowest hours, anomalies, 12-1 vs 1-2 PM (when not asking for a specific vehicle category breakdown)
    if not categories and any(
        w in norm or w in raw_lower
        for w in [
            "ذروه",
            "اعلي ساعه",
            "اكثر ساعه",
            "اقصي ساعه",
            "ترتيب الساعات",
            "ادني ساعه",
            "ادني ساعات",
            "اقل ساعه",
            "اقل ساعات",
            "اهدا",
            "اهدأ",
            "متطرفه",
            "شذوذ",
            "قفزه",
            "نمط متكرر",
            "peak",
            "highest hour",
            "busiest hour",
            "top hours",
            "lowest hour",
            "lowest traffic hours",
            "quietest",
            "outlier",
            "anomaly",
            "spike",
        ]
    ):
        rank_order: Literal["top", "bottom"] = (
            "bottom"
            if any(w in norm or w in raw_lower for w in ["ادني", "اقل", "اهدا", "اهدأ", "lowest", "quietest", "minimum", "bottom"])
            else "top"
        )
        res = tool_analyze_peaks_and_anomalies(
            PeaksAndAnomaliesInput(
                dates=dates,
                directions=directions,
                periods=periods,
                categories=categories,
                rank_order=rank_order,
                top_n=10,
            ),
            model=model,
        )
        res.inherited_context_notes_ar = inherited_ar
        res.inherited_context_notes_en = inherited_en
        ctx.update(
            {
                "last_tool": "analyze_peaks_and_anomalies",
                "last_dates": dates,
                "last_directions": directions,
                "last_categories": categories,
                "last_periods": periods,
            }
        )
        return res, ctx

    # 2e. Vehicle classifications / specific category or group-by queries
    if categories or any(
        w in norm or w in raw_lower
        for w in [
            "فئات",
            "تصنيف",
            "انواع المركبات",
            "توزيع المركبات",
            "توزيع فئات",
            "نسبه كل فئه",
            "حسب اليوم",
            "حسب التاريخ",
            "لكل يوم",
            "يوميا",
            "حسب الفتره",
            "لكل ساعه",
            "categories",
            "classification",
            "vehicle types",
            "vehicle mix",
            "breakdown",
            "by day",
            "by date",
            "daily",
            "by period",
            "time period",
            "hourly",
        ]
    ):
        group_by: List[str] = []
        if any(w in norm or w in raw_lower for w in ["حسب اليوم", "حسب التاريخ", "لكل يوم", "يوميا", "الايام", "by day", "by date", "daily"]):
            group_by.append("date")
        if any(w in norm or w in raw_lower for w in ["حسب الفتره", "لكل فتره", "حسب الساعه", "ساعات اليوم", "متي تبلغ ذروتها", "ذروتها", "by period", "time period", "by hour", "hourly"]) or periods:
            group_by.append("period")
        if any(w in norm or w in raw_lower for w in ["حسب الاتجاه", "لكل اتجاه", "في الاتجاهين", "والاتجاه", "بين الاتجاهين", "by direction", "direction"]) or directions:
            group_by.append("direction")
        if categories or any(
            w in norm or w in raw_lower
            for w in ["فئات", "تصنيف", "انواع المركبات", "توزيع المركبات", "توزيع فئات", "نسبه كل فئه", "categories", "classification", "vehicle types", "mix"]
        ):
            group_by.append("vehicle_category")

        if not group_by:
            group_by = ["vehicle_category", "direction"] if categories else ["date", "direction"]

        metric = "mean_vehicles_per_hour" if any(w in norm or w in raw_lower for w in ["متوسط", "معدل", "كثافه", "mean", "average", "rate"]) else "total_vehicles"

        res = tool_query_traffic_aggregations(
            AggregationQueryInput(
                dates=dates,
                directions=directions,
                periods=periods,
                categories=categories,
                group_by=group_by,
                metric=metric,
                sort_order="desc",
            ),
            model=model,
        )
        res.inherited_context_notes_ar = inherited_ar
        res.inherited_context_notes_en = inherited_en
        ctx.update(
            {
                "last_tool": "query_traffic_aggregations",
                "last_group_by": group_by,
                "last_metric": metric,
                "last_dates": dates,
                "last_directions": directions,
                "last_categories": categories,
                "last_periods": periods,
            }
        )
        return res, ctx

    # 2f. Short follow-up that only changed filter parameters (e.g. "طب يوم 14؟")
    if ent["is_followup"] and last_tool:
        if last_tool == "analyze_pedestrian_and_coactivity":
            res = tool_analyze_pedestrian_and_coactivity(
                PedestrianCoactivityInput(dates=dates, directions=directions, periods=periods),
                model=model,
            )
        elif last_tool == "compare_directions_fairly":
            res = tool_compare_directions_fairly(
                DirectionComparisonInput(dates=dates, periods=periods, categories=categories),
                model=model,
            )
        elif last_tool == "analyze_peaks_and_anomalies":
            res = tool_analyze_peaks_and_anomalies(
                PeaksAndAnomaliesInput(dates=dates, directions=directions, periods=periods, categories=categories),
                model=model,
            )
        elif last_tool == "query_traffic_aggregations":
            res = tool_query_traffic_aggregations(
                AggregationQueryInput(
                    dates=dates,
                    directions=directions,
                    periods=periods,
                    categories=categories,
                    group_by=ctx.get("last_group_by", ["direction"]),
                    metric=ctx.get("last_metric", "total_vehicles"),
                ),
                model=model,
            )
        else:
            res = tool_get_overview_and_kpis(
                ScopeInputModel(dates=dates, directions=directions, periods=periods, categories=categories),
                model=model,
            )
        res.inherited_context_notes_ar = inherited_ar
        res.inherited_context_notes_en = inherited_en
        ctx.update(
            {
                "last_dates": dates,
                "last_directions": directions,
                "last_categories": categories,
                "last_periods": periods,
            }
        )
        return res, ctx

    # 2g. Default: Comprehensive Overview & KPIs for the requested scope
    res = tool_get_overview_and_kpis(
        ScopeInputModel(
            dates=dates,
            directions=directions,
            periods=periods,
            categories=categories,
        ),
        model=model,
    )
    res.inherited_context_notes_ar = inherited_ar
    res.inherited_context_notes_en = inherited_en
    ctx.update(
        {
            "last_tool": "get_overview_and_kpis",
            "last_dates": dates,
            "last_directions": directions,
            "last_categories": categories,
            "last_periods": periods,
        }
    )
    return res, ctx
