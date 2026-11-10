"""Deterministic unit tests for baseline metrics, Star Schema, and Power BI 171-context reconciliation."""
from __future__ import annotations

import json
import math
import zipfile
from typing import Any, Optional

import pytest

from src.config import POWERBI_ZIP_PATH, SOURCE_VEHICLE_COLS
from src.data_loader import (
    get_audit_log_reconciliation,
    load_audit_operation_logs_df,
    load_raw_survey_df,
    verify_input_manifest,
)
from src.metrics_engine import (
    FilterScope,
    compute_all_powerbi_measures,
    compute_category_breakdown,
    compute_coactivity_windows,
    compute_daily_summary,
    compute_kpis,
    compute_period_profile,
    compute_ranked_hours,
    run_baseline_qa_checks,
)
from src.semantic_model import build_semantic_model


def test_input_manifest_sha256_integrity() -> None:
    """Verify that all 5 reference files in inputs/ are untouched and match input_manifest.json."""
    res = verify_input_manifest()
    assert res["valid"] is True
    assert len(res["files"]) == 5
    for fname, info in res["files"].items():
        assert info["exists"] is True, f"Missing input file: {fname}"
        assert info["valid"] is True, f"SHA-256 or byte mismatch for {fname}"


def test_star_schema_dimensions_and_fact_row_counts() -> None:
    """Verify exact Star Schema table row counts matching Traffic.SemanticModel."""
    model = build_semantic_model()
    assert len(model.survey_raw) == 96
    assert len(model.dim_date) == 6
    assert len(model.dim_direction) == 2
    assert len(model.dim_period) == 10
    assert len(model.dim_vehicle_category) == 8
    assert len(model.fact_vehicle_count) == 768  # 96 rows * 8 categories
    assert len(model.fact_pedestrian_count) == 96
    assert len(model.fact_coverage) == 81
    assert len(model.fact_period_summary) == 81


def test_all_12_baseline_qa_checks_pass() -> None:
    """Verify all 12 baseline QA checks pass."""
    qa = run_baseline_qa_checks()
    assert qa["all_passed"] is True
    for chk in qa["checks"]:
        assert chk["passed"] is True, f"Baseline check failed: {chk}"


def test_exact_reconciled_headline_metrics() -> None:
    """Verify every required baseline metric from AGENT_TASK.md."""
    k = compute_kpis(FilterScope())

    # Survey records & coverage
    assert k["عدد_صفوف_الرصد"] == 96
    assert k["عدد سجلات الحصر من سندوب إلى الصينية"] == 52
    assert k["عدد سجلات الحصر من الصينية إلى سندوب"] == 44
    assert k["ساعات_الرصد_الفعلي"] == 81
    assert k["ساعات_سندوب_إلى_الصينية"] == 43
    assert k["ساعات_الصينية_إلى_سندوب"] == 38
    assert k["عدد الساعات الزمنية المرصودة"] == 43
    assert k["عدد_الساعات_المشتركة"] == 38

    # Vehicle totals & hourly means
    assert k["إجمالي_المركبات"] == 127769
    assert k["مركبات_سندوب_إلى_الصينية"] == 74248
    assert k["مركبات_الصينية_إلى_سندوب"] == 53521
    assert math.isclose(k["متوسط_سندوب_إلى_الصينية"], 74248 / 43, rel_tol=1e-9)
    assert math.isclose(k["متوسط_الصينية_إلى_سندوب"], 53521 / 38, rel_tol=1e-9)
    assert math.isclose(k["متوسط_المركبات_لكل_ساعة_رصد"], 127769 / 81, rel_tol=1e-9)

    # Matched 38-hour comparison
    assert k["مركبات_سندوب_المشتركة"] == 65279
    assert k["مركبات_الصينية_المشتركة"] == 53521
    assert k["فرق_الساعات_المشتركة"] == 11758
    assert math.isclose(k["نسبة_فرق_الساعات_المشتركة"], 11758 / 53521 * 100.0, rel_tol=1e-9)

    # Peak hours
    assert k["أعلى عدد مركبات في ساعة من سندوب إلى الصينية"] == 4425
    assert k["تاريخ وفترة الذروة من سندوب إلى الصينية"] == "2026-10-14 | 12:00 م - 01:00 م"
    assert k["أعلى عدد مركبات في ساعة من الصينية إلى سندوب"] == 2687
    assert k["تاريخ وفترة الذروة من الصينية إلى سندوب"] == "2026-10-17 | 12:00 م - 01:00 م"

    top5 = compute_ranked_hours(FilterScope(), top_n=5, ascending=False)
    assert top5["إجمالي_المركبات"].tolist() == [4425, 3636, 2687, 2546, 2508]

    # Independent pedestrian metrics (never mixed with vehicles)
    assert k["إجمالي_عبور_المشاة"] == 24342
    assert k["عبور_سندوب_إلى_الصينية"] == 17478
    assert k["عبور_الصينية_إلى_سندوب"] == 6864
    assert math.isclose(k["متوسط_عبور_سندوب_لكل_ساعة"], 17478 / 43, rel_tol=1e-9)
    assert math.isclose(k["متوسط_عبور_الصينية_لكل_ساعة"], 6864 / 38, rel_tol=1e-9)


def test_sustained_pattern_and_coverage_vs_intensity() -> None:
    """Verify 1-2 PM recurring peak pattern and Oct 14 vs Oct 16 coverage vs intensity."""
    period_df = compute_period_profile(FilterScope())
    p780_d1 = period_df[(period_df["ترتيب الفترة"] == 780) & (period_df["مفتاح الاتجاه"] == 1)].iloc[0]
    p780_d2 = period_df[(period_df["ترتيب الفترة"] == 780) & (period_df["مفتاح الاتجاه"] == 2)].iloc[0]
    assert int(p780_d1["فترات_الرصد"]) == 5
    assert math.isclose(float(p780_d1["متوسط_المركبات_في_الساعة"]), 2141.60, abs_tol=1e-2)
    assert int(p780_d2["فترات_الرصد"]) == 4
    assert math.isclose(float(p780_d2["متوسط_المركبات_في_الساعة"]), 1663.75, abs_tol=1e-2)

    daily_df = compute_daily_summary(FilterScope())
    oct14_d1 = daily_df[(daily_df["تاريخ_نص"] == "2026-10-14") & (daily_df["مفتاح الاتجاه"] == 1)].iloc[0]
    oct16_d1 = daily_df[(daily_df["تاريخ_نص"] == "2026-10-16") & (daily_df["مفتاح الاتجاه"] == 1)].iloc[0]
    assert int(oct14_d1["ساعات_الرصد"]) == 7
    assert math.isclose(float(oct14_d1["متوسط_المركبات_في_الساعة"]), 2543.43, abs_tol=1e-2)
    assert int(oct16_d1["ساعات_الرصد"]) == 9
    assert math.isclose(float(oct16_d1["متوسط_المركبات_في_الساعة"]), 777.89, abs_tol=1e-2)


def test_vehicle_categories_and_unnamed_class_preservation() -> None:
    """Verify all 8 vehicle categories, shares, heavy truck peak, and custom_1792135578939 = 30."""
    cat_df = compute_category_breakdown(FilterScope())
    assert len(cat_df) == 8
    assert int(cat_df["إجمالي_المركبات"].sum()) == 127769
    assert math.isclose(float(cat_df["النسبة_من_إجمالي_المركبات"].sum()), 100.0, rel_tol=1e-9)

    cars = cat_df[cat_df["ترتيب الفئة"] == 1].iloc[0]
    assert int(cars["إجمالي_المركبات"]) == 63559
    assert round(float(cars["النسبة_من_إجمالي_المركبات"]), 2) == 49.75
    assert round(float(cars["حصة_سندوب_إلى_الصينية_%"]), 2) == 51.74
    assert round(float(cars["حصة_الصينية_إلى_سندوب_%"]), 2) == 46.98

    heavy = cat_df[cat_df["ترتيب الفئة"] == 5].iloc[0]
    assert int(heavy["إجمالي_المركبات"]) == 7478
    assert round(float(heavy["النسبة_من_إجمالي_المركبات"]), 2) == 5.85
    assert round(float(heavy["حصة_سندوب_إلى_الصينية_%"]), 2) == 5.11
    assert round(float(heavy["حصة_الصينية_إلى_سندوب_%"]), 2) == 6.88

    unnamed = cat_df[cat_df["ترتيب الفئة"] == 8].iloc[0]
    assert int(unnamed["إجمالي_المركبات"]) == 30
    assert int(unnamed["سندوب_إلى_الصينية"]) == 0
    assert int(unnamed["الصينية_إلى_سندوب"]) == 30


def test_coactivity_windows_and_multi_observer_and_audit_logs() -> None:
    """Verify P75 co-activity windows (9 = 4 + 5), 15 multi-observer periods (30 rows), and 68,196 audit logs."""
    co_df, thresholds = compute_coactivity_windows(FilterScope())
    assert thresholds[1] == (2113.0, 563.0)
    assert thresholds[2] == (1730.25, 281.0)
    assert len(co_df) == 9
    assert int((co_df["اتجاه_مختصر"] == "سندوب → الصينية").sum()) == 4
    assert int((co_df["اتجاه_مختصر"] == "الصينية → سندوب").sum()) == 5

    model = build_semantic_model()
    ps = model.fact_period_summary
    multi = ps[ps["تعدد_الراصدين"]]
    assert len(multi) == 15
    assert int(multi["عدد السجلات"].sum()) == 30

    raw = load_raw_survey_df()
    count_cols = SOURCE_VEHICLE_COLS + ["مشاة (عبور المشاة)"]
    total_zeros = sum(int((raw[c] == 0).sum()) for c in count_cols)
    assert total_zeros == 216

    logs_df = load_audit_operation_logs_df()
    assert len(logs_df) == 68196
    assert int(logs_df["قيمة_التغير"].sum()) == 151810
    rec = get_audit_log_reconciliation()
    assert rec["net_difference"] == 301


def test_reconcile_powerbi_validation_json_oracles() -> None:
    """Reconcile compute_all_powerbi_measures against TrafficArabic/QA/calculation_validation.json and baseline/validation.json."""
    with zipfile.ZipFile(POWERBI_ZIP_PATH, "r") as z:
        calc_oracle = json.loads(
            z.read("TrafficArabic/QA/calculation_validation.json").decode("utf-8")
        )
        base_oracle = json.loads(
            z.read("TrafficArabic/QA/baseline/validation.json").decode("utf-8")
        )

    assert calc_oracle["pass"] is True
    assert calc_oracle["counted_windows"] == 81
    assert calc_oracle["original_contexts"] == 171

    computed_full = compute_all_powerbi_measures(FilterScope())
    for m_name, exp_val in calc_oracle["all_study_totals"].items():
        assert m_name in computed_full, f"Missing measure: {m_name}"
        act_val = computed_full[m_name]
        if exp_val is None:
            assert act_val is None, f"Measure {m_name}: expected None, got {act_val}"
        elif isinstance(exp_val, float):
            assert act_val is not None and math.isclose(
                float(act_val), exp_val, rel_tol=1e-7, abs_tol=1e-7
            ), f"Measure {m_name}: expected {exp_val}, got {act_val}"
        else:
            assert act_val == exp_val, f"Measure {m_name}: expected {exp_val}, got {act_val}"

    # Verify top_five and simultaneous_windows match calculation_validation.json
    top5 = compute_ranked_hours(FilterScope(), top_n=5, ascending=False)
    assert len(top5) == len(calc_oracle["top_five"]) == 5
    for idx, exp_item in enumerate(calc_oracle["top_five"]):
        exp_vehicles = exp_item["vehicles"] if isinstance(exp_item, dict) else exp_item[-2]
        exp_pedestrians = exp_item["pedestrians"] if isinstance(exp_item, dict) else exp_item[-1]
        assert int(top5.iloc[idx]["إجمالي_المركبات"]) == int(exp_vehicles)
        assert int(top5.iloc[idx]["عبور_مشاة_الفترة"]) == int(exp_pedestrians)

    co_df, thresholds = compute_coactivity_windows(FilterScope())
    assert len(co_df) == len(calc_oracle["simultaneous_windows"]) == 9
    assert list(thresholds[1]) == calc_oracle["full_sample_thresholds"]["1"]
    assert list(thresholds[2]) == calc_oracle["full_sample_thresholds"]["2"]

    # Also verify baseline/validation.json table_reference_counts
    ref_counts = base_oracle["table_reference_counts"]
    model = build_semantic_model()
    assert len(model.dim_date) == ref_counts["DimDate"]
    assert len(model.dim_direction) == ref_counts["DimDirection"]
    assert len(model.dim_period) == ref_counts["DimPeriod"]
    assert len(model.dim_vehicle_category) == ref_counts["DimVehicleCategory"]
    assert len(model.fact_vehicle_count) == ref_counts["FactVehicleCount"]
    assert len(model.fact_pedestrian_count) == ref_counts["FactPedestrianCount"]
    assert len(model.fact_coverage) == ref_counts["FactCoverage"]

