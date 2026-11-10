"""Unit tests for Pydantic tool parameter validation, security whitelists, and Plotly chart builders."""
from __future__ import annotations

import plotly.graph_objects as go
import pytest
from pydantic import ValidationError

from src.analysis_tools import (
    AggregationQueryInput,
    CoverageAndQualityInput,
    DirectionComparisonInput,
    EvidenceBoundaryInput,
    PeaksAndAnomaliesInput,
    PedestrianCoactivityInput,
    ScopeInputModel,
    WhatIfScenarioInput,
    tool_analyze_peaks_and_anomalies,
    tool_analyze_pedestrian_and_coactivity,
    tool_check_evidence_boundaries,
    tool_compare_directions_fairly,
    tool_evaluate_what_if_scenario,
    tool_get_overview_and_kpis,
    tool_inspect_coverage_and_data_quality,
    tool_query_traffic_aggregations,
)
from src.charts import (
    build_chart_for_analysis_result,
    build_daily_trend_chart,
    build_date_period_heatmap,
    build_direction_category_stacked_chart,
    build_direction_comparison_chart,
    build_matched_vs_raw_chart,
    build_peak_hours_chart,
    build_pedestrian_by_period_chart,
    build_pedestrian_coactivity_scatter,
    build_period_profile_chart,
    build_vehicle_category_bar_chart,
)
from src.metrics_engine import (
    FilterScope,
    compute_category_breakdown,
    compute_coactivity_windows,
    compute_daily_summary,
    compute_direction_comparison,
    compute_period_profile,
    compute_ranked_hours,
)


def test_pydantic_whitelist_blocks_invalid_dimensions_and_metrics() -> None:
    """Verify that unknown group_by dimensions or metrics raise ValidationError."""
    with pytest.raises(ValidationError):
        AggregationQueryInput(group_by=["unknown_column"])

    with pytest.raises(ValidationError):
        AggregationQueryInput(metric="sum_vehicles_and_pedestrians")

    with pytest.raises(ValidationError):
        ScopeInputModel(dates=["2024-05-01"])


def test_pydantic_blocks_code_and_sql_injection_strings() -> None:
    """Verify that code execution or SQL injection tokens are rejected by input validators."""
    with pytest.raises(ValidationError):
        AggregationQueryInput(group_by=["__import__('os').system('dir')"])

    with pytest.raises(ValidationError):
        AggregationQueryInput(metric="DROP TABLE FactVehicleCount")

    with pytest.raises(ValidationError):
        ScopeInputModel(dates=["eval(1+1)"])


def test_all_8_guarded_tools_return_valid_results_and_charts() -> None:
    """Execute each of the 8 guarded analytical tools and verify its evidence table and Plotly chart."""
    r1 = tool_get_overview_and_kpis(ScopeInputModel())
    assert not r1.evidence_df.empty
    f1 = build_chart_for_analysis_result(r1, lang="ar")
    assert isinstance(f1, go.Figure) and len(f1.data) >= 1

    r2 = tool_query_traffic_aggregations(
        AggregationQueryInput(group_by=["date", "direction"], metric="mean_vehicles_per_hour")
    )
    assert len(r2.evidence_df) == 12  # 6 days * 2 directions
    f2 = build_chart_for_analysis_result(r2, lang="ar")
    assert isinstance(f2, go.Figure) and len(f2.data) >= 1

    r2_cat = tool_query_traffic_aggregations(
        AggregationQueryInput(group_by=["vehicle_category", "direction"], metric="total_vehicles")
    )
    assert len(r2_cat.evidence_df) == 16  # 8 categories * 2 directions
    f2_cat = build_chart_for_analysis_result(r2_cat, lang="en")
    assert isinstance(f2_cat, go.Figure)

    r3 = tool_compare_directions_fairly(DirectionComparisonInput(target="vehicles"))
    assert len(r3.evidence_df) == 4
    assert r3.headline_metrics["common_hours"] == 38
    assert r3.headline_metrics["matched_diff"] == 11758
    f3 = build_chart_for_analysis_result(r3, lang="ar")
    assert isinstance(f3, go.Figure)

    r4 = tool_analyze_peaks_and_anomalies(PeaksAndAnomaliesInput(rank_order="top", top_n=5))
    assert len(r4.evidence_df) == 5
    assert int(r4.evidence_df["إجمالي_المركبات"].iloc[0]) == 4425
    f4 = build_chart_for_analysis_result(r4, lang="ar")
    assert isinstance(f4, go.Figure)

    r5 = tool_analyze_pedestrian_and_coactivity(
        PedestrianCoactivityInput(focus="coactivity_windows")
    )
    assert len(r5.evidence_df) == 9
    f5 = build_chart_for_analysis_result(r5, lang="ar")
    assert isinstance(f5, go.Figure)

    r6 = tool_inspect_coverage_and_data_quality(CoverageAndQualityInput(aspect="unnamed_category"))
    assert len(r6.evidence_df) == 5
    assert int(r6.evidence_df["custom_1792135578939"].sum()) == 30
    f6 = build_chart_for_analysis_result(r6, lang="ar")
    assert isinstance(f6, go.Figure)

    r7 = tool_evaluate_what_if_scenario(
        WhatIfScenarioInput(scenario_type="pedestrian_bridge_diversion", change_pct=70.0)
    )
    assert not r7.evidence_df.empty
    f7 = build_chart_for_analysis_result(r7, lang="ar")
    assert isinstance(f7, go.Figure)

    r8 = tool_check_evidence_boundaries(EvidenceBoundaryInput(topic="accidents"))
    assert r8.is_boundary_or_refusal is True
    f8 = build_chart_for_analysis_result(r8, lang="ar")
    assert isinstance(f8, go.Figure)


def test_dashboard_chart_builders_ar_and_en() -> None:
    """Verify all standalone dashboard chart builders in both Arabic and English."""
    scope = FilterScope()
    dir_df = compute_direction_comparison(scope)
    daily_df = compute_daily_summary(scope)
    period_df = compute_period_profile(scope)
    ranked_df = compute_ranked_hours(scope, top_n=10)
    cat_df = compute_category_breakdown(scope)
    co_df, thresholds = compute_coactivity_windows(scope)
    ps_df = scope.filter_period_summary()

    for lang in ("ar", "en"):
        assert len(build_direction_comparison_chart(dir_df, lang=lang).data) == 2
        assert len(build_matched_vs_raw_chart(dir_df, lang=lang).data) == 2
        assert len(build_daily_trend_chart(daily_df, metric_mode="mean", lang=lang).data) == 2
        assert len(build_period_profile_chart(period_df, lang=lang).data) == 2
        assert len(build_peak_hours_chart(ranked_df, lang=lang).data) == 1
        assert len(build_date_period_heatmap(ps_df, direction_key=1, lang=lang).data) == 1
        assert len(build_vehicle_category_bar_chart(cat_df, lang=lang).data) == 1
        assert len(build_direction_category_stacked_chart(cat_df, lang=lang).data) == 8
        assert len(build_pedestrian_coactivity_scatter(ps_df, thresholds, lang=lang).data) >= 2
        assert len(build_pedestrian_by_period_chart(period_df, lang=lang).data) == 2
