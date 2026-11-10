"""Reusable RTL/LTR UI components, CSS styling, KPI cards, and AI Analyst response renderer."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src.ai_orchestrator import AIAnalystResponse
from src.config import COLORS, STUDY_DATES_AR, STUDY_LOCATION_AR


def inject_custom_css(lang: str = "ar") -> None:
    """Inject polished Arabic-first RTL (or English LTR) CSS styling."""
    direction = "rtl" if lang == "ar" else "ltr"
    text_align = "right" if lang == "ar" else "left"
    border_side = "border-right" if lang == "ar" else "border-left"

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');

    html, body, [class*="css"], .stMarkdown, .stText, .stDataFrame, .stTable, .stButton button, .stSelectbox, .stMultiSelect {{
        font-family: 'Cairo', 'Segoe UI', Tahoma, sans-serif !important;
    }}

    .main .block-container {{
        direction: {direction};
        text-align: {text_align};
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1440px;
    }}

    /* Header Banner */
    .hero-banner {{
        background: linear-gradient(135deg, #0F172A 0%, #0F766E 100%);
        color: #FFFFFF;
        padding: 1.35rem 1.75rem;
        border-radius: 14px;
        margin-bottom: 1.1rem;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.12);
        direction: {direction};
        text-align: {text_align};
    }}
    .hero-banner h1 {{
        margin: 0 0 0.35rem 0;
        font-size: 1.65rem;
        font-weight: 800;
        color: #FFFFFF;
    }}
    .hero-banner p {{
        margin: 0;
        font-size: 0.96rem;
        color: #E2E8F0;
        opacity: 0.95;
    }}

    /* KPI Card */
    .kpi-card {{
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        {border_side}: 5px solid #0F766E;
        border-radius: 12px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        direction: {direction};
        text-align: {text_align};
    }}
    .kpi-card.dir2 {{
        {border_side}: 5px solid #334155;
    }}
    .kpi-card.ped {{
        {border_side}: 5px solid #7C3AED;
    }}
    .kpi-card.matched {{
        {border_side}: 5px solid #0284C7;
        background: #F0F9FF;
    }}
    .kpi-title {{
        font-size: 0.84rem;
        font-weight: 700;
        color: #475569;
        margin-bottom: 0.25rem;
    }}
    .kpi-value {{
        font-size: 1.55rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
    }}
    .kpi-sub {{
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 0.3rem;
    }}

    /* AI Analyst 5-Part Response Cards */
    .ai-section-card {{
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        {border_side}: 4px solid #0F766E;
        border-radius: 10px;
        padding: 0.9rem 1.15rem;
        margin-bottom: 0.65rem;
        direction: {direction};
        text-align: {text_align};
    }}
    .ai-section-card.evidence {{
        {border_side}-color: #0284C7;
        background: #F8FAFC;
    }}
    .ai-section-card.insight {{
        {border_side}-color: #7C3AED;
    }}
    .ai-section-card.rec {{
        {border_side}-color: #059669;
        background: #ECFDF5;
    }}
    .ai-section-card.limit {{
        {border_side}-color: #D97706;
        background: #FFFBEB;
    }}
    .ai-section-header {{
        font-weight: 800;
        font-size: 0.92rem;
        margin-bottom: 0.35rem;
        color: #0F172A;
    }}

    /* Scope Status Pill */
    .scope-pill {{
        display: inline-block;
        background: #F1F5F9;
        border: 1px solid #CBD5E1;
        color: #0F172A;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        font-size: 0.84rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_hero_header(lang: str = "ar") -> None:
    """Render top study banner with location, dates, and verified baseline status."""
    if lang == "en":
        st.markdown(
            """
            <div class="hero-banner">
                <h1>🌉 El-Salab Bridge (Sandoub – El-Siniya) Traffic & Pedestrian AI Analyst</h1>
                <p>Field Survey: 12–17 October 2026 · 96 Validated Survey Records · 81 Observed Direction-Hours · 38 Matched Common Hours · Reconciled 100% with Power BI Semantic Model</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="hero-banner">
                <h1>🌉 منصة التحليل الذكي لحركة المرور وعبور المشاة — {STUDY_LOCATION_AR}</h1>
                <p>فترة الدراسة الميدانية: {STUDY_DATES_AR} · 96 سجل حصر معتمد · 81 فترة اتجاه-ساعة فعلية · 38 ساعة مشتركة متناظرة · مطابقة 100% لنموذج Power BI</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_kpi_card(
    title: str,
    value: str,
    subtitle: str,
    variant: str = "dir1",
) -> None:
    """Render a styled HTML KPI card."""
    st.markdown(
        f"""
        <div class="kpi-card {variant}">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ai_analyst_response(
    resp: AIAnalystResponse,
    key_prefix: str = "ai_resp",
) -> None:
    """Render the structured 5-part AI Analyst response, dynamic Plotly chart, CSV download, and lineage expander."""
    is_ar = resp.lang == "ar"
    badge_text = resp.status_badge_ar if is_ar else resp.status_badge_en

    # Engine mode badge + inherited context notes
    badge_color = "#059669" if resp.engine_mode == "gemini_live" else ("#DC2626" if resp.engine_mode == "security_guardrail" else "#0284C7")
    st.markdown(
        f"<div style='margin-bottom:0.5rem;'><span style='background:{badge_color};color:white;padding:3px 10px;border-radius:99px;font-size:0.78rem;font-weight:700;'>⚡ {badge_text}</span></div>",
        unsafe_allow_html=True,
    )

    notes = (
        resp.tool_result.inherited_context_notes_ar
        if is_ar
        else resp.tool_result.inherited_context_notes_en
    )
    if notes:
        st.caption("🔗 " + " | ".join(notes))

    # 1. Direct Answer
    lbl_direct = "1️⃣ الإجابة المباشرة (Direct Answer)" if is_ar else "1️⃣ Direct Answer"
    st.markdown(
        f"""
        <div class="ai-section-card">
            <div class="ai-section-header">{lbl_direct}</div>
            <div>{resp.direct_answer}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Numeric Evidence & Method
    lbl_ev = "2️⃣ الأدلة الرقمية وطريقة الحساب (Verified Numeric Evidence)" if is_ar else "2️⃣ Verified Numeric Evidence & Calculation Method"
    st.markdown(
        f"""
        <div class="ai-section-card evidence">
            <div class="ai-section-header">{lbl_ev}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(resp.evidence_and_method)

    # 3. Defensible Insight & 4. Recommendation side-by-side
    c1, c2 = st.columns(2)
    with c1:
        lbl_ins = "3️⃣ الاستنتاج التحليلي (Defensible Insight)" if is_ar else "3️⃣ Defensible Analytical Insight"
        st.markdown(
            f"""
            <div class="ai-section-card insight">
                <div class="ai-section-header">{lbl_ins}</div>
                <div>{resp.defensible_insight}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        lbl_rec = "4️⃣ التوصية العملية وسبب جدواها (Actionable Recommendation)" if is_ar else "4️⃣ Actionable Recommendation & Why It Works"
        st.markdown(
            f"""
            <div class="ai-section-card rec">
                <div class="ai-section-header">{lbl_rec}</div>
                <div>{resp.recommendation}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 5. Material Limitation
    lbl_lim = "5️⃣ حدود البيانات والمنهجية (Material Limitation & Scope Boundary)" if is_ar else "5️⃣ Material Limitation & Evidence Boundary"
    st.markdown(
        f"""
        <div class="ai-section-card limit">
            <div class="ai-section-header">{lbl_lim}</div>
            <div>{resp.limitation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dynamic Plotly Chart & Evidence Table
    tab_chart, tab_table = st.tabs(
        [
            "📈 الرسم البياني الديناميكي (Dynamic Chart)" if is_ar else "📈 Dynamic Chart",
            "📋 جدول الأدلة الرقمية وتنزيل CSV (Evidence Table)" if is_ar else "📋 Evidence Table & CSV Download",
        ]
    )
    with tab_chart:
        st.plotly_chart(
            resp.figure,
            use_container_width=True,
            key=f"{key_prefix}_chart",
        )
    with tab_table:
        ev_df = resp.tool_result.evidence_df
        if ev_df is not None and not ev_df.empty:
            st.dataframe(ev_df, use_container_width=True, hide_index=True)
            csv_bytes = ev_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 تنزيل جدول الأدلة (CSV)" if is_ar else "📥 Download Evidence Table (CSV)",
                data=csv_bytes,
                file_name=f"evidence_{resp.tool_result.tool_name}.csv",
                mime="text/csv",
                key=f"{key_prefix}_dl_btn",
            )
        else:
            st.info("لا توجد صفوف مطابقة في جدول الأدلة." if is_ar else "No rows in evidence table.")

    # How this was calculated expander
    exp_title = "🔍 طريقة الحساب وتفاصيل الأداة البرمجية / How this was calculated"
    with st.expander(exp_title, expanded=False):
        lineage = resp.tool_result.lineage_ar if is_ar else resp.tool_result.lineage_en
        st.markdown(f"**الأداة المنفذة (Executed Tool):** `{resp.tool_result.tool_name}`")
        st.markdown(f"**النطاق المطبق (Active Scope):** `{resp.tool_result.scope_used.describe_ar() if is_ar else resp.tool_result.scope_used.describe_en()}`")
        for k, v in lineage.items():
            st.markdown(f"- **{k}:** {v}")
        st.json(resp.tool_result.parameters)
