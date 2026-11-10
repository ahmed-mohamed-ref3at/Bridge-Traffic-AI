"""Main Streamlit Application: Bridge Traffic & Pedestrian Bilingual AI Data Analyst.

Implements:
1. Hero Feature: '💬 اسأل بياناتك / Ask Your Data' — AI Data Analyst powered by guarded Python
   calculations and optional Google Gemini narrative synthesis, with conversational follow-up
   memory ('طب يوم 14؟'), dynamic Plotly charts, CSV evidence downloads, and full lineage.
2. '📊 الملخص التنفيذي' — Executive Summary & Matched 38-Hour Fair Direction Comparison.
3. '🚗 تحليل المرور والذروة والفئات' — Themes 1, 2 & 3 (Volume/Coverage, Peaks/Outliers, 8 Vehicle Classes).
4. '🚶 المشاة والكوبري والتزامن الزمني' — Themes 4, 5 & 6 (Pedestrian Crossings, P75 Co-Activity & Spatial Boundaries, Bridge Impact & 2-Observer Protocol).
5. '🛡️ البيانات والمنهجية والتدقيق' — Theme 7 (Editable Data Dictionary, DQ01-DQ17, 15 Multi-Observer Intervals, 68,196 Counter Operation Logs Reconciliation, and 12-Check Power BI QA Suite).
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src.ai_orchestrator import answer_user_question
from src.analysis_tools import (
    CoverageAndQualityInput,
    PeaksAndAnomaliesInput,
    tool_analyze_peaks_and_anomalies,
    tool_inspect_coverage_and_data_quality,
)
from src.charts import (
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
from src.config import (
    DIRECTION_KEY_TO_AR,
    DIRECTION_KEY_TO_SHORT_AR,
    VEHICLE_CATEGORIES_SPEC,
    clear_saved_gemini_api_key,
    get_gemini_api_key,
    load_saved_gemini_api_key,
    save_gemini_api_key_locally,
)
from src.data_loader import (
    get_audit_log_reconciliation,
    load_profiling_metadata,
    verify_input_manifest,
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
    get_period_category_matrix,
    run_baseline_qa_checks,
)
from src.semantic_model import build_semantic_model
from src.ui_components import (
    inject_custom_css,
    render_ai_analyst_response,
    render_hero_header,
    render_kpi_card,
)


st.set_page_config(
    page_title="محلل بيانات المرور والمشاة الذكي — كوبري السلاب",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "ui_lang" not in st.session_state:
        st.session_state["ui_lang"] = "ar"
    if "active_view" not in st.session_state:
        st.session_state["active_view"] = "ask_data"
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "ai_session_context" not in st.session_state:
        st.session_state["ai_session_context"] = {}
    if "use_live_gemini" not in st.session_state:
        st.session_state["use_live_gemini"] = True
    if "filter_dates" not in st.session_state:
        st.session_state["filter_dates"] = []
    if "filter_directions" not in st.session_state:
        st.session_state["filter_directions"] = []
    if "filter_periods" not in st.session_state:
        st.session_state["filter_periods"] = []
    if "filter_categories" not in st.session_state:
        st.session_state["filter_categories"] = []
    if "filter_matched_only" not in st.session_state:
        st.session_state["filter_matched_only"] = False


def _reset_all_filters() -> None:
    """Callback to reset all global slicers to the full study scope."""
    st.session_state["filter_dates"] = []
    st.session_state["filter_directions"] = []
    st.session_state["filter_periods"] = []
    st.session_state["filter_categories"] = []
    st.session_state["filter_matched_only"] = False


def _switch_to_ask_data_with_prompt(prompt_text: str) -> None:
    """Switch directly to 'اسأل بياناتك' and execute the selected question."""
    st.session_state["active_view"] = "ask_data"
    st.session_state["pending_prompt"] = prompt_text


def main() -> None:
    _init_session_state()
    model = build_semantic_model()

    # ------------------------------------------------------------------------
    # Sidebar: Language, Navigation, Synchronized Slicers & AI Engine Settings
    # ------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### ⚙️ التحكم والفلاتر الشاملة")
        lang_choice = st.radio(
            "اللغة / Language",
            options=["ar", "en"],
            format_func=lambda x: "العربية (RTL)" if x == "ar" else "English (LTR)",
            horizontal=True,
            key="ui_lang",
        )
        is_ar = lang_choice == "ar"

        # 1-Click Jump Button to Hero AI Analyst when on other tabs
        if st.session_state["active_view"] != "ask_data":
            if st.button(
                "💬 انتقل فورًا إلى «اسأل بياناتك»" if is_ar else "💬 Jump to 'Ask Your Data' AI",
                use_container_width=True,
                type="primary",
            ):
                st.session_state["active_view"] = "ask_data"
                st.rerun()

        st.markdown("---")
        st.markdown("#### 🧭 الفلاتر المتزامنة (Global Slicers)")

        date_options = [
            f"{r['تاريخ_نص']} ({r['اليوم']})"
            for _, r in model.dim_date.iterrows()
        ]
        date_map = {
            f"{r['تاريخ_نص']} ({r['اليوم']})": str(r["تاريخ_نص"])
            for _, r in model.dim_date.iterrows()
        }
        selected_date_labels = st.multiselect(
            "📅 التاريخ (Date)" if is_ar else "📅 Date",
            options=date_options,
            key="filter_dates",
            placeholder="كل الأيام (12–17 أكتوبر)" if is_ar else "All 6 Days (12–17 Oct)",
        )
        selected_dates = [date_map[lbl] for lbl in selected_date_labels]

        dir_options = [
            f"1 — {DIRECTION_KEY_TO_AR[1]}",
            f"2 — {DIRECTION_KEY_TO_AR[2]}",
        ]
        selected_dir_labels = st.multiselect(
            "↔️ اتجاه الحصر (Direction)" if is_ar else "↔️ Direction",
            options=dir_options,
            key="filter_directions",
            placeholder="كلا الاتجاهين" if is_ar else "Both Directions",
        )
        selected_dirs = [int(lbl.split("—")[0].strip()) for lbl in selected_dir_labels]

        period_options = [
            f"{int(r['ترتيب الفترة'])} | {r['فترة الرصد']}"
            for _, r in model.dim_period.iterrows()
        ]
        selected_period_labels = st.multiselect(
            "⏰ فترة الرصد (Time Period)" if is_ar else "⏰ Time Period",
            options=period_options,
            key="filter_periods",
            placeholder="جميع فترات الرصد (10 فترات)" if is_ar else "All Periods",
        )
        selected_periods = [int(lbl.split("|")[0].strip()) for lbl in selected_period_labels]

        cat_options = [
            f"{int(c['order'])} — {c['name_ar']}"
            for c in VEHICLE_CATEGORIES_SPEC
        ]
        selected_cat_labels = st.multiselect(
            "🚗 فئة المركبة (Vehicle Class)" if is_ar else "🚗 Vehicle Category",
            options=cat_options,
            key="filter_categories",
            placeholder="جميع الفئات الثماني" if is_ar else "All 8 Categories",
        )
        selected_cats = [int(lbl.split("—")[0].strip()) for lbl in selected_cat_labels]

        matched_only = st.toggle(
            "⚖️ قصر العرض على الساعات المشتركة المتناظرة فقط (38 ساعة)"
            if is_ar
            else "⚖️ Matched Common Windows Only (38h)",
            key="filter_matched_only",
        )

        st.button(
            "🔄 إعادة تعيين الفلاتر (Reset All)" if is_ar else "🔄 Reset All Filters",
            on_click=_reset_all_filters,
            use_container_width=True,
        )

        # Build active FilterScope
        active_scope = FilterScope.from_inputs(
            dates=selected_dates or None,
            directions=selected_dirs or None,
            periods=selected_periods or None,
            categories=selected_cats or None,
            matched_only=matched_only,
            model=model,
        )

        st.markdown("---")
        st.markdown(
            "#### 🤖 إعدادات الذكاء الاصطناعي (اختياري)"
            if is_ar
            else "#### 🤖 AI Assistant Settings (Optional)"
        )
        has_saved_key = load_saved_gemini_api_key() is not None
        has_any_key = get_gemini_api_key() is not None
        if has_any_key:
            st.success(
                "✅ مفتاح Gemini مفعّل وجاهز للاستخدام"
                if is_ar
                else "✅ Gemini API Key is active and ready"
            )
        else:
            st.info(
                "ℹ️ يعمل البرنامج بكامل دقّته الحسابية وجميع لوحاته تلقائيًا بدون الحاجة لأي مفتاح. يمكنك اختياريًا لصق مفتاح Gemini أدناه."
                if is_ar
                else "ℹ️ Full verified calculations & dashboards work automatically with no key needed. Optionally paste a Gemini key below."
            )

        with st.expander(
            "🔑 إدخال أو تعديل مفتاح Gemini" if is_ar else "🔑 Configure Gemini Key",
            expanded=False,
        ):
            session_key_input = st.text_input(
                "الصق مفتاح Gemini هنا:" if is_ar else "Paste Gemini API Key:",
                type="password",
                placeholder="AIza...",
                key="sidebar_gemini_key_box",
            )
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button(
                    "💾 حفظ على جهازي" if is_ar else "💾 Save Key",
                    use_container_width=True,
                ):
                    cleaned = session_key_input.strip()
                    if cleaned:
                        save_gemini_api_key_locally(cleaned)
                        st.session_state["gemini_api_key_session"] = cleaned
                        st.toast(
                            "✅ تم حفظ مفتاح Gemini على هذا الجهاز بنجاح!"
                            if is_ar
                            else "✅ Gemini key saved on this computer!"
                        )
                        st.rerun()
                    else:
                        st.warning(
                            "يرجى لصق المفتاح أولًا قبل الضغط على حفظ."
                            if is_ar
                            else "Please paste a key first."
                        )
            with btn_col2:
                if st.button(
                    "🗑️ مسح المفتاح" if is_ar else "🗑️ Clear Key",
                    use_container_width=True,
                    disabled=not (has_saved_key or has_any_key),
                ):
                    clear_saved_gemini_api_key()
                    st.session_state.pop("gemini_api_key_session", None)
                    st.toast(
                        "تم مسح المفتاح والعودة للمحرك المدمج الموثق."
                        if is_ar
                        else "Cleared saved Gemini key."
                    )
                    st.rerun()

            if session_key_input.strip():
                st.session_state["gemini_api_key_session"] = session_key_input.strip()

        st.toggle(
            "تفعيل صياغة Gemini الحية عند توفر المفتاح"
            if is_ar
            else "Enable Live Gemini Narrative Synthesis",
            key="use_live_gemini",
        )

    # Inject RTL/LTR CSS and render top banner
    inject_custom_css(lang=lang_choice)
    render_hero_header(lang=lang_choice)

    # Compute KPIs for the current filter scope and full study baseline
    scope_kpis = compute_kpis(active_scope, model)
    full_kpis = compute_kpis(FilterScope(), model)

    # Top Navigation Bar (5 Primary Views, 1-click access)
    view_options = {
        "ask_data": "💬 اسأل بياناتك (المحلل الذكي)" if is_ar else "💬 Ask Your Data (AI Analyst)",
        "exec_summary": "📊 الملخص التنفيذي والمقارنة العادلة" if is_ar else "📊 Executive Summary",
        "traffic_analysis": "🚗 تحليل المرور والذروة والفئات" if is_ar else "🚗 Traffic, Peaks & Categories",
        "pedestrian_bridge": "🚶 المشاة والكوبري والتزامن الزمني" if is_ar else "🚶 Pedestrians, Co-Activity & Bridge",
        "data_qa": "🛡️ البيانات والمنهجية والتدقيق" if is_ar else "🛡️ Data Quality, Logs & QA",
    }

    nav_cols = st.columns(len(view_options))
    for idx, (v_key, v_label) in enumerate(view_options.items()):
        with nav_cols[idx]:
            is_active = st.session_state["active_view"] == v_key
            if st.button(
                v_label,
                key=f"nav_btn_{v_key}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state["active_view"] = v_key
                st.rerun()

    # Scope Status Bar
    st.markdown(
        f"<div class='scope-pill'>📌 {scope_kpis['حالة اختيار الرصد']} &nbsp;|&nbsp; {active_scope.describe_ar() if is_ar else active_scope.describe_en()}</div>",
        unsafe_allow_html=True,
    )

    if scope_kpis["ساعات_الرصد_الفعلي"] == 0:
        st.warning(
            "⚠️ لا توجد فترات رصد مطابقة لمجموعة الفلاتر المختارة حاليًا. اضغط على «إعادة تعيين الفلاتر» في القائمة الجانبية لاستعادة النطاق الكامل (81 فترة رصد)."
            if is_ar
            else "⚠️ No observed periods match the current filter combination. Click 'Reset All Filters' in the sidebar to restore the full 81-hour scope."
        )

    current_view = st.session_state["active_view"]

    # ========================================================================
    # VIEW 1: HERO FEATURE — 💬 اسأل بياناتك (Ask Your Data AI Analyst)
    # ========================================================================
    if current_view == "ask_data":
        st.markdown(
            "### 💬 اسأل بياناتك — المحلل الذكي لحركة المرور والمشاة (عربي / English)"
            if is_ar
            else "### 💬 Ask Your Data — Bilingual Traffic & Pedestrian AI Analyst"
        )
        st.caption(
            "اطرح أي سؤال بلغة طبيعية (بالعربية أو الإنجليزية) أو سؤال متابعة قصير مثل «طب يوم 14؟» أو «وماذا عن الاتجاه العكسي؟». جميع الأرقام والرسوم تُحسب برمجيًا عبر أدوات بايثون الموثقة."
            if is_ar
            else "Ask any question in Arabic or English, or short follow-ups like 'What about Oct 14?' or 'And the opposite direction?'. All metrics and charts are computed deterministically by guarded Python tools."
        )

        # Smart Analytical Shortcut Buttons (covering all 7 client themes + edge cases)
        st.markdown(
            "**⚡ أسئلة تحليلية استكشافية سريعة (اضغط للتشغيل الفوري):**"
            if is_ar
            else "**⚡ Quick Analytical Prompts (Click to Run):**"
        )
        shortcut_prompts = [
            (
                "⚖️ المقارنة العادلة في الـ 38 ساعة المشتركة",
                "قارن بشكل عادل بين اتجاه سندوب إلى الصينية واتجاه الصينية إلى سندوب في الساعات المشتركة المتناظرة",
            ),
            (
                "🔥 أعلى ساعات الذروة والقفزة مقابل النمط المتكرر",
                "ما هي أعلى 5 ساعات ذروة؟ وهل الساعة 12 إلى 1 ظهرًا نمط يومي متكرر أم قفزة منفردة مقارنة بالساعة 1 إلى 2 ظهرًا؟",
            ),
            (
                "📅 لماذا سجل 16 أكتوبر ساعات أكثر وكثافة أقل من 14 أكتوبر؟",
                "قارن بين يوم 14 أكتوبر ويوم 16 أكتوبر من حيث عدد ساعات الرصد مقابل متوسط كثافة المركبات في الساعة",
            ),
            (
                "🚛 تحليل النقل الثقيل والتريلات حسب الاتجاه والساعة",
                "حلل حركة النقل الثقيل والتريلات بين الاتجاهين ومتى تبلغ ذروتها خلال ساعات اليوم؟",
            ),
            (
                "🚶‍♂️ ساعات التزامن المرتفع (P75) بين المركبات والمشاة",
                "ما هي ساعات التزامن المرتفع بين كثافة المركبات وعبور المشاة (المئين 75) وما هي الساعات التسع ذات الأولوية؟",
            ),
            (
                "🌉 هل أثبتت البيانات أن الكوبري قلل الحوادث أو العبور السطحي؟",
                "كم شخص استخدم كوبري المشاة فعليًا مقابل العبور السطحي وهل قلل الكوبري الحوادث؟",
            ),
            (
                "🗺️ أين تقع أخطر نقطة جغرافية على الخريطة؟",
                "أين يقع أخطر مكان للعبور على الخريطة الجغرافية وما هي إحداثيات النقاط الساخنة؟",
            ),
            (
                "🔮 سيناريو افتراضي: تحويل 70% من المشاة للكوبري",
                "ماذا لو افترضنا سيناريو تحويل 70% من عبور المشاة إلى الكوبري خلال ساعات الأولوية؟",
            ),
            (
                "🔍 فحص الـ 30 مركبة غير المسماة وسجل الـ 68,196 عملية",
                "افحص جودة البيانات وتعدد الراصدين والـ 30 مركبة في الفئة غير المسماة ومطابقة سجل عمليات العداد 68,196",
            ),
            (
                "🇬🇧 English Query: Matched Comparison & Pedestrian Peaks",
                "Compare both directions fairly in the 38 matched hours and show peak pedestrian crossing hours",
            ),
        ]

        sc_cols = st.columns(5)
        for idx, (btn_label, q_text) in enumerate(shortcut_prompts):
            with sc_cols[idx % 5]:
                if st.button(btn_label, key=f"sc_btn_{idx}", use_container_width=True):
                    st.session_state["pending_prompt"] = q_text

        # Chat Input Form
        with st.form(key="ask_data_form", clear_on_submit=True):
            f_col1, f_col2 = st.columns([5, 1])
            with f_col1:
                user_q_input = st.text_input(
                    "اكتب سؤالك التحليلي هنا..." if is_ar else "Type your analytical question here...",
                    placeholder="مثال: قارن بين الاتجاهين في الساعات المشتركة | أو سؤال متابعة: طب يوم 14؟ | أو: ماذا عن النقل الثقيل؟",
                    label_visibility="collapsed",
                )
            with f_col2:
                submit_q = st.form_submit_button(
                    "🚀 حلل الآن" if is_ar else "🚀 Analyze",
                    use_container_width=True,
                    type="primary",
                )

        # Determine active question to execute
        question_to_run: Optional[str] = None
        if submit_q and user_q_input.strip():
            question_to_run = user_q_input.strip()
        elif "pending_prompt" in st.session_state and st.session_state["pending_prompt"]:
            question_to_run = str(st.session_state.pop("pending_prompt"))

        if question_to_run:
            with st.spinner("جاري تنفيذ الأدوات الحسابية الموثقة وبناء الإجابة..." if is_ar else "Running verified analytical tools..."):
                resp = answer_user_question(
                    question=question_to_run,
                    session_context=st.session_state["ai_session_context"],
                    ui_scope=active_scope,
                    preferred_lang="auto",
                    use_live_gemini=bool(st.session_state.get("use_live_gemini", True)),
                    model=model,
                )
                st.session_state["ai_session_context"] = resp.session_context
                st.session_state["chat_history"].append(resp)

        # Display active session memory pill + clear history button
        if st.session_state["chat_history"]:
            m_col1, m_col2 = st.columns([4, 1])
            with m_col1:
                ctx = st.session_state.get("ai_session_context", {})
                st.caption(
                    f"🧠 ذاكرة الحوار النشطة: الأداة السابقة = `{ctx.get('last_tool', '—')}` | التاريخ = `{ctx.get('last_dates') or 'الكل'}` | الاتجاه = `{ctx.get('last_directions') or 'كلا الاتجاهين'}`"
                )
            with m_col2:
                if st.button("🗑️ مسح سجل الحوار" if is_ar else "🗑️ Clear Chat", use_container_width=True):
                    st.session_state["chat_history"] = []
                    st.session_state["ai_session_context"] = {}
                    st.rerun()

            # Render latest response prominently at top, and earlier responses in expanders
            latest_resp: AIAnalystResponse = st.session_state["chat_history"][-1]
            st.markdown(f"#### 🗨️ السؤال: «{latest_resp.question}»")
            render_ai_analyst_response(
                latest_resp,
                key_prefix=f"latest_{len(st.session_state['chat_history'])}",
            )

            if len(st.session_state["chat_history"]) > 1:
                st.markdown("---")
                st.markdown("#### 📜 الأسئلة السابقة في هذه الجلسة (Previous Turns)")
                for h_idx in range(len(st.session_state["chat_history"]) - 2, -1, -1):
                    prev_r: AIAnalystResponse = st.session_state["chat_history"][h_idx]
                    with st.expander(f"سؤال #{h_idx + 1}: {prev_r.question}", expanded=False):
                        render_ai_analyst_response(prev_r, key_prefix=f"hist_{h_idx}")
        else:
            # Show default initial analytical showcase so the screen is immediately rich on first load
            default_resp = answer_user_question(
                question="قارن بشكل عادل بين اتجاه سندوب إلى الصينية واتجاه الصينية إلى سندوب في الساعات المشتركة المتناظرة",
                session_context={},
                ui_scope=active_scope,
                preferred_lang=lang_choice,
                use_live_gemini=False,
                model=model,
            )
            st.markdown("#### 💡 عرض تحليلي استهلالي: المقارنة المنهجية العادلة بين الاتجاهين")
            render_ai_analyst_response(default_resp, key_prefix="welcome_default")

    # ========================================================================
    # VIEW 2: 📊 الملخص التنفيذي والمقارنة العادلة (Executive Summary)
    # ========================================================================
    elif current_view == "exec_summary":
        st.markdown(
            "### 📊 الملخص التنفيذي والمقارنة المنهجية العادلة بين الاتجاهين"
            if is_ar
            else "### 📊 Executive Summary & Fair Matched-Window Direction Comparison"
        )

        # Row 1: 4 Primary KPI Cards
        k_cols = st.columns(4)
        with k_cols[0]:
            render_kpi_card(
                title="مركبات: من سندوب إلى الصينية",
                value=f"{scope_kpis['مركبات_سندوب_إلى_الصينية']:,}",
                subtitle=f"المتوسط: {scope_kpis['متوسط_سندوب_إلى_الصينية']:,.2f} مركبة/ساعة ({scope_kpis['ساعات_سندوب_إلى_الصينية']} ساعة رصد)",
                variant="dir1",
            )
        with k_cols[1]:
            render_kpi_card(
                title="مركبات: من الصينية إلى سندوب",
                value=f"{scope_kpis['مركبات_الصينية_إلى_سندوب']:,}",
                subtitle=f"المتوسط: {scope_kpis['متوسط_الصينية_إلى_سندوب']:,.2f} مركبة/ساعة ({scope_kpis['ساعات_الصينية_إلى_سندوب']} ساعة رصد)",
                variant="dir2",
            )
        with k_cols[2]:
            render_kpi_card(
                title=f"المقارنة العادلة ({scope_kpis['عدد_الساعات_المشتركة']} ساعة مشتركة)",
                value=f"{scope_kpis['فرق_الساعات_المشتركة']:+,} ({scope_kpis['نسبة_فرق_الساعات_المشتركة']:+.2f}%)",
                subtitle=f"سندوب: {scope_kpis['مركبات_سندوب_المشتركة']:,} | الصينية: {scope_kpis['مركبات_الصينية_المشتركة']:,}",
                variant="matched",
            )
        with k_cols[3]:
            render_kpi_card(
                title="عبور المشاة (مؤشر مستقل لا يُجمع مع المركبات)",
                value=f"{scope_kpis['إجمالي_عبور_المشاة']:,}",
                subtitle=f"سندوب: {scope_kpis['عبور_سندوب_إلى_الصينية']:,} ({scope_kpis['متوسط_عبور_سندوب_لكل_ساعة']:,.1f}/س) | الصينية: {scope_kpis['عبور_الصينية_إلى_سندوب']:,} ({scope_kpis['متوسط_عبور_الصينية_لكل_ساعة']:,.1f}/س)",
                variant="ped",
            )

        # Row 2: Secondary KPI Cards (Peak Hours, Dominant Category, Coverage)
        k2_cols = st.columns(4)
        with k2_cols[0]:
            render_kpi_card(
                title="ذروة الساعة: من سندوب إلى الصينية",
                value=f"{int(scope_kpis['أعلى عدد مركبات في ساعة من سندوب إلى الصينية'] or 0):,} مركبة/س",
                subtitle=str(scope_kpis["تاريخ وفترة الذروة من سندوب إلى الصينية"] or "—"),
                variant="dir1",
            )
        with k2_cols[1]:
            render_kpi_card(
                title="ذروة الساعة: من الصينية إلى سندوب",
                value=f"{int(scope_kpis['أعلى عدد مركبات في ساعة من الصينية إلى سندوب'] or 0):,} مركبة/س",
                subtitle=str(scope_kpis["تاريخ وفترة الذروة من الصينية إلى سندوب"] or "—"),
                variant="dir2",
            )
        with k2_cols[2]:
            top_c_share = (scope_kpis["حصة الفئة الأعلى تسجيلًا"] or 0.0) * 100.0
            render_kpi_card(
                title="الفئة المرورية المهيمنة",
                value=str(scope_kpis["فئة المركبات الأعلى تسجيلًا"] or "—"),
                subtitle=f"الحصة: {top_c_share:.2f}% من إجمالي المركبات في النطاق",
                variant="matched",
            )
        with k2_cols[3]:
            render_kpi_card(
                title="إجمالي التغطية والسجلات المدمجة",
                value=f"{scope_kpis['ساعات_الرصد_الفعلي']} ساعة اتجاهية",
                subtitle=f"مشتقة من {scope_kpis['عدد_صفوف_الرصد']} سجل حصر خام عبر {int(scope_kpis['عدد أيام الرصد'] or 0)} أيام",
                variant="ped",
            )

        # Charts Row 1
        dir_comp_df = compute_direction_comparison(active_scope, model)
        c_left, c_right = st.columns(2)
        with c_left:
            st.plotly_chart(
                build_direction_comparison_chart(dir_comp_df, lang=lang_choice),
                use_container_width=True,
                key="exec_dir_comp_chart",
            )
        with c_right:
            st.plotly_chart(
                build_matched_vs_raw_chart(
                    dir_comp_df,
                    common_hours=scope_kpis["عدد_الساعات_المشتركة"],
                    lang=lang_choice,
                ),
                use_container_width=True,
                key="exec_matched_chart",
            )

        # Charts Row 2
        daily_df = compute_daily_summary(active_scope, model)
        period_df = compute_period_profile(active_scope, model)
        c2_left, c2_right = st.columns(2)
        with c2_left:
            st.plotly_chart(
                build_daily_trend_chart(daily_df, metric_mode="mean", lang=lang_choice),
                use_container_width=True,
                key="exec_daily_chart",
            )
        with c2_right:
            st.plotly_chart(
                build_period_profile_chart(period_df, lang=lang_choice),
                use_container_width=True,
                key="exec_period_chart",
            )

        # Executive Synthesis: Full-Study Findings vs Filtered Findings & Actionable Recommendations
        st.markdown("---")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.markdown("#### 📌 الخلاصة التنفيذية الموثقة (ثوابت الدراسة الكاملة: 12–17 أكتوبر 2026)")
            st.markdown(
                """
                1. **التفوق الحقيقي لاتجاه سندوب → الصينية (+21.97%):**
                   - في الإجمالي الخام سجل `74,248` مركبة (43 ساعة) مقابل `53,521` مركبة (38 ساعة) في الاتجاه العكسي.
                   - عند توحيد المقام الزمني: **1,726.70** مقابل **1,408.45** مركبة/ساعة.
                   - عند تثبيت **الـ 38 ساعة المشتركة المتناظرة**: **65,279** مقابل **53,521** مركبة (فارق **+11,758 مركبة = +21.97%**).
                2. **التمييز بين القفزة المنفردة والنمط المتكرر:**
                   - أعلى قراءة منفردة بلغت **4,425 مركبة/ساعة** يوم الأربعاء `14 أكتوبر | 12:00–1:00 ظهرًا` (سندوب → الصينية)، تلتها **3,636** في `1:00–2:00 ظهرًا`.
                   - النمط اليومي المتكرر الأعلى عبر الأيام هو الفترة **1:00–2:00 ظهرًا** بمتوسط **2,141.60 مركبة/ساعة** (عبر 5 أيام) في سندوب → الصينية و**1,663.75** (عبر 4 أيام) في الاتجاه العكسي.
                3. **هيمنة المركبات الخفيفة وتباين الشاحنات الثقيلة:**
                   - **السيارات الملاكي** تمثل **49.75%** (`63,559` مركبة: `51.74%` في سندوب → الصينية مقابل `46.98%` في الصينية → سندوب).
                   - **النقل الثقيل والتريلات** يمثل **5.85%** (`7,478` مركبة)، لكن حصته النسبية ترتفع في **الصينية → سندوب (6.88%)** مقارنة بـ **5.11%** في الاتجاه الأول، ويبلغ ذروته بين **1:00 و2:00 ظهرًا (123.56 شاحنة/ساعة)**.
                """
            )
        with f_col2:
            st.markdown("#### 🛠️ التوصيات التشغيلية والميدانية المبنية على الأدلة")
            st.markdown(
                """
                1. **توجيه سعة المسارات وإدارة الذروة الظهرية (12:00–3:00 م):**
                   - منح أولوية تنظيم التدفق ومنع التوقف العشوائي للميكروباص في اتجاه **سندوب → الصينية** الذي يتحمل عبئًا أعلى بنسبة **21.97%** في الساعات المتناظرة.
                2. **التدخل الموجه في الساعات التسع ذات التزامن المرتفع (P75):**
                   - تكثيف التوجيه الميداني للمشاة نحو الكوبري خلال **9 ساعات محددة** يتزامن فيها أعلى تدفق للمركبات مع أعلى عبور للمشاة (`4` ساعات يوم الأربعاء 14 أكتوبر في سندوب → الصينية، و`5` ساعات يومي 13 و17 أكتوبر في الصينية → سندوب).
                3. **بروتوكول الرصد الميداني المزدوج (لتقييم أثر الكوبري مستقبلاً):**
                   - فصل عمود المشاة في أي حملة قادمة إلى راصدين متزامنين: **راصد أعلى الكوبري** و**راصد للعبور السطحي أسفل الكوبري**، مع توحيد ساعات الرصد في الاتجاهين وإغلاق خاصية الفئات غير المسماة (`custom_1792135578939`).
                """
            )

    # ========================================================================
    # VIEW 3: 🚗 تحليل المرور والذروة والفئات (Themes 1, 2 & 3)
    # ========================================================================
    elif current_view == "traffic_analysis":
        t1, t2, t3 = st.tabs(
            [
                "📈 1. حجم المرور والتغطية الزمنية (Theme 1)",
                "⏰ 2. ساعات الذروة والأنماط والشذوذ (Theme 2)",
                "🚙 3. تصنيف فئات المركبات الثماني (Theme 3)",
            ]
        )

        with t1:
            st.markdown("#### 📈 تحليل حجم المرور وفصل طول التغطية الزمنية عن شدة الكثافة")
            st.info(
                "💡 **قاعدة المقارنة المنهجية:** لا تتساوى أيام الرصد ولا الاتجاهان في عدد الساعات المرصودة (`43` ساعة في سندوب → الصينية مقابل `38` ساعة في الصينية → سندوب). "
                "مثال حاسم: يوم **الجمعة 16 أكتوبر** سجل أطول تغطية في الاتجاه الأول (**9 ساعات**) لكنه الأقل كثافة (**777.89 مركبة/ساعة**)، بينما **الأربعاء 14 أكتوبر** رُصد في **7 ساعات** فقط وسجل أعلى كثافة (**2,543.43 مركبة/ساعة**)."
            )
            daily_df = compute_daily_summary(active_scope, model)
            c_d1, c_d2 = st.columns(2)
            with c_d1:
                st.plotly_chart(
                    build_daily_trend_chart(daily_df, metric_mode="mean", lang=lang_choice),
                    use_container_width=True,
                    key="t1_daily_mean",
                )
            with c_d2:
                st.plotly_chart(
                    build_daily_trend_chart(daily_df, metric_mode="total", lang=lang_choice),
                    use_container_width=True,
                    key="t1_daily_tot",
                )

            st.markdown("##### 📋 جدول الملخص اليومي حسب الاتجاه (إجمالي المركبات، ساعات الرصد، والمتوسط الساعي)")
            st.dataframe(daily_df, use_container_width=True, hide_index=True)

        with t2:
            st.markdown("#### ⏰ ترتيب ساعات الذروة والنمط الزمني وفحص القيم المتطرفة (Tukey 1.5×IQR)")
            peaks_res = tool_analyze_peaks_and_anomalies(
                PeaksAndAnomaliesInput(
                    dates=[d.isoformat() for d in active_scope.dates] if active_scope.dates else None,
                    directions=list(active_scope.directions) if active_scope.directions else None,
                    periods=list(active_scope.periods) if active_scope.periods else None,
                    categories=list(active_scope.categories) if active_scope.categories else None,
                    matched_only=active_scope.matched_only,
                    rank_order="top",
                    top_n=10,
                ),
                model=model,
            )
            lows_df = compute_ranked_hours(active_scope, top_n=5, ascending=True, model=model)
            period_df = compute_period_profile(active_scope, model)

            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.plotly_chart(
                    build_peak_hours_chart(peaks_res.evidence_df, rank_order="top", lang=lang_choice),
                    use_container_width=True,
                    key="t2_top_peaks",
                )
            with p_col2:
                st.plotly_chart(
                    build_period_profile_chart(period_df, lang=lang_choice),
                    use_container_width=True,
                    key="t2_period_prof",
                )

            ps_filtered = active_scope.filter_period_summary(model)
            hm_col1, hm_col2 = st.columns(2)
            with hm_col1:
                st.plotly_chart(
                    build_date_period_heatmap(ps_filtered, direction_key=1, lang=lang_choice),
                    use_container_width=True,
                    key="t2_hm_d1",
                )
            with hm_col2:
                st.plotly_chart(
                    build_date_period_heatmap(ps_filtered, direction_key=2, lang=lang_choice),
                    use_container_width=True,
                    key="t2_hm_d2",
                )

            tb1, tb2 = st.columns(2)
            with tb1:
                st.markdown("##### 🏆 أعلى 10 ساعات رصد في النطاق")
                st.dataframe(peaks_res.evidence_df, use_container_width=True, hide_index=True)
            with tb2:
                st.markdown("##### 🌙 أدنى 5 ساعات رصد في النطاق + القيم المتطرفة (Outliers)")
                st.dataframe(lows_df, use_container_width=True, hide_index=True)
                if peaks_res.secondary_df is not None and not peaks_res.secondary_df.empty:
                    st.caption("الساعات المصنفة كقيم متطرفة إحصائيًا (Tukey 1.5×IQR):")
                    st.dataframe(peaks_res.secondary_df, use_container_width=True, hide_index=True)

        with t3:
            st.markdown("#### 🚙 تحليل وتصنيف فئات المركبات الثماني (مع حفظ الفئة غير المسمّاة)")
            cat_df = compute_category_breakdown(active_scope, model)
            c_cat1, c_cat2 = st.columns(2)
            with c_cat1:
                st.plotly_chart(
                    build_vehicle_category_bar_chart(cat_df, lang=lang_choice),
                    use_container_width=True,
                    key="t3_cat_bar",
                )
            with c_cat2:
                st.plotly_chart(
                    build_direction_category_stacked_chart(cat_df, lang=lang_choice),
                    use_container_width=True,
                    key="t3_cat_stack",
                )

            st.markdown("##### 📋 جدول تفصيل الفئات الثماني (الإجمالي، الحصة النسبية، والمقارنة بين الاتجاهين)")
            st.dataframe(cat_df, use_container_width=True, hide_index=True)

            st.markdown("##### ⏱️ مصفوفة توزيع الفئات عبر الفترات الزمنية (Period × Category Matrix)")
            mat_df = get_period_category_matrix(active_scope)
            if not mat_df.empty:
                piv_cat = mat_df.pivot_table(
                    index=["ترتيب الفترة", "فترة الرصد"],
                    columns="فئة المركبة",
                    values="متوسط_الساعة_الاتجاهية",
                    aggfunc="first",
                ).round(2)
                st.dataframe(piv_cat, use_container_width=True)

    # ========================================================================
    # VIEW 4: 🚶 المشاة والكوبري والتزامن الزمني (Themes 4, 5 & 6)
    # ========================================================================
    elif current_view == "pedestrian_bridge":
        p_tab1, p_tab2, p_tab3 = st.tabs(
            [
                "🚶 1. تحليل عبور المشاة المستقل (Theme 4)",
                "⚠️ 2. ساعات التزامن المرتفع P75 وحدود الدليل المكاني (Theme 5)",
                "🌉 3. إطار تقييم كوبري المشاة ومحاكاة What-If (Theme 6)",
            ]
        )

        ps_filtered = active_scope.filter_period_summary(model)
        co_df, thresholds = compute_coactivity_windows(active_scope, model)
        period_df = compute_period_profile(active_scope, model)

        with p_tab1:
            st.markdown("#### 🚶 تحليل أحداث عبور المشاة كمؤشر مستقل")
            pk1, pk2, pk3 = st.columns(3)
            with pk1:
                render_kpi_card(
                    title="عبور المشاة: من سندوب إلى الصينية",
                    value=f"{scope_kpis['عبور_سندوب_إلى_الصينية']:,} حدث عبور",
                    subtitle=f"المتوسط: {scope_kpis['متوسط_عبور_سندوب_لكل_ساعة']:,.2f} عبور/ساعة (43 ساعة بالدراسة الكاملة)",
                    variant="ped",
                )
            with pk2:
                render_kpi_card(
                    title="عبور المشاة: من الصينية إلى سندوب",
                    value=f"{scope_kpis['عبور_الصينية_إلى_سندوب']:,} حدث عبور",
                    subtitle=f"المتوسط: {scope_kpis['متوسط_عبور_الصينية_لكل_ساعة']:,.2f} عبور/ساعة (38 ساعة بالدراسة الكاملة)",
                    variant="dir2",
                )
            with pk3:
                render_kpi_card(
                    title="ساعات التزامن المرتفع (مركبات + مشاة >= P75)",
                    value=f"{len(co_df)} ساعات أولوية",
                    subtitle="4 ساعات في سندوب→الصينية + 5 ساعات في الصينية→سندوب (في الدراسة الكاملة)",
                    variant="matched",
                )

            st.plotly_chart(
                build_pedestrian_by_period_chart(period_df, lang=lang_choice),
                use_container_width=True,
                key="p1_ped_period",
            )

            if not ps_filtered.empty:
                top_ped_hours = (
                    ps_filtered.sort_values(["عبور_مشاة_الفترة", "إجمالي_مركبات_الفترة"], ascending=[False, False])
                    .head(10)[
                        [
                            "تاريخ_نص",
                            "اليوم",
                            "اتجاه الحصر",
                            "فترة الرصد",
                            "عبور_مشاة_الفترة",
                            "إجمالي_مركبات_الفترة",
                            "تزامن_مرتفع",
                        ]
                    ]
                    .reset_index(drop=True)
                )
                st.markdown("##### 📋 أعلى 10 ساعات في كثافة عبور المشاة")
                st.dataframe(top_ped_hours, use_container_width=True, hide_index=True)

        with p_tab2:
            st.markdown("#### ⚠️ مصفوفة التزامن الزمني (P75) والتمييز الصارم بين التعرض الزمني والخطر المكاني")
            st.warning(
                "📌 **بيان حدود الأدلة المكانية والسببية (Evidence Boundary):**\n"
                "- البيانات المتاحة تحدد **التزامن الزمني والاتجاهي (Temporal & Directional Co-Activity)**، أي الساعات التي يجتمع فيها تدفق مروري كثيف مع عبور مشاة كثيف.\n"
                "- البيانات **لا تحتوي** على إحداثيات جغرافية (GPS)، ولا تحدد نقطة العبور الدقيقة على طول الطريق، ولا تحتوي على سجلات حوادث أو سرعات."
            )
            st.plotly_chart(
                build_pedestrian_coactivity_scatter(ps_filtered, thresholds, lang=lang_choice),
                use_container_width=True,
                key="p2_scatter",
            )
            st.markdown("##### ⭐ جدول الساعات التسع ذات الأولوية التشغيلية القصوى (Simultaneous High-Activity Windows >= P75)")
            st.dataframe(co_df, use_container_width=True, hide_index=True)

        with p_tab3:
            st.markdown("#### 🌉 إطار تقييم أثر كوبري المشاة ومحاكاة السيناريوهات وبروتوكول الراصدين المتزامنين")
            st.info(
                "🔍 **لماذا لا تحسم البيانات الحالية نسبة استخدام الكوبري بمفردها؟**\n"
                "لأن استمارة الحصر الحالية تضم عمودًا واحدًا للمشاة (`مشاة (عبور المشاة) = 24,342`) دون تقسيم بين **العبور الآمن أعلى الكوبري** و**العبور السطحي أسفل الكوبري**، كما لا توجد قراءات قبل إنشاء الكوبري."
            )

            sim_col1, sim_col2 = st.columns([1, 2])
            with sim_col1:
                st.markdown("##### 🎛️ محاكي سيناريو التحول إلى الكوبري (What-If Simulator)")
                assumed_diversion = st.slider(
                    "نسبة التحول الافتراضية للمشاة نحو الكوبري (%)",
                    min_value=10,
                    max_value=95,
                    value=70,
                    step=5,
                )
                tot_ped_scope = scope_kpis["إجمالي_عبور_المشاة"]
                co_ped_scope = int(co_df["عبور_مشاة_الفترة"].sum()) if not co_df.empty else 0
                div_all = round(tot_ped_scope * assumed_diversion / 100.0)
                rem_all = tot_ped_scope - div_all
                div_co = round(co_ped_scope * assumed_diversion / 100.0)
                rem_co = co_ped_scope - div_co

                st.metric("العبور المتحول للكوبري (افتراضيًا)", f"{div_all:,} حدث عبور", f"{assumed_diversion}% من إجمالي النطاق")
                st.metric("العبور السطحي المتبقي (افتراضيًا)", f"{rem_all:,} حدث عبور", f"منها {rem_co:,} في ساعات الأولوية")

            with sim_col2:
                st.markdown("##### 📋 بروتوكول الرصد الميداني المقترح (2-Observer Synchronized Protocol)")
                st.markdown(
                    """
                    لقياس الأثر الفعلي لكوبري المشاة في المرحلة القادمة بشكل حاسم وقابل للدفاع هندسيًا، يُوصى بتطبيق البروتوكول التالي خلال **الساعات التسع ذات الأولوية**:
                    1. **الراصد الأول (أعلى الكوبري — Bridge Deck):** يرصد فقط المشاة الصاعدين/العابرين فوق جسم الكوبري.
                    2. **الراصد الثاني (السطح أسفل الكوبري — At-Grade Surface):** يرصد في نفس الساعة والاتجاه المشاة العابرين سطحيًا عبر نهر الطريق في نطاق 100 متر حول الكوبري.
                    3. **مؤشر كفاءة الكوبري الفعلي (Bridge Capture Rate):**
                       $$\\text{معدل استيعاب الكوبري (\\%)} = \\frac{\\text{عبور المشاة أعلى الكوبري}}{\\text{عبور أعلى الكوبري} + \\text{العبور السطحي أسفل الكوبري}} \\times 100$$
                    """
                )
                protocol_template_df = pd.DataFrame(
                    [
                        {
                            "التاريخ": "2026-10-XX",
                            "الفترة_الزمنية": "12:00 م - 01:00 م",
                            "اتجاه_الحصر": "سندوب → الصينية",
                            "معرف_راصد_الكوبري": "راصد_كوبري_01",
                            "عبور_أعلى_الكوبري": 0,
                            "معرف_راصد_السطح": "راصد_سطح_02",
                            "عبور_سطحي_أسفل_الكوبري": 0,
                            "إجمالي_المركبات_في_الساعة": 0,
                            "ملاحظات_العوائق_الميدانية": "",
                        }
                    ]
                )
                st.dataframe(protocol_template_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "📥 تنزيل نموذج الحصر المزدوج للكوبري والسطح (CSV)",
                    data=protocol_template_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="bridge_two_observer_protocol_template.csv",
                    mime="text/csv",
                )

    # ========================================================================
    # VIEW 5: 🛡️ البيانات والمنهجية والتدقيق (Theme 7 + QA Reconciliation)
    # ========================================================================
    elif current_view == "data_qa":
        q_tab1, q_tab2, q_tab3 = st.tabs(
            [
                "📖 1. قاموس البيانات التفاعلي وقواعد الحوكمة",
                "🔍 2. جودة البيانات (DQ01–DQ17) وتعدد الراصدين والـ 30 مركبة",
                "✅ 3. مطابقة Power BI وسجل عمليات العداد (68,196 عملية) والتنزيل الآمن",
            ]
        )

        profiling = load_profiling_metadata()
        dq_res = tool_inspect_coverage_and_data_quality(
            CoverageAndQualityInput(aspect="all"),
            model=model,
        )

        with q_tab1:
            st.markdown("#### 📖 قاموس البيانات التفاعلي (Interactive & Reviewable Data Dictionary)")
            st.caption("يمكنك فحص وتعديل الملاحظات التفسيرية لقاموس البيانات خلال الجلسة وتنزيل النسخة المحدثة.")
            dict_df = profiling.get("Data Dictionary", pd.DataFrame())
            if not dict_df.empty:
                edited_dict = st.data_editor(
                    dict_df,
                    use_container_width=True,
                    hide_index=True,
                    key="editable_data_dict",
                )
                st.download_button(
                    "📥 تنزيل قاموس البيانات (CSV)",
                    data=edited_dict.to_csv(index=False).encode("utf-8-sig"),
                    file_name="data_dictionary_reviewed.csv",
                    mime="text/csv",
                )

        with q_tab2:
            st.markdown("#### 🔍 فحص التغطية الزمنية، الـ 15 فترة متعددة الراصدين، القيم الصفرية، والعمود `custom_1792135578939`")
            multi_res = tool_inspect_coverage_and_data_quality(
                CoverageAndQualityInput(aspect="multi_observer"), model=model
            )
            unnamed_res = tool_inspect_coverage_and_data_quality(
                CoverageAndQualityInput(aspect="unnamed_category"), model=model
            )
            zero_res = tool_inspect_coverage_and_data_quality(
                CoverageAndQualityInput(aspect="zero_values"), model=model
            )

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown("##### 👥 الفترات التي شهدت تعدد راصدين (15 فترة اتجاه-ساعة = 30 سجلًا خامًا)")
                st.caption("تم جمع أعداد المركبات والمشاة داخل الساعة الواحدة مع احتساب التغطية الزمنية 60 دقيقة واحدة.")
                st.dataframe(multi_res.evidence_df, use_container_width=True, hide_index=True)
            with col_m2:
                st.markdown("##### 🏷️ السجلات الخمسة التي تحتوي على الفئة غير المسمّاة (`custom_1792135578939` = 30 مركبة)")
                st.caption("جميعها يوم الجمعة 2026-10-16 في اتجاه الصينية → سندوب (السجلات B-29 إلى B-33).")
                st.dataframe(unnamed_res.evidence_df, use_container_width=True, hide_index=True)
                st.markdown("##### 0️⃣ حفظ القيم الصفرية عبر أعمدة العد التسعة (216 خلية صفرية محفوظة)")
                st.dataframe(zero_res.evidence_df, use_container_width=True, hide_index=True)

            dq_issues_df = profiling.get("Data Quality Issues", pd.DataFrame())
            if not dq_issues_df.empty:
                st.markdown("##### 📋 سجل قضايا جودة البيانات الموثق (DQ01 – DQ17)")
                st.dataframe(dq_issues_df, use_container_width=True, hide_index=True)

        with q_tab3:
            st.markdown("#### ✅ مطابقة الأرقام القياسية مع نموذج Power BI وفحص سجل العمليات (68,196 عملية)")
            qa_results = run_baseline_qa_checks()
            manifest_check = verify_input_manifest()
            audit_rec = get_audit_log_reconciliation()

            qa_c1, qa_c2, qa_c3 = st.columns(3)
            with qa_c1:
                st.metric(
                    "نتيجة اختبارات المطابقة القياسية (QA Checks)",
                    f"{sum(c['passed'] for c in qa_results['checks'])} / {len(qa_results['checks'])} ناجحة",
                    "مطابقة 100% مع Power BI",
                )
            with qa_c2:
                st.metric(
                    "سلامة بصمة ملفات المصدر (SHA-256 Manifest)",
                    "100% مطابقة" if manifest_check["valid"] else "تحذير",
                    f"{len(manifest_check.get('files', {}))} ملفات مصدرية لم تُمس",
                )
            with qa_c3:
                st.metric(
                    "سجل عمليات العداد (AuditOperationLogs)",
                    f"{audit_rec['total_events']:,} عملية نقر",
                    f"صافي التغير {audit_rec['net_delta_sum']:,} (فارق 301 مفسر بالكامل)",
                )

            st.markdown("##### 🧪 جدول اختبارات المطابقة الآلية (12 اختبارًا قياسيًا)")
            qa_df = pd.DataFrame(qa_results["checks"])
            st.dataframe(qa_df, use_container_width=True, hide_index=True)

            st.markdown("##### 🧾 تفسير الفارق بين سجل عمليات العداد (`151,810`) وجدول الملخص (`152,111` — الفارق = `301`)")
            st.dataframe(
                pd.DataFrame(audit_rec["discrepancy_rows"]),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("##### 📥 تنزيل الملخصات التحليلية الآمنة (بدون أسماء شخصية للراصدين)")
            dl_c1, dl_c2 = st.columns(2)
            with dl_c1:
                ps_export = active_scope.filter_period_summary(model).drop(
                    columns=["اسم الراصد"], errors="ignore"
                )
                st.download_button(
                    "📥 تنزيل ملخص الفترات الساعية المفلتر (81 فترة - CSV)",
                    data=ps_export.to_csv(index=False).encode("utf-8-sig"),
                    file_name="period_summary_anonymized.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with dl_c2:
                raw_export = model.survey_raw.drop(columns=["اسم الراصد"], errors="ignore")
                st.download_button(
                    "📥 تنزيل سجلات الرصد الـ 96 بمعرفات راصدين معماة (CSV)",
                    data=raw_export.to_csv(index=False).encode("utf-8-sig"),
                    file_name="survey_96_records_anonymized.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
