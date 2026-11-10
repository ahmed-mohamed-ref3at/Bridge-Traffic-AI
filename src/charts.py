"""Dynamic Plotly chart builders for the Dashboard and AI Data Analyst ('اسأل بياناتك').

All charts follow the Power BI Enhanced Report color palette, preserve Arabic RTL
readability, explicitly state grain and denominator in titles/subtitles, and never
mix vehicles and pedestrians into a single combined KPI axis.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.analysis_tools import AnalysisToolResult
from src.config import (
    CATEGORY_PALETTE,
    COLORS,
    DIRECTION_KEY_TO_AR,
    DIRECTION_KEY_TO_EN,
    DIRECTION_KEY_TO_SHORT_AR,
)

FONT_FAMILY = "Cairo, Segoe UI, Tahoma, sans-serif"

DIRECTION_COLOR_MAP = {
    DIRECTION_KEY_TO_AR[1]: COLORS["dir1"],
    DIRECTION_KEY_TO_SHORT_AR[1]: COLORS["dir1"],
    DIRECTION_KEY_TO_EN[1]: COLORS["dir1"],
    DIRECTION_KEY_TO_AR[2]: COLORS["dir2"],
    DIRECTION_KEY_TO_SHORT_AR[2]: COLORS["dir2"],
    DIRECTION_KEY_TO_EN[2]: COLORS["dir2"],
}


def _apply_common_layout(
    fig: go.Figure,
    title: str,
    subtitle: str = "",
    height: int = 410,
    lang: str = "ar",
) -> go.Figure:
    """Apply consistent Arabic/English typography, clean gridlines, and responsive margins."""
    full_title = (
        f"<b>{title}</b><br><span style='font-size:12px;color:#475569;'>{subtitle}</span>"
        if subtitle
        else f"<b>{title}</b>"
    )
    fig.update_layout(
        title={
            "text": full_title,
            "x": 0.98 if lang == "ar" else 0.02,
            "xanchor": "right" if lang == "ar" else "left",
            "font": {"family": FONT_FAMILY, "size": 16, "color": COLORS["text_primary"]},
        },
        font={"family": FONT_FAMILY, "size": 12, "color": COLORS["text_primary"]},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        height=height,
        margin={"l": 45, "r": 45, "t": 78, "b": 55},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left" if lang == "ar" else "right",
            "x": 0.0 if lang == "ar" else 1.0,
            "bgcolor": "rgba(255,255,255,0.85)",
        },
        hoverlabel={
            "bgcolor": "#0F172A",
            "font_size": 12,
            "font_family": FONT_FAMILY,
        },
    )
    fig.update_xaxes(showgrid=False, linecolor="#CBD5E1")
    fig.update_yaxes(gridcolor="#E2E8F0", zerolinecolor="#CBD5E1")
    return fig


def build_direction_comparison_chart(
    dir_comp_df: pd.DataFrame,
    lang: str = "ar",
) -> go.Figure:
    """Compare directions across Hourly Mean (all observed hours) and Matched Common Hours Mean."""
    if dir_comp_df is None or dir_comp_df.empty:
        fig = go.Figure()
        return _apply_common_layout(fig, "لا توجد بيانات لعرض مقارنة الاتجاهين", lang=lang)

    d1 = dir_comp_df[dir_comp_df["مفتاح الاتجاه"] == 1].iloc[0]
    d2 = dir_comp_df[dir_comp_df["مفتاح الاتجاه"] == 2].iloc[0]

    if lang == "en":
        categories = [
            "Mean Vehicles / Observed Hour",
            "Mean Vehicles / Common Matched Hour",
            "Peak Single Hour",
        ]
        name_1 = DIRECTION_KEY_TO_EN[1] + f" ({int(d1['ساعات_الرصد'])}h)"
        name_2 = DIRECTION_KEY_TO_EN[2] + f" ({int(d2['ساعات_الرصد'])}h)"
        title = "Fair Direction Comparison: Normalized Rate & Matched Windows"
        subtitle = "Normalizing for coverage difference (43h vs 38h) and comparing identical (date, period) slots"
    else:
        categories = [
            "متوسط المركبات / ساعة رصد فعلية",
            "متوسط المركبات / ساعة مشتركة متناظرة",
            "أعلى ساعة رصد منفردة (الذروة)",
        ]
        name_1 = f"{DIRECTION_KEY_TO_SHORT_AR[1]} ({int(d1['ساعات_الرصد'])} ساعة)"
        name_2 = f"{DIRECTION_KEY_TO_SHORT_AR[2]} ({int(d2['ساعات_الرصد'])} ساعة)"
        title = "مقارنة الكثافة المرورية بين الاتجاهين (توحيد المقام والفترات المشتركة)"
        subtitle = "الوحدة: مركبة/ساعة | يراعي اختلاف تغطية الرصد (43 ساعة مقابل 38 ساعة)"

    vals_1 = [
        float(d1["متوسط_المركبات_لكل_ساعة"]),
        float(d1["متوسط_المشتركة_لكل_ساعة"]),
        float(d1["ذروة_الساعة"]),
    ]
    vals_2 = [
        float(d2["متوسط_المركبات_لكل_ساعة"]),
        float(d2["متوسط_المشتركة_لكل_ساعة"]),
        float(d2["ذروة_الساعة"]),
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name=name_1,
            x=categories,
            y=vals_1,
            marker_color=COLORS["dir1"],
            text=[f"{v:,.1f}" if i < 2 else f"{v:,.0f}" for i, v in enumerate(vals_1)],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            name=name_2,
            x=categories,
            y=vals_2,
            marker_color=COLORS["dir2"],
            text=[f"{v:,.1f}" if i < 2 else f"{v:,.0f}" for i, v in enumerate(vals_2)],
            textposition="outside",
        )
    )
    fig.update_layout(barmode="group", yaxis_title="مركبة / ساعة" if lang == "ar" else "Vehicles / Hour")
    max_y = max(vals_1 + vals_2 + [100]) * 1.18
    fig.update_yaxes(range=[0, max_y])
    return _apply_common_layout(fig, title, subtitle, height=400, lang=lang)


def build_matched_vs_raw_chart(
    dir_comp_df: pd.DataFrame,
    common_hours: int = 38,
    lang: str = "ar",
) -> go.Figure:
    """Compare Raw Total Vehicles vs Matched Common Windows Total Vehicles."""
    if dir_comp_df is None or dir_comp_df.empty:
        return _apply_common_layout(go.Figure(), "لا توجد بيانات", lang=lang)

    d1 = dir_comp_df[dir_comp_df["مفتاح الاتجاه"] == 1].iloc[0]
    d2 = dir_comp_df[dir_comp_df["مفتاح الاتجاه"] == 2].iloc[0]

    if lang == "en":
        x_labels = [
            f"Raw Total ({int(d1['ساعات_الرصد'])}h vs {int(d2['ساعات_الرصد'])}h)",
            f"Matched Common Windows ({common_hours}h in both)",
        ]
        n1, n2 = DIRECTION_KEY_TO_EN[1], DIRECTION_KEY_TO_EN[2]
        title = "Total Observed Vehicles vs Matched Common-Window Vehicles"
        subtitle = f"In the {common_hours} shared hours: {int(d1['المركبات_المشتركة']):,} vs {int(d2['المركبات_المشتركة']):,} (+21.97%)"
    else:
        x_labels = [
            f"الإجمالي الخام ({int(d1['ساعات_الرصد'])} س مقابل {int(d2['ساعات_الرصد'])} س)",
            f"الساعات المشتركة المتناظرة ({common_hours} ساعة لكلا الاتجاهين)",
        ]
        n1, n2 = DIRECTION_KEY_TO_SHORT_AR[1], DIRECTION_KEY_TO_SHORT_AR[2]
        diff = int(d1["المركبات_المشتركة"]) - int(d2["المركبات_المشتركة"])
        pct = (diff / int(d2["المركبات_المشتركة"]) * 100.0) if int(d2["المركبات_المشتركة"]) else 0.0
        title = "إجمالي المركبات الخام مقابل مركبات الساعات المشتركة المتناظرة"
        subtitle = f"في الساعات الـ {common_hours} المشتركة: فارق +{diff:,} مركبة (+{pct:.2f}%) لصالح سندوب → الصينية"

    v1 = [int(d1["إجمالي_المركبات"]), int(d1["المركبات_المشتركة"])]
    v2 = [int(d2["إجمالي_المركبات"]), int(d2["المركبات_المشتركة"])]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name=n1,
            x=x_labels,
            y=v1,
            marker_color=COLORS["dir1"],
            text=[f"{val:,}" for val in v1],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            name=n2,
            x=x_labels,
            y=v2,
            marker_color=COLORS["dir2"],
            text=[f"{val:,}" for val in v2],
            textposition="outside",
        )
    )
    fig.update_layout(barmode="group", yaxis_title="عدد المركبات" if lang == "ar" else "Vehicles")
    fig.update_yaxes(range=[0, max(v1 + v2 + [1000]) * 1.18])
    return _apply_common_layout(fig, title, subtitle, height=400, lang=lang)


def build_daily_trend_chart(
    daily_df: pd.DataFrame,
    metric_mode: str = "mean",
    lang: str = "ar",
) -> go.Figure:
    """Build daily trend chart by direction showing hourly mean (or total) + observed coverage hours."""
    if daily_df is None or daily_df.empty:
        return _apply_common_layout(go.Figure(), "لا توجد بيانات يومية ضمن النطاق", lang=lang)

    df = daily_df.copy()
    mean_col = (
        "متوسط_المركبات_في_الساعة"
        if "متوسط_المركبات_في_الساعة" in df.columns
        else "متوسط_المركبات_لكل_ساعة"
    )
    tot_col = "إجمالي_المركبات"
    hrs_col = "ساعات_الرصد"
    dir_col = "اتجاه_مختصر" if "اتجاه_مختصر" in df.columns else "اتجاه الحصر"

    df["تسمية_اليوم"] = df["اليوم"].astype(str) + " (" + df["تاريخ_نص"].astype(str).str[-5:] + ")"
    y_col = mean_col if metric_mode == "mean" else tot_col

    fig = go.Figure()
    if "مفتاح الاتجاه" in df.columns:
        for di in sorted(df["مفتاح الاتجاه"].unique()):
            sub = df[df["مفتاح الاتجاه"] == di]
            if sub.empty:
                continue
            color = COLORS["dir1"] if di == 1 else COLORS["dir2"]
            d_name = DIRECTION_KEY_TO_SHORT_AR.get(int(di), "الاتجاه")
            if lang == "en":
                d_name = DIRECTION_KEY_TO_EN.get(int(di), d_name)
            fig.add_trace(
                go.Bar(
                    name=d_name,
                    x=sub["تسمية_اليوم"],
                    y=sub[y_col],
                    marker_color=color,
                    text=[
                        f"{val:,.0f}<br><span style='font-size:10px'>({int(h)} س)</span>"
                        for val, h in zip(sub[y_col], sub[hrs_col])
                    ],
                    textposition="outside",
                    customdata=sub[[tot_col, hrs_col, mean_col]].values,
                    hovertemplate=(
                        "<b>%{x}</b> — " + d_name + "<br>"
                        "متوسط المركبات/ساعة: %{customdata[2]:,.2f}<br>"
                        "إجمالي المركبات اليومي: %{customdata[0]:,}<br>"
                        "ساعات الرصد الفعلية: %{customdata[1]} ساعة<extra></extra>"
                    ),
                )
            )
    else:
        d_name = "كلا الاتجاهين" if lang == "ar" else "Both Directions"
        fig.add_trace(
            go.Bar(
                name=d_name,
                x=df["تسمية_اليوم"],
                y=df[y_col],
                marker_color=COLORS["dir1"],
                text=[
                    f"{val:,.0f}<br><span style='font-size:10px'>({int(h)} س)</span>"
                    for val, h in zip(df[y_col], df[hrs_col])
                ],
                textposition="outside",
            )
        )

    if metric_mode == "mean":
        title = (
            "متوسط كثافة المركبات لكل ساعة رصد حسب اليوم والاتجاه"
            if lang == "ar"
            else "Daily Mean Vehicles per Observed Hour by Direction"
        )
        subtitle = (
            "يوضح بين قوسين (عدد ساعات الرصد الفعلية) لكل يوم لفصل طول التغطية عن شدة الازدحام"
            if lang == "ar"
            else "Parentheses show observed hours per day to separate coverage duration from traffic intensity"
        )
        y_title = "متوسط المركبات / ساعة" if lang == "ar" else "Mean Vehicles / Hour"
    else:
        title = (
            "إجمالي المركبات المرصودة يوميًا حسب الاتجاه (مع عدد ساعات الرصد)"
            if lang == "ar"
            else "Daily Total Observed Vehicles by Direction (with Coverage Hours)"
        )
        subtitle = (
            "تنبيه: تختلف ساعات الرصد بين الأيام (من 5 إلى 9 ساعات)؛ راجع المتوسط الساعي للمقارنة المعيارية"
            if lang == "ar"
            else "Note: Observed coverage varies across days (5 to 9 hours)"
        )
        y_title = "إجمالي المركبات" if lang == "ar" else "Total Vehicles"

    fig.update_layout(barmode="group", yaxis_title=y_title)
    max_val = float(df[y_col].max()) if not df.empty else 100.0
    fig.update_yaxes(range=[0, max_val * 1.24])
    return _apply_common_layout(fig, title, subtitle, height=420, lang=lang)


def build_period_profile_chart(
    period_df: pd.DataFrame,
    lang: str = "ar",
) -> go.Figure:
    """Build time-of-day profile showing mean vehicles/hour by period and direction."""
    if period_df is None or period_df.empty:
        return _apply_common_layout(go.Figure(), "لا توجد بيانات للفترات الزمنية", lang=lang)

    df = period_df.copy().sort_values(["ترتيب الفترة", "مفتاح الاتجاه"] if "مفتاح الاتجاه" in period_df.columns else ["ترتيب الفترة"])
    mean_col = (
        "متوسط_المركبات_في_الساعة"
        if "متوسط_المركبات_في_الساعة" in df.columns
        else "متوسط_المركبات_لكل_ساعة"
    )
    cnt_col = "فترات_الرصد" if "فترات_الرصد" in df.columns else "ساعات_الرصد"

    fig = go.Figure()
    if "مفتاح الاتجاه" in df.columns:
        for di in sorted(df["مفتاح الاتجاه"].unique()):
            sub = df[df["مفتاح الاتجاه"] == di]
            color = COLORS["dir1"] if di == 1 else COLORS["dir2"]
            name = DIRECTION_KEY_TO_SHORT_AR[int(di)] if lang == "ar" else DIRECTION_KEY_TO_EN[int(di)]
            fig.add_trace(
                go.Scatter(
                    name=name,
                    x=sub["فترة الرصد"],
                    y=sub[mean_col],
                    mode="lines+markers+text",
                    line={"color": color, "width": 3},
                    marker={"size": 9, "color": color},
                    text=[f"{v:,.0f}" for v in sub[mean_col]],
                    textposition="top center",
                    customdata=sub[["إجمالي_المركبات", cnt_col]].values,
                    hovertemplate=(
                        "<b>%{x}</b> — " + name + "<br>"
                        "متوسط المركبات/ساعة: %{y:,.2f}<br>"
                        "إجمالي المركبات في الفترة: %{customdata[0]:,}<br>"
                        "عدد أيام الرصد لهذه الساعة: %{customdata[1]} يوم<extra></extra>"
                    ),
                )
            )
    else:
        fig.add_trace(
            go.Bar(
                name="متوسط المركبات/ساعة",
                x=df["فترة الرصد"],
                y=df[mean_col],
                marker_color=COLORS["dir1"],
                text=[f"{v:,.0f}" for v in df[mean_col]],
                textposition="outside",
            )
        )

    title = (
        "النمط الزمني لكثافة المركبات عبر ساعات اليوم (مقارنة القفزة المنفردة بالنمط المتكرر)"
        if lang == "ar"
        else "Time-of-Day Traffic Intensity Profile Across Observed Days"
    )
    subtitle = (
        "الساعة 12–1 ظهرًا في سندوب→الصينية رُصدت يوم 14 أكتوبر فقط (4,425)، بينما 1–2 ظهرًا ذروة متكررة عبر 5 أيام (2,141.60/ساعة)"
        if lang == "ar"
        else "12–1 PM in Dir 1 was observed once on Oct 14 (4,425), whereas 1–2 PM is a sustained peak across 5 days (2,141.60/hr)"
    )
    fig.update_layout(yaxis_title="متوسط المركبات / ساعة" if lang == "ar" else "Mean Vehicles / Hour")
    max_y = float(df[mean_col].max()) if not df.empty else 100.0
    fig.update_yaxes(range=[0, max_y * 1.2])
    return _apply_common_layout(fig, title, subtitle, height=420, lang=lang)


def build_peak_hours_chart(
    ranked_df: pd.DataFrame,
    rank_order: str = "top",
    lang: str = "ar",
) -> go.Figure:
    """Horizontal bar chart of top-N or bottom-N observed direction-period hours."""
    if ranked_df is None or ranked_df.empty:
        return _apply_common_layout(go.Figure(), "لا توجد ساعات مرصودة في النطاق", lang=lang)

    df = ranked_df.head(10).copy()
    v_col = "إجمالي_المركبات" if "إجمالي_المركبات" in df.columns else "المركبات_المختارة"
    df["تسمية_مختصرة"] = (
        "#"
        + df["الرتبة"].astype(str)
        + " | "
        + df["تاريخ_نص"].astype(str).str[-5:]
        + " ("
        + df["اليوم"].astype(str)
        + ") | "
        + df["فترة الرصد"].astype(str)
    )
    # Reverse order so #1 is at top of horizontal bar chart
    df = df.iloc[::-1].reset_index(drop=True)
    bar_colors = [
        COLORS["dir1"] if "سندوب إلى الصينية" in str(d) or "سندوب → الصينية" in str(d) else COLORS["dir2"]
        for d in df["اتجاه الحصر"]
    ]

    fig = go.Figure(
        go.Bar(
            x=df[v_col],
            y=df["تسمية_مختصرة"],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{int(v):,} مركبة ({d})" for v, d in zip(df[v_col], df["اتجاه الحصر"])],
            textposition="auto",
            customdata=df[["اتجاه الحصر", "عبور_مشاة_الفترة", "عدد السجلات"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "الاتجاه: %{customdata[0]}<br>"
                "المركبات: %{x:,} مركبة/ساعة<br>"
                "عبور المشاة في نفس الساعة: %{customdata[1]:,} حدث عبور<br>"
                "عدد سجلات الرصد المدمجة: %{customdata[2]}<extra></extra>"
            ),
        )
    )

    if rank_order == "bottom":
        title = "أهدأ ساعات الرصد (الأدنى كثافة مرورية)" if lang == "ar" else "Quietest Observed Hours (Lowest Volume)"
    else:
        title = "أعلى ساعات الذروة المرورية على مستوى (التاريخ × الفترة × الاتجاه)" if lang == "ar" else "Top Peak Traffic Hours (Date × Period × Direction)"
    subtitle = (
        "الوحدة: مركبة/ساعة (بعد جمع سجلات تعدد الراصدين داخل نفس الساعة والاتجاه)"
        if lang == "ar"
        else "Grain: 60-minute direction-period (multi-observer records summed within slot)"
    )
    fig.update_layout(xaxis_title="عدد المركبات في الساعة" if lang == "ar" else "Vehicles in Hour")
    return _apply_common_layout(fig, title, subtitle, height=430, lang=lang)


def build_date_period_heatmap(
    ps_df: pd.DataFrame,
    direction_key: Optional[int] = 1,
    lang: str = "ar",
) -> go.Figure:
    """Build Date × Period heatmap explicitly distinguishing unobserved slots from observed traffic."""
    if ps_df is None or ps_df.empty:
        return _apply_common_layout(go.Figure(), "لا توجد بيانات للخريطة الحرارية", lang=lang)

    sub = ps_df.copy()
    if direction_key in (1, 2) and "مفتاح الاتجاه" in sub.columns:
        sub = sub[sub["مفتاح الاتجاه"] == direction_key]
        dir_title = DIRECTION_KEY_TO_SHORT_AR[direction_key] if lang == "ar" else DIRECTION_KEY_TO_EN[direction_key]
    else:
        dir_title = "كلا الاتجاهين" if lang == "ar" else "Both Directions"

    if sub.empty:
        return _apply_common_layout(go.Figure(), f"لا توجد بيانات للاتجاه ({dir_title})", lang=lang)

    v_col = "المركبات_المختارة" if "المركبات_المختارة" in sub.columns else "إجمالي_مركبات_الفترة"
    piv = sub.pivot_table(
        index=["تاريخ_نص", "اليوم"],
        columns=["ترتيب الفترة", "فترة الرصد"],
        values=v_col,
        aggfunc="sum",
    )
    piv = piv.sort_index(level=0).sort_index(axis=1, level=0)

    y_labels = [f"{idx[1]} ({idx[0][-5:]})" for idx in piv.index]
    x_labels = [col[1] for col in piv.columns]
    z_vals = piv.values
    text_vals = [
        [f"{int(v):,}" if pd.notna(v) else "غير مرصود" for v in row]
        for row in z_vals
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals,
            x=x_labels,
            y=y_labels,
            text=text_vals,
            texttemplate="%{text}",
            colorscale="Tealgrn" if direction_key != 2 else "Blues",
            hoverongaps=False,
            colorbar={"title": "مركبة/ساعة" if lang == "ar" else "Veh/Hr"},
        )
    )
    title = (
        f"مصفوفة الكثافة الساعية والتغطية الزمنية — {dir_title}"
        if lang == "ar"
        else f"Hourly Traffic & Coverage Heatmap — {dir_title}"
    )
    subtitle = (
        "الخلايا المكتوب عليها «غير مرصود» تمثل ساعات خارج جدول الرصد الميداني وليست صفر مركبات"
        if lang == "ar"
        else "Cells labeled 'غير مرصود' are unobserved hours, never treated as zero traffic"
    )
    return _apply_common_layout(fig, title, subtitle, height=400, lang=lang)


def build_vehicle_category_bar_chart(
    cat_df: pd.DataFrame,
    lang: str = "ar",
) -> go.Figure:
    """Horizontal bar chart of all 8 vehicle categories preserving custom_1792135578939."""
    if cat_df is None or cat_df.empty:
        return _apply_common_layout(go.Figure(), "لا توجد بيانات لفئات المركبات", lang=lang)

    df = cat_df.sort_values("إجمالي_المركبات", ascending=True).reset_index(drop=True)
    label_col = "فئة المركبة" if lang == "ar" else "category_en"
    share_col = (
        "النسبة_من_إجمالي_المركبات"
        if "النسبة_من_إجمالي_المركبات" in df.columns
        else "النسبة_المئوية"
    )

    colors = [
        CATEGORY_PALETTE[(int(o) - 1) % len(CATEGORY_PALETTE)]
        for o in df["ترتيب الفئة"]
    ]

    fig = go.Figure(
        go.Bar(
            x=df["إجمالي_المركبات"],
            y=df[label_col],
            orientation="h",
            marker_color=colors,
            text=[
                f"{int(v):,} ({s:.2f}%)" if s >= 0.1 else f"{int(v):,} ({s:.3f}%)"
                for v, s in zip(df["إجمالي_المركبات"], df[share_col])
            ],
            textposition="outside",
        )
    )
    max_x = float(df["إجمالي_المركبات"].max()) if not df.empty else 100.0
    fig.update_xaxes(range=[0, max_x * 1.25], title="عدد المركبات" if lang == "ar" else "Vehicles")
    title = (
        "توزيع المركبات حسب الفئات الثماني المعتمدة (شاملاً الفئة غير المسمّاة)"
        if lang == "ar"
        else "Vehicle Breakdown Across All 8 Categories (Including Unnamed Category)"
    )
    subtitle = (
        "السيارات الملاكي تمثل نحو نصف الحركة (49.75%)، والعمود custom_1792135578939 محفوظ بدقة (30 مركبة = 0.0235%)"
        if lang == "ar"
        else "Private cars represent 49.75% of flow; custom_1792135578939 is preserved (30 vehicles = 0.0235%)"
    )
    return _apply_common_layout(fig, title, subtitle, height=420, lang=lang)


def build_direction_category_stacked_chart(
    cat_df: pd.DataFrame,
    lang: str = "ar",
) -> go.Figure:
    """100% stacked bar chart comparing the 8-category mix across the two directions."""
    if cat_df is None or cat_df.empty or "حصة_سندوب_إلى_الصينية_%" not in cat_df.columns:
        return build_vehicle_category_bar_chart(cat_df, lang=lang)

    df = cat_df.sort_values("ترتيب الفئة").reset_index(drop=True)
    dirs = (
        [DIRECTION_KEY_TO_SHORT_AR[1], DIRECTION_KEY_TO_SHORT_AR[2]]
        if lang == "ar"
        else [DIRECTION_KEY_TO_EN[1], DIRECTION_KEY_TO_EN[2]]
    )

    fig = go.Figure()
    for idx, r in df.iterrows():
        c_name = str(r["فئة المركبة"] if lang == "ar" else r["category_en"])
        shares = [float(r["حصة_سندوب_إلى_الصينية_%"]), float(r["حصة_الصينية_إلى_سندوب_%"])]
        counts = [int(r["سندوب_إلى_الصينية"]), int(r["الصينية_إلى_سندوب"])]
        color = CATEGORY_PALETTE[(int(r["ترتيب الفئة"]) - 1) % len(CATEGORY_PALETTE)]
        fig.add_trace(
            go.Bar(
                name=c_name,
                x=dirs,
                y=shares,
                marker_color=color,
                text=[f"{s:.1f}%" if s >= 4.0 else "" for s in shares],
                textposition="inside",
                customdata=counts,
                hovertemplate=(
                    "<b>%{x}</b> — " + c_name + "<br>"
                    "الحصة داخل الاتجاه: %{y:.2f}%<br>"
                    "عدد المركبات: %{customdata:,}<extra></extra>"
                ),
            )
        )

    fig.update_layout(barmode="stack", yaxis_title="النسبة المئوية (%)" if lang == "ar" else "Share (%)")
    fig.update_yaxes(range=[0, 100])
    title = (
        "مقارنة التركيب النسبي لفئات المركبات (100%) بين الاتجاهين"
        if lang == "ar"
        else "100% Stacked Vehicle Category Mix by Direction"
    )
    subtitle = (
        "الملاكي أعلى نسبيًا في سندوب→الصينية (51.74%)، بينما النقل الثقيل أعلى نسبيًا في الصينية→سندوب (6.88% مقابل 5.11%)"
        if lang == "ar"
        else "Private cars are higher in Dir 1 (51.74%), while Heavy Trucks have a higher share in Dir 2 (6.88% vs 5.11%)"
    )
    return _apply_common_layout(fig, title, subtitle, height=420, lang=lang)


def build_pedestrian_coactivity_scatter(
    ps_df: pd.DataFrame,
    thresholds: Dict[int, Tuple[float, float]],
    lang: str = "ar",
) -> go.Figure:
    """Scatter plot of the 81 observed direction-hours showing vehicles vs pedestrian crossings and P75 quadrants."""
    if ps_df is None or ps_df.empty:
        return _apply_common_layout(go.Figure(), "لا توجد بيانات للتزامن بين المركبات والمشاة", lang=lang)

    df = ps_df.copy()
    v_col = "إجمالي_مركبات_الفترة"
    p_col = "عبور_مشاة_الفترة"

    fig = go.Figure()
    for di in (1, 2):
        sub = df[df["مفتاح الاتجاه"] == di]
        if sub.empty:
            continue
        base_color = COLORS["dir1"] if di == 1 else COLORS["dir2"]
        d_label = DIRECTION_KEY_TO_SHORT_AR[di] if lang == "ar" else DIRECTION_KEY_TO_EN[di]

        # Normal hours vs Co-active priority hours
        normal = sub[~sub["تزامن_مرتفع"]] if "تزامن_مرتفع" in sub.columns else sub
        high = sub[sub["تزامن_مرتفع"]] if "تزامن_مرتفع" in sub.columns else pd.DataFrame()

        if not normal.empty:
            fig.add_trace(
                go.Scatter(
                    name=f"{d_label} (ساعات اعتيادية)",
                    x=normal[v_col],
                    y=normal[p_col],
                    mode="markers",
                    marker={"size": 9, "color": base_color, "opacity": 0.65, "symbol": "circle"},
                    customdata=normal[["تاريخ_نص", "اليوم", "فترة الرصد", "اتجاه الحصر"]].values,
                    hovertemplate=(
                        "<b>%{customdata[0]} (%{customdata[1]}) | %{customdata[2]}</b><br>"
                        "الاتجاه: %{customdata[3]}<br>"
                        "المركبات: %{x:,} مركبة/ساعة<br>"
                        "عبور المشاة: %{y:,} حدث عبور/ساعة<extra></extra>"
                    ),
                )
            )
        if not high.empty:
            fig.add_trace(
                go.Scatter(
                    name=f"{d_label} (تزامن مرتفع P75 — {len(high)} س)",
                    x=high[v_col],
                    y=high[p_col],
                    mode="markers+text",
                    marker={
                        "size": 14,
                        "color": COLORS["coactivity"] if di == 1 else "#EA580C",
                        "symbol": "diamond" if di == 1 else "star",
                        "line": {"width": 2, "color": "#0F172A"},
                    },
                    text=[f"{r['تاريخ_نص'][-2:]} أكتوبر | {r['فترة الرصد'].split()[0]}" for _, r in high.iterrows()],
                    textposition="top center",
                    customdata=high[["تاريخ_نص", "اليوم", "فترة الرصد", "اتجاه الحصر"]].values,
                    hovertemplate=(
                        "<b>★ ساعة تزامن مرتفع (أولوية تشغيلية)</b><br>"
                        "%{customdata[0]} (%{customdata[1]}) | %{customdata[2]}<br>"
                        "الاتجاه: %{customdata[3]}<br>"
                        "المركبات: %{x:,} مركبة/ساعة<br>"
                        "عبور المشاة: %{y:,} حدث عبور/ساعة<extra></extra>"
                    ),
                )
            )

    # Add P75 reference threshold lines
    if 1 in thresholds:
        v1_cut, p1_cut = thresholds[1]
        fig.add_vline(
            x=v1_cut,
            line_dash="dash",
            line_color=COLORS["dir1"],
            annotation_text=f"مئين 75 مركبات سندوب ({v1_cut:,.0f})",
            annotation_position="top left",
        )
        fig.add_hline(
            y=p1_cut,
            line_dash="dash",
            line_color=COLORS["dir1"],
            annotation_text=f"مئين 75 مشاة سندوب ({p1_cut:,.0f})",
            annotation_position="bottom right",
        )
    if 2 in thresholds:
        v2_cut, p2_cut = thresholds[2]
        fig.add_vline(
            x=v2_cut,
            line_dash="dot",
            line_color=COLORS["dir2"],
            annotation_text=f"P75 الصينية ({v2_cut:,.1f})",
            annotation_position="bottom left",
        )
        fig.add_hline(
            y=p2_cut,
            line_dash="dot",
            line_color=COLORS["dir2"],
            annotation_text=f"P75 مشاة الصينية ({p2_cut:,.0f})",
            annotation_position="top right",
        )

    title = (
        "مصفوفة التزامن بين كثافة المركبات وعبور المشاة (81 فترة رصد — 9 ساعات أولوية قصوى)"
        if lang == "ar"
        else "Vehicle Flow vs Pedestrian Crossing Co-Activity Matrix (81 Windows — 9 Priority Windows)"
    )
    subtitle = (
        "المحور الأفقي: مركبة/ساعة | المحور الرأسي: حدث عبور مشاة/ساعة (وحدتان مستقلتان لا تُجمعان)"
        if lang == "ar"
        else "X-Axis: Vehicles/Hour | Y-Axis: Pedestrian Crossings/Hour (Independent units, never summed together)"
    )
    fig.update_layout(
        xaxis_title="إجمالي المركبات في الساعة (مركبة/ساعة)" if lang == "ar" else "Vehicles per Hour",
        yaxis_title="أحداث عبور المشاة في الساعة (عبور/ساعة)" if lang == "ar" else "Pedestrian Crossings per Hour",
    )
    return _apply_common_layout(fig, title, subtitle, height=450, lang=lang)


def build_pedestrian_by_period_chart(
    period_df: pd.DataFrame,
    lang: str = "ar",
) -> go.Figure:
    """Grouped bar chart of mean pedestrian crossings per hour by time period and direction."""
    if period_df is None or period_df.empty:
        return _apply_common_layout(go.Figure(), "لا توجد بيانات لعبور المشاة", lang=lang)

    df = period_df.copy().sort_values(["ترتيب الفترة", "مفتاح الاتجاه"])
    p_mean_col = (
        "متوسط_عبور_المشاة_في_الساعة"
        if "متوسط_عبور_المشاة_في_الساعة" in df.columns
        else "متوسط_المشاة_لكل_ساعة"
    )

    fig = go.Figure()
    for di in (1, 2):
        sub = df[df["مفتاح الاتجاه"] == di]
        if sub.empty:
            continue
        color = COLORS["pedestrian"] if di == 1 else "#64748B"
        name = DIRECTION_KEY_TO_SHORT_AR[di] if lang == "ar" else DIRECTION_KEY_TO_EN[di]
        fig.add_trace(
            go.Bar(
                name=name,
                x=sub["فترة الرصد"],
                y=sub[p_mean_col],
                marker_color=color,
                text=[f"{v:,.0f}" for v in sub[p_mean_col]],
                textposition="outside",
            )
        )

    title = (
        "متوسط أحداث عبور المشاة لكل ساعة رصد حسب الفترة الزمنية والاتجاه"
        if lang == "ar"
        else "Mean Pedestrian Crossings per Observed Hour by Time Period & Direction"
    )
    subtitle = (
        "أعلى ذروة صباحية للمشاة في الساعة 9:00–10:00 صباحًا (بلغت 1,314 في 12 أكتوبر باتجاه سندوب→الصينية)"
        if lang == "ar"
        else "Morning pedestrian peak occurs at 9:00–10:00 AM (reaching 1,314 on Oct 12 in Sandoub→El-Siniya)"
    )
    fig.update_layout(
        barmode="group",
        yaxis_title="حدث عبور مشاة / ساعة" if lang == "ar" else "Pedestrian Crossings / Hour",
    )
    max_y = float(df[p_mean_col].max()) if not df.empty else 100.0
    fig.update_yaxes(range=[0, max_y * 1.22])
    return _apply_common_layout(fig, title, subtitle, height=400, lang=lang)


def build_chart_for_analysis_result(
    result: AnalysisToolResult,
    lang: str = "ar",
) -> go.Figure:
    """Dynamically construct the appropriate interactive Plotly figure for any AI Analyst tool result."""
    spec = result.chart_spec or {}
    ctype = spec.get("chart_type", "generic_bar")

    if ctype == "direction_overview":
        dir_df = spec.get("dir_comp_df")
        return build_direction_comparison_chart(dir_df, lang=lang)

    if ctype == "daily_trend":
        df = spec.get("df", result.evidence_df)
        return build_daily_trend_chart(df, metric_mode="mean", lang=lang)

    if ctype == "period_profile":
        df = spec.get("df", result.evidence_df)
        return build_period_profile_chart(df, lang=lang)

    if ctype == "peaks_and_profile":
        ranked_df = spec.get("ranked_df", result.evidence_df)
        rank_order = spec.get("rank_order", "top")
        return build_peak_hours_chart(ranked_df, rank_order=rank_order, lang=lang)

    if ctype == "category_breakdown":
        df = spec.get("df", result.evidence_df)
        return build_vehicle_category_bar_chart(df, lang=lang)

    if ctype == "direction_category_stacked":
        df = spec.get("df", result.evidence_df)
        if "حصة_سندوب_إلى_الصينية_%" in df.columns:
            return build_direction_category_stacked_chart(df, lang=lang)
        # Grouped bar by category and direction
        fig = px.bar(
            df,
            x="فئة المركبة",
            y=spec.get("metric_col", "إجمالي_المركبات"),
            color="اتجاه الحصر" if "اتجاه الحصر" in df.columns else None,
            barmode="group",
            color_discrete_map=DIRECTION_COLOR_MAP,
            text_auto=",.0f",
        )
        return _apply_common_layout(
            fig,
            spec.get("title_ar" if lang == "ar" else "title_en", "تحليل فئات المركبات"),
            lang=lang,
        )

    if ctype == "pedestrian_coactivity_scatter":
        ps_df = spec.get("ps_df")
        thresholds = spec.get("thresholds", {})
        return build_pedestrian_coactivity_scatter(ps_df, thresholds, lang=lang)

    if ctype == "coverage_matrix":
        ps_df = spec.get("ps_df")
        return build_date_period_heatmap(ps_df, direction_key=1, lang=lang)

    if ctype == "what_if_comparison":
        df = spec.get("df", result.evidence_df)
        if df is not None and not df.empty:
            fig = go.Figure()
            if "الوضع_التاريخي_المرصود" in df.columns:
                fig.add_trace(
                    go.Bar(
                        name="الوضع التاريخي المرصود" if lang == "ar" else "Observed Baseline",
                        x=df["البند"],
                        y=df["الوضع_التاريخي_المرصود"],
                        marker_color=COLORS["dir2"],
                        text=[f"{int(v):,}" for v in df["الوضع_التاريخي_المرصود"]],
                        textposition="outside",
                    )
                )
                fig.add_trace(
                    go.Bar(
                        name="المتبقي على السطح (افتراضي)" if lang == "ar" else "Hypothetical Remaining Surface",
                        x=df["البند"],
                        y=df["المتبقي_على_السطح_افتراضيًا"],
                        marker_color=COLORS["pedestrian"],
                        text=[f"{int(v):,}" for v in df["المتبقي_على_السطح_افتراضيًا"]],
                        textposition="outside",
                    )
                )
                fig.add_trace(
                    go.Bar(
                        name="المتحول للكوبري (افتراضي)" if lang == "ar" else "Hypothetical Diverted to Bridge",
                        x=df["البند"],
                        y=df["المتحول_إلى_الكوبري_افتراضيًا"],
                        marker_color=COLORS["dir1"],
                        text=[f"{int(v):,}" for v in df["المتحول_إلى_الكوبري_افتراضيًا"]],
                        textposition="outside",
                    )
                )
            elif "القيمة_التاريخية_المرصودة" in df.columns:
                fig.add_trace(
                    go.Bar(
                        name="القيمة المرصودة" if lang == "ar" else "Observed Value",
                        x=df["المؤشر"],
                        y=df["القيمة_التاريخية_المرصودة"],
                        marker_color=COLORS["dir2"],
                        text=[f"{v:,.0f}" for v in df["القيمة_التاريخية_المرصودة"]],
                        textposition="outside",
                    )
                )
                fig.add_trace(
                    go.Bar(
                        name="القيمة الافتراضية (What-If)" if lang == "ar" else "Hypothetical Value (What-If)",
                        x=df["المؤشر"],
                        y=df["القيمة_الافتراضية_بعد_التغيير"],
                        marker_color=COLORS["coactivity"],
                        text=[f"{v:,.0f}" for v in df["القيمة_الافتراضية_بعد_التغيير"]],
                        textposition="outside",
                    )
                )
            fig.update_layout(barmode="group")
            return _apply_common_layout(
                fig,
                spec.get("title_ar" if lang == "ar" else "title_en", "محاكاة السيناريو الافتراضي"),
                subtitle="تنبيه: القيم الافتراضية هي محاكاة حسابية وليست رصدًا ميدانيًا",
                lang=lang,
            )

    # Generic fallback chart from evidence_df
    df = result.evidence_df
    if df is not None and not df.empty and len(df.columns) >= 2:
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        str_cols = [c for c in df.columns if c not in num_cols]
        x_col = str_cols[0] if str_cols else df.columns[0]
        y_col = spec.get("metric_col") if spec.get("metric_col") in num_cols else (num_cols[0] if num_cols else df.columns[1])
        fig = px.bar(
            df.head(15),
            x=x_col,
            y=y_col,
            color="اتجاه الحصر" if "اتجاه الحصر" in df.columns else None,
            barmode="group",
            color_discrete_map=DIRECTION_COLOR_MAP,
            text_auto=",.1f",
        )
        return _apply_common_layout(
            fig,
            spec.get("title_ar" if lang == "ar" else "title_en", "نتيجة التحليل البياني"),
            lang=lang,
        )

    return _apply_common_layout(go.Figure(), "لا توجد بيانات بيانية", lang=lang)
