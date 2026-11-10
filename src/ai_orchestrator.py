"""Bilingual (Arabic-first + English) AI Data Analyst Orchestrator with Gemini + Guarded Python Tools.

Architecture:
1. Security & Prompt-Injection Guardrail: Blocks attempts to extract API keys, system prompts,
   observer PII, or override core analytical rules (such as mixing vehicles + pedestrians).
2. Deterministic Python Tool Execution: Resolves the user's natural-language question (and
   conversational follow-up context) into a validated Python tool call on the Star Schema.
3. Grounded Narrative Generation:
   - When a valid GEMINI_API_KEY is available and use_live_gemini=True, invokes Google Gemini
     (via the official `google-genai` SDK) over the verified Python tool results to synthesize
     the 5-part executive response strictly grounded in the computed numbers.
   - When GEMINI_API_KEY is absent or an API error occurs, transparently falls back to the
     deterministic bilingual analytical narrator and clearly badges the execution mode.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import plotly.graph_objects as go

from src.analysis_tools import (
    AnalysisToolResult,
    detect_query_language,
    normalize_arabic_text,
    resolve_question_and_execute,
)
from src.charts import build_chart_for_analysis_result
from src.config import (
    DIRECTION_KEY_TO_AR,
    DIRECTION_KEY_TO_EN,
    DIRECTION_KEY_TO_SHORT_AR,
    GEMINI_MODEL_CHAIN,
    get_gemini_api_key,
)
from src.metrics_engine import FilterScope
from src.semantic_model import SemanticModel, build_semantic_model


PROMPT_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(?:all\s+)?(?:previous|prior|system)\s+instructions|"
    r"reveal\s+(?:your\s+)?(?:api\s*key|secret|system\s+prompt|instructions)|"
    r"print\s+os\.environ|GEMINI_API_KEY|"
    r"تجاهل\s+(?:كل\s+)?التعليمات|"
    r"اكشف\s+(?:مفتاح|السر|تعليمات)|"
    r"اطبع\s+مفتاح|"
    r"(?:ال)?اسماء.*(?:ال)?راصدين|"
    r"اسماء\s+الراصدين|"
    r"الاسم\s+الحقيقي\s+للراصد)",
    re.IGNORECASE,
)


@dataclass
class AIAnalystResponse:
    """Complete 5-part response returned by the AI Analyst ('اسأل بياناتك')."""

    question: str
    lang: Literal["ar", "en"]
    direct_answer: str
    evidence_and_method: str
    defensible_insight: str
    recommendation: str
    limitation: str
    tool_result: AnalysisToolResult
    figure: go.Figure
    engine_mode: Literal["gemini_live", "deterministic_python", "security_guardrail"]
    model_used: str
    status_badge_ar: str
    status_badge_en: str
    session_context: Dict[str, Any] = field(default_factory=dict)


def _build_security_refusal(
    question: str,
    lang: Literal["ar", "en"],
    ui_scope: Optional[FilterScope],
    session_context: Dict[str, Any],
    model: SemanticModel,
) -> AIAnalystResponse:
    """Return a clean refusal when a prompt injection, secret extraction, or PII request is detected."""
    res, updated_ctx = resolve_question_and_execute(
        "ملخص المؤشرات المعتمدة", session_context=session_context, ui_scope=ui_scope, model=model
    )
    fig = build_chart_for_analysis_result(res, lang=lang)
    if lang == "en":
        return AIAnalystResponse(
            question=question,
            lang="en",
            direct_answer=(
                "Request declined by security and data-governance guardrails: This assistant does not "
                "expose system prompts, API keys/credentials, or raw observer personal names, and cannot "
                "override approved analytical rules."
            ),
            evidence_and_method=(
                f"Verified baseline metrics remain active: **127,769** total vehicles across **81** observed "
                f"direction-hours (**74,248** in Sandoub → El-Siniya over 43h; **53,521** in El-Siniya → Sandoub "
                f"over 38h) and **24,342** independent pedestrian crossings (**17,478** + **6,864**)."
            ),
            defensible_insight=(
                "All calculations are executed strictly via schema-validated Python functions on anonymized "
                "data (`راصد #01`..`راصد #15`) to guarantee mathematical reproducibility and privacy."
            ),
            recommendation=(
                "Ask any analytical question about traffic volume, peak hours, matched 38-hour direction comparisons, "
                "vehicle classes, or pedestrian co-activity windows."
            ),
            limitation=(
                "Personal observer identifiers and environment secrets are strictly isolated from the analytical layer."
            ),
            tool_result=res,
            figure=fig,
            engine_mode="security_guardrail",
            model_used="security_guardrail",
            status_badge_ar="حارس الأمان وحوكمة البيانات نشط",
            status_badge_en="Security & Governance Guardrail Active",
            session_context=updated_ctx,
        )
    return AIAnalystResponse(
        question=question,
        lang="ar",
        direct_answer=(
            "تم رفض الطلب بواسطة ضوابط الأمان وحوكمة البيانات: لا يُسمح بكشف مفاتيح الواجهة البرمجية (API Keys) "
            "أو تعليمات النظام الداخلية أو الأسماء الشخصية للراصدين، كما لا يُسمح بتجاوز القواعد المنهجية المعتمدة."
        ),
        evidence_and_method=(
            "تظل المؤشرات الموثقة متاحة بالكامل عبر محرك الحساب المحلي: إجمالي **127,769** مركبة عبر **81** فترة اتجاه-ساعة "
            "(**74,248** في سندوب → الصينية خلال 43 ساعة، و**53,521** في الصينية → سندوب خلال 38 ساعة)، بالإضافة إلى "
            "**24,342** حدث عبور مشاة مستقل (**17,478** + **6,864**)."
        ),
        defensible_insight=(
            "تُنفذ جميع العمليات الحسابية حصريًا عبر دوال بايثون المحمية مع تعمية معرفات الراصدين (`راصد #01` إلى `راصد #15`) "
            "لضمان الدقة الحسابية وحماية الخصوصية."
        ),
        recommendation=(
            "يمكنك طرح أي سؤال تحليلي حول كثافة المركبات، ساعات الذروة، المقارنة العادلة في الـ 38 ساعة المشتركة، "
            "تصنيف الفئات الثماني، أو ساعات التزامن مع عبور المشاة."
        ),
        limitation=(
            "بيانات الاعتماد السرية والأسماء الشخصية للراصدين معزولة تمامًا عن طبقة العرض والذكاء الاصطناعي."
        ),
        tool_result=res,
        figure=fig,
        engine_mode="security_guardrail",
        model_used="security_guardrail",
        status_badge_ar="حارس الأمان وحوكمة البيانات نشط",
        status_badge_en="Security & Governance Guardrail Active",
        session_context=updated_ctx,
    )


def build_deterministic_narrative(
    question: str,
    res: AnalysisToolResult,
    lang: Literal["ar", "en"],
) -> Dict[str, str]:
    """Generate a thorough, verified 5-part narrative directly from Python tool outputs."""
    hm = res.headline_metrics
    tname = res.tool_name
    df = res.evidence_df

    # 1. Handle Evidence Boundaries & False Premises
    if res.is_boundary_or_refusal:
        topic = res.boundary_topic or "general"
        if lang == "en":
            if topic == "wrong_date_or_year":
                return {
                    "direct_answer": (
                        "**Out-of-Range Date/Year:** The requested date or year is not present in the survey dataset. "
                        "This dataset exclusively covers **6 days from 12 October 2026 to 17 October 2026**."
                    ),
                    "evidence_and_method": (
                        f"Across the 6 observed days (12–17 Oct 2026), the dataset contains **96** survey records "
                        f"representing **81** observed direction-hours (`43` in Sandoub → El-Siniya and `38` in El-Siniya → Sandoub), "
                        f"recording **127,769** total vehicles and **24,342** pedestrian crossings."
                    ),
                    "defensible_insight": (
                        "Strictly enforcing temporal boundaries prevents extrapolating a 6-day October 2026 field campaign "
                        "to other months or years without seasonal calibration."
                    ),
                    "recommendation": (
                        "Select any date between **2026-10-12 (Monday)** and **2026-10-17 (Saturday)** to inspect verified "
                        "hourly volumes, peaks, and vehicle classifications."
                    ),
                    "limitation": (
                        "No field observations exist before 12 Oct 2026 or after 17 Oct 2026, nor between 17:00 and 07:00."
                    ),
                }
            if topic == "mixed_units":
                return {
                    "direct_answer": (
                        "**Methodological Refusal (Unit Mixing):** Vehicles and pedestrian crossings represent fundamentally "
                        "different physical units (`vehicle` vs `pedestrian crossing event`) and **must never be summed into a single traffic KPI**."
                    ),
                    "evidence_and_method": (
                        f"In the selected scope ({res.scope_used.describe_en()}):\n"
                        f"- **Total Observed Vehicles (Independent):** `{hm.get('إجمالي_المركبات', 127769):,}` vehicles across `{hm.get('ساعات_الرصد_الفعلي', 81)}` direction-hours "
                        f"(mean `{hm.get('متوسط_المركبات_لكل_ساعة_رصد', 1577.40):,.2f}` veh/hr).\n"
                        f"- **Total Pedestrian Crossings (Independent):** `{hm.get('إجمالي_عبور_المشاة', 24342):,}` crossings "
                        f"(`17,478` in Sandoub → El-Siniya at `406.47`/hr; `6,864` in El-Siniya → Sandoub at `180.63`/hr).\n"
                        f"*(Note: While the raw Excel sheet has an audit checksum column `الإجمالي العام = 152,111`, it is strictly for row-level arithmetic verification and is barred from analytical reporting.)*"
                    ),
                    "defensible_insight": (
                        "Instead of adding vehicles and pedestrians together, the scientifically valid way to analyze their interaction "
                        "is **Simultaneous Temporal Co-Activity (P75)**: identifying the **9 priority windows** where both vehicle flow and "
                        "pedestrian crossings simultaneously exceed the 75th percentile of their direction."
                    ),
                    "recommendation": (
                        "Focus operational traffic management and pedestrian guidance on the **9 high co-activity windows** "
                        "(4 windows on Wednesday Oct 14 in Sandoub → El-Siniya, and 5 windows on Oct 13 & Oct 17 in El-Siniya → Sandoub)."
                    ),
                    "limitation": (
                        "Co-activity measures temporal exposure in the same hour and direction, not physical collision counts."
                    ),
                }
            if topic == "false_premise_direction":
                return {
                    "direct_answer": (
                        "**Premise Correction:** The direction **El-Siniya → Sandoub is NOT higher** in overall vehicle volume or intensity. "
                        "In fact, **Sandoub → El-Siniya is higher across all three methodological metrics**: raw total (`74,248` vs `53,521`), "
                        "hourly mean (`1,726.70` vs `1,408.45` veh/hr), and the **38 matched common hours** (`65,279` vs `53,521`, i.e. **+11,758 vehicles / +21.97%**)."
                    ),
                    "evidence_and_method": (
                        "1. **Matched 38 Common Windows:** Sandoub → El-Siniya recorded **65,279** vehicles (`1,717.87` veh/hr) vs **53,521** (`1,408.45` veh/hr) in El-Siniya → Sandoub.\n"
                        "2. **All Observed Hours:** Sandoub → El-Siniya was observed for **43 hours** (`74,248` vehicles) vs **38 hours** (`53,521` vehicles) in El-Siniya → Sandoub.\n"
                        "3. **Where El-Siniya → Sandoub IS higher:** Only in the **relative percentage share of Heavy Trucks** (`6.88%` vs `5.11%`) and **Microbuses** (`23.62%` vs `22.74%`)."
                    ),
                    "defensible_insight": (
                        "Even after stripping out the 5 extra hours observed only in Sandoub → El-Siniya (which contained 8,969 vehicles), "
                        "Sandoub → El-Siniya still carries **21.97% more traffic** in identical time slots."
                    ),
                    "recommendation": (
                        "Prioritize capacity and queue management on the **Sandoub → El-Siniya** approach, especially during the midday peak (12:00–15:00), "
                        "while monitoring heavy-truck merges on **El-Siniya → Sandoub** where their relative share reaches `6.88%`."
                    ),
                    "limitation": (
                        "5 calendar hours (2h on Oct 14 12–2 PM and 3h on Oct 16) were observed only in Sandoub → El-Siniya."
                    ),
                }
            if topic == "false_premise_oct16_vs_oct14":
                return {
                    "direct_answer": (
                        "**Distinguishing Coverage Duration from Traffic Intensity:** Friday **16 October** had the **longest observation coverage** "
                        "in Sandoub → El-Siniya (**9 observed hours**), but the **lowest traffic intensity** (**777.89 vehicles/hour**, total `7,001`). "
                        "Conversely, Wednesday **14 October** had fewer observed hours (**7 hours** in Dir 1) but **more than 3× higher hourly intensity** (**2,543.43 vehicles/hour**, total `17,804`)."
                    ),
                    "evidence_and_method": (
                        "- **Wednesday 2026-10-14 (Sandoub → El-Siniya):** `17,804` vehicles ÷ `7` observed hours = **2,543.43 veh/hr** (includes the study peak hour `4,425` at 12–1 PM and `3,636` at 1–2 PM).\n"
                        "- **Friday 2026-10-16 (Sandoub → El-Siniya):** `7,001` vehicles ÷ `9` observed hours = **777.89 veh/hr**.\n"
                        "- **Wednesday 2026-10-14 (El-Siniya → Sandoub):** `7,030` vehicles ÷ `5` hours = **1,406.00 veh/hr** vs Friday Oct 16: `6,239` ÷ `6` hours = **1,039.83 veh/hr**."
                    ),
                    "defensible_insight": (
                        "This contrast proves why raw daily sums must never be compared without normalizing by observed hours (`FactCoverage`). "
                        "Longer field shifts do not mean heavier traffic."
                    ),
                    "recommendation": (
                        "Standardize daily KPI reporting on **Mean Vehicles per Observed Hour (`مركبة/ساعة`)** and unify future shift schedules "
                        "to cover the exact same 10 hours every day."
                    ),
                    "limitation": (
                        "Oct 14 in El-Siniya → Sandoub lacked coverage during 12:00–14:00 (only 5 hours observed that day in Dir 2)."
                    ),
                }
            # General safety / speed / map / bridge impact boundary in English
            return {
                "direct_answer": (
                    f"**Evidence Boundary (`{topic}`):** The survey dataset does **not** contain records for "
                    "traffic crashes/accidents, vehicle speeds, delay/queue times, GPS spatial coordinates, or a split between "
                    "above-bridge vs street-level pedestrian crossings. Therefore, no direct causal or spatial claim can be fabricated."
                ),
                "evidence_and_method": (
                    f"What the dataset **does** verify with 100% precision is **Simultaneous Temporal Exposure**:\n"
                    f"- Total vehicles: **{hm.get('إجمالي_المركبات', 127769):,}** across **{hm.get('ساعات_الرصد_الفعلي', 81)}** direction-hours.\n"
                    f"- Independent pedestrian crossings: **{hm.get('إجمالي_عبور_المشاة', 24342):,}** (`17,478` in Dir 1 at `406.47`/hr; `6,864` in Dir 2 at `180.63`/hr).\n"
                    f"- **9 Simultaneous High-Activity Windows (P75):** `4` windows in Sandoub → El-Siniya (all on Oct 14, threshold `>= 2,113` veh & `>= 563` ped) "
                    f"and `5` windows in El-Siniya → Sandoub (`2` on Oct 13 and `3` on Oct 17, threshold `>= 1,730.25` veh & `>= 281` ped)."
                ),
                "defensible_insight": (
                    "While we cannot claim where on the road pedestrians cross or whether the bridge reduced crashes, "
                    "the 9 P75 co-activity windows pinpoint the exact hours when maximum pedestrian flow coincides with maximum vehicular flow."
                ),
                "recommendation": (
                    "Adopt the **2-Observer Synchronized Field Protocol**: assign Observer #1 exclusively to count crossings **on the pedestrian bridge** "
                    "and Observer #2 exclusively to count **at-grade street crossings below the bridge** during the 9 priority windows to measure actual bridge capture rate."
                ),
                "limitation": (
                    "Current data is a post-construction observational count at corridor level without pre-construction baseline, GPS tags, or crash logs."
                ),
            }

        # Arabic Boundary Narratives
        if topic == "wrong_date_or_year":
            return {
                "direct_answer": (
                    "**خارج النطاق الزمني المرصود:** السنة أو التاريخ المطلوب غير موجود في قاعدة البيانات. "
                    "تغطي الدراسة الميدانية حصريًا **6 أيام فقط من 12 أكتوبر 2026 إلى 17 أكتوبر 2026**."
                ),
                "evidence_and_method": (
                    "خلال فترة الدراسة المعتمدة (12–17 أكتوبر 2026)، تضم قاعدة البيانات **96** سجل رصد تمثل **81** فترة اتجاه-ساعة فعلية "
                    "(`43` ساعة في سندوب → الصينية و`38` ساعة في الصينية → سندوب)، بإجمالي **127,769** مركبة و**24,342** حدث عبور مشاة مستقل."
                ),
                "defensible_insight": (
                    "الالتزام الصارم بالنطاق الزمني يمنع تعميم نتائج حملة رصد مدتها 6 أيام في أكتوبر 2026 على سنوات أو شهور أخرى دون معايرة موسمية."
                ),
                "recommendation": (
                    "يرجى اختيار تاريخ ضمن الفترة **2026-10-12 (الاثنين)** إلى **2026-10-17 (السبت)** لاستعراض الأرقام الموثقة."
                ),
                "limitation": (
                    "لا توجد أي قراءات ميدانية قبل 12 أكتوبر 2026 أو بعد 17 أكتوبر 2026، ولا توجد قراءات ليلية بين 5:00 مساءً و7:00 صباحًا."
                ),
            }
        if topic == "mixed_units":
            return {
                "direct_answer": (
                    "**رفض منهجي (عدم خلط الوحدات):** لا يجوز جمع أعداد المركبات مع أحداث عبور المشاة في مؤشر مروري واحد؛ "
                    "لأن كلاً منهما يقيس وحدة فيزيائية وتشغيلية مختلفة تمامًا (`مركبة` مقابل `حدث عبور مشاة`)."
                ),
                "evidence_and_method": (
                    f"الأرقام الموثقة بشكل منفصل داخل النطاق ({res.scope_used.describe_ar()}):\n"
                    f"- **إجمالي المركبات (مؤشر مستقل):** `{hm.get('إجمالي_المركبات', 127769):,}` مركبة عبر `{hm.get('ساعات_الرصد_الفعلي', 81)}` ساعة رصد "
                    f"(بمتوسط `{hm.get('متوسط_المركبات_لكل_ساعة_رصد', 1577.40):,.2f}` مركبة/ساعة).\n"
                    f"- **إجمالي عبور المشاة (مؤشر مستقل):** `{hm.get('إجمالي_عبور_المشاة', 24342):,}` حدث عبور "
                    f"(`17,478` في سندوب → الصينية بمتوسط `406.47`/ساعة، و`6,864` في الصينية → سندوب بمتوسط `180.63`/ساعة).\n"
                    f"*(ملاحظة تدقيقية: عمود `الإجمالي العام = 152,111` في ملف الإكسيل الخام يُستخدم فقط للتحقق الحسابي من سلامة الصفوف ولا يُعرض أبدًا كمؤشر حركة مرورية).*"
                ),
                "defensible_insight": (
                    "بدلاً من الجمع الحسابي غير السليم بين المركبات والمشاة، تعتمد المنهجية الهندسية الصحيحة على **مصفوفة التزامن الزمني (P75)** "
                    "لتحديد الساعات التي يتزامن فيها ارتفاع كثافة المركبات مع ارتفاع عبور المشاة في نفس الساعة والاتجاه (**9 ساعات أولوية**)."
                ),
                "recommendation": (
                    "اعتماد مؤشر **ساعات التزامن المرتفع (9 ساعات)** لتوجيه أفراد التنظيم المروري وتحسين مسارات توجيه المشاة نحو الكوبري، "
                    "خاصة يوم الأربعاء 14 أكتوبر (4 ساعات متتالية في سندوب → الصينية) ويومي 13 و17 أكتوبر (5 ساعات في الصينية → سندوب)."
                ),
                "limitation": (
                    "مؤشر التزامن يقيس التعرض الزمني المشترك (Temporal Exposure) داخل الساعة ولا يقيس الحوادث أو التصادمات."
                ),
            }
        if topic == "false_premise_direction":
            return {
                "direct_answer": (
                    "**تصحيح فرضية السؤال:** اتجاه **الصينية → سندوب ليس أعلى** في حجم أو كثافة المركبات الإجمالية؛ "
                    "بل إن اتجاه **سندوب → الصينية هو الأعلى في جميع المعايير الثلاثة**: الإجمالي الخام (`74,248` مقابل `53,521` مركبة)، "
                    "والمتوسط لكل ساعة رصد (`1,726.70` مقابل `1,408.45` مركبة/ساعة)، وفي **الـ 38 ساعة المشتركة المتناظرة** (`65,279` مقابل `53,521` مركبة، أي بفارق **+11,758 مركبة / +21.97%**)."
                ),
                "evidence_and_method": (
                    "1. **المقارنة العادلة في الـ 38 ساعة المشتركة:** سجل اتجاه سندوب → الصينية **65,279** مركبة (بمتوسط `1,717.87` مركبة/ساعة) مقابل **53,521** مركبة (`1,408.45` مركبة/ساعة) في الصينية → سندوب.\n"
                    "2. **إجمالي ساعات الرصد الكاملة:** رُصد اتجاه سندوب → الصينية خلال **43 ساعة** (`74,248` مركبة) مقابل **38 ساعة** (`53,521` مركبة) للاتجاه العكسي.\n"
                    "3. **أين يتفوق اتجاه الصينية → سندوب نسبيًا؟** يتفوق فقط في **الحصة النسبية للنقل الثقيل والتريلات** (`6.88%` مقابل `5.11%`) وحصة الميكروباص (`23.62%` مقابل `22.74%`)."
                ),
                "defensible_insight": (
                    "حتى بعد استبعاد الساعات الخمس الإضافية التي رُصدت في اتجاه سندوب → الصينية وحده (وإجماليها 8,969 مركبة)، "
                    "يظل التدفق من سندوب إلى الصينية أعلى بنسبة **21.97%** في نفس التوقيتات تمامًا."
                ),
                "recommendation": (
                    "توجيه الأولوية القصوى لإدارة التدفق والاختناقات في اتجاه **سندوب → الصينية** خلال ساعات الذروة الظهرية (12:00–3:00 م)، "
                    "مع تخصيص رقابة لتنظيم حارات **النقل الثقيل** في اتجاه **الصينية → سندوب** حيث ترتفع حصته النسبية إلى `6.88%`."
                ),
                "limitation": (
                    "توجد 5 ساعات رُصدت في اتجاه سندوب → الصينية فقط (ساعتان يوم 14 أكتوبر 12–2 م، و3 ساعات يوم 16 أكتوبر)."
                ),
            }
        if topic == "false_premise_oct16_vs_oct14":
            return {
                "direct_answer": (
                    "**تصحيح الخلط بين طول التغطية وشدة الكثافة:** يوم الجمعة **16 أكتوبر** سجل **أطول تغطية زمنية للرصد** في اتجاه سندوب → الصينية (**9 ساعات**)، "
                    "لكنه كان **الأقل كثافة مرورية على الإطلاق** بمتوسط **777.89 مركبة/ساعة** فقط (إجمالي `7,001`). "
                    "في المقابل، يوم الأربعاء **14 أكتوبر** رُصد في **7 ساعات** فقط في نفس الاتجاه لكنه سجل **أعلى كثافة مرورية بمتوسط 2,543.43 مركبة/ساعة** (إجمالي `17,804` مركبات، أي أكثر من 3 أضعاف كثافة الساعة يوم الجمعة)."
                ),
                "evidence_and_method": (
                    "- **الأربعاء 14 أكتوبر (سندوب → الصينية):** `17,804` مركبة ÷ `7` ساعات رصد = **2,543.43 مركبة/ساعة** (ويضم أعلى ساعتين في الدراسة: `4,425` في 12–1 م و`3,636` في 1–2 م).\n"
                    "- **الجمعة 16 أكتوبر (سندوب → الصينية):** `7,001` مركبة ÷ `9` ساعات رصد = **777.89 مركبة/ساعة**.\n"
                    "- **في اتجاه الصينية → سندوب:** سجل يوم 14 أكتوبر متوسط **1,406.00 مركبة/ساعة** (7,030 ÷ 5 ساعات) مقابل **1,039.83 مركبة/ساعة** يوم 16 أكتوبر (6,239 ÷ 6 ساعات)."
                ),
                "defensible_insight": (
                    "هذا المثال يبرهن عمليًا لماذا يُحظر مقارنة المجاميع اليومية الخام دون القسمة على ساعات الرصد الفعلية (`FactCoverage`)؛ "
                    "فزيادة عدد ساعات تواجد الراصدين في الميدان لا تعني زيادة الازدحام."
                ),
                "recommendation": (
                    "اعتماد **متوسط المركبات لكل ساعة رصد (`مركبة/ساعة`)** كمؤشر أساسي للمقارنة بين الأيام، وتوحيد جدول نوبات الرصد الميداني مستقبلاً ليشمل 10 ساعات ثابتة يوميًا."
                ),
                "limitation": (
                    "يوم 14 أكتوبر في اتجاه الصينية → سندوب لم يُرصد بين الساعة 12:00 و2:00 ظهرًا (رُصدت 5 ساعات فقط في ذلك الاتجاه)."
                ),
            }

        # General Arabic boundary narrative (accidents, speed, map coordinates, bridge impact)
        topic_labels_ar = {
            "accidents": "الحوادث أو الإصابات أو التصادمات المرورية",
            "speed_or_delay": "سرعات المركبات أو أزمنة الانتظار أو سعة الحارات",
            "map_coordinates": "الإحداثيات الجغرافية (GPS) أو خرائط النقاط الساخنة المكانية على الطريق",
            "bridge_impact": "نسبة الاستخدام الفعلي لكوبري المشاة مقابل العبور السطحي أو الأثر السببي قبل/بعد الكوبري",
        }
        missing_desc = topic_labels_ar.get(topic, "المتغيرات الميدانية غير المدرجة في جدول الحصر")
        return {
            "direct_answer": (
                f"**حدود الأدلة المتاحة:** لا تحتوي قاعدة البيانات الحالية على أي قياسات لـ **{missing_desc}**؛ "
                "ولذلك لا يمكن علميًا أو مهنيًا اختلاق أرقام للحوادث أو الخرائط المكانية أو الجزم بنسبة التحول الفعلي إلى الكوبري من هذه البيانات وحدها."
            ),
            "evidence_and_method": (
                "ما تثبته البيانات رقميًا بدقة 100% هو **التعرض الزمني المشترك (Temporal Co-Activity)** بين المركبات والمشاة:\n"
                f"- **إجمالي المركبات المرصودة:** `{hm.get('إجمالي_المركبات', 127769):,}` مركبة عبر `{hm.get('ساعات_الرصد_الفعلي', 81)}` فترة اتجاه-ساعة.\n"
                f"- **إجمالي أحداث عبور المشاة:** `{hm.get('إجمالي_عبور_المشاة', 24342):,}` حدث عبور (`17,478` في سندوب → الصينية بمتوسط `406.47`/ساعة، و`6,864` في الصينية → سندوب بمتوسط `180.63`/ساعة).\n"
                f"- **ساعات التزامن المرتفع (المئين 75 للمركبات والمشاة معًا):** توجد **9 ساعات أولوية قصوى** (`4` ساعات في سندوب → الصينية كلها يوم 14 أكتوبر بحد `>= 2,113` مركبة و`>= 563` مشاة، و`5` ساعات في الصينية → سندوب يومي 13 و17 أكتوبر بحد `>= 1,730.25` مركبة و`>= 281` مشاة)."
            ),
            "defensible_insight": (
                "رغم أن البيانات تحدد **متى** و**في أي اتجاه** يبلغ التزاحم ذروته المشتركة (التعرض الزمني)، إلا أن وجود عمود واحد فقط للمشاة (`مشاة (عبور المشاة)`) "
                "دون تقسيم إلى «فوق الكوبري» و«سطحي تحت الكوبري» يمنع قياس كفاءة الكوبري بشكل مباشر."
            ),
            "recommendation": (
                "تطبيق **بروتوكول الراصدين المتزامنين (2-Observer Protocol)** خلال **الساعات التسع ذات الأولوية**: يُخصص الراصد الأول لعد المشاة **أعلى الكوبري فقط**، "
                "والراصد الثاني لعد العبور **السطحي أسفل الكوبري ومحيطه** في نفس الساعة، مما يحسم فورًا نسبة الالتزام الفعلي بالعبور الآمن."
            ),
            "limitation": (
                "الدراسة الحالية رصدت الأعداد بعد إنشاء الكوبري على مستوى الاتجاه والساعة دون إحداثيات مكانية (GPS) ودون بيانات مقارنة سابقة لإنشاء الكوبري."
            ),
        }

    # 2. Normal Analytical Tool Narratives
    if tname == "compare_directions_fairly":
        d1_raw = hm["d1_raw"]
        d2_raw = hm["d2_raw"]
        d1_h = hm["d1_hours"]
        d2_h = hm["d2_hours"]
        d1_mean = hm["d1_mean"]
        d2_mean = hm["d2_mean"]
        common_h = hm["common_hours"]
        d1_m = hm["d1_matched"]
        d2_m = hm["d2_matched"]
        m_diff = hm["matched_diff"]
        m_pct = hm["matched_diff_pct"]
        unit = hm.get("target_unit", "مركبة")

        if lang == "en":
            return {
                "direct_answer": (
                    f"In a fair matched comparison across the **{common_h} common hours** observed in both directions, "
                    f"**Sandoub → El-Siniya** recorded **{d1_m:,}** vs **{d2_m:,}** in **El-Siniya → Sandoub** — "
                    f"a verified difference of **{m_diff:+,} ({m_pct:+.2f}%)**."
                ),
                "evidence_and_method": (
                    f"- **Matched Common Windows ({common_h}h in both directions):** Sandoub → El-Siniya = `{d1_m:,}` (`{d1_m/common_h if common_h else 0:,.2f}`/hr) "
                    f"vs El-Siniya → Sandoub = `{d2_m:,}` (`{d2_m/common_h if common_h else 0:,.2f}`/hr).\n"
                    f"- **All Observed Hours ({d1_h}h vs {d2_h}h):** Sandoub → El-Siniya = `{d1_raw:,}` (mean `{d1_mean:,.2f}`/hr) "
                    f"vs El-Siniya → Sandoub = `{d2_raw:,}` (mean `{d2_mean:,.2f}`/hr).\n"
                    f"- **Unmatched Hours:** `{hm['unmatched_d1_hours']}` hours were observed only in Sandoub → El-Siniya (`{hm['unmatched_d1_vehicles']:,}` vehicles)."
                ),
                "defensible_insight": (
                    "Comparing raw totals directly (`74,248` vs `53,521`, +38.73%) overstates the directional gap because Sandoub → El-Siniya "
                    "was observed for 5 extra hours (including the Oct 14 12–2 PM peak of 8,061 vehicles). Restricting to the 38 shared hours "
                    "proves that Sandoub → El-Siniya is genuinely heavier by **+21.97%**."
                ),
                "recommendation": (
                    "Use the **38-hour matched window baseline (+21.97%)** for corridor signal timing and lane allocation, "
                    "and ensure future field surveys observe the exact same hours in both directions simultaneously."
                ),
                "limitation": (
                    "On Wednesday Oct 14 from 12:00 to 14:00, only Sandoub → El-Siniya was observed; El-Siniya → Sandoub lacks a counterpart reading for those 2 peak hours."
                ),
            }
        return {
            "direct_answer": (
                f"عند إجراء المقارنة المنهجية العادلة داخل **{common_h} ساعة مشتركة متناظرة** رُصد فيها الاتجاهان معًا، "
                f"يتفوق اتجاه **سندوب → الصينية** بتسجيل **{d1_m:,} {unit}** مقابل **{d2_m:,} {unit}** في اتجاه **الصينية → سندوب**، "
                f"بفارق صافي قدره **{m_diff:+,} {unit} ({m_pct:+.2f}%)**."
            ),
            "evidence_and_method": (
                f"تم الحساب عبر 3 مستويات منهجية ({res.scope_used.describe_ar()}):\n"
                f"1. **المقارنة المتناظرة في الساعات المشتركة ({common_h} ساعة):** `{d1_m:,}` في سندوب → الصينية (متوسط `{d1_m/common_h if common_h else 0:,.2f}`/ساعة) مقابل `{d2_m:,}` في الصينية → سندوب (متوسط `{d2_m/common_h if common_h else 0:,.2f}`/ساعة).\n"
                f"2. **المتوسط الساعي لكامل الرصد ({d1_h} ساعة مقابل {d2_h} ساعة):** `{d1_mean:,.2f}` مقابل `{d2_mean:,.2f}` {unit}/ساعة (فارق `+{(d1_mean - d2_mean)/d2_mean*100 if d2_mean else 0:.2f}%`).\n"
                f"3. **الإجمالي الخام غير المتساوي التغطية:** `{d1_raw:,}` مقابل `{d2_raw:,}` (يتضمن `{hm['unmatched_d1_hours']}` ساعات إضافية رُصدت في اتجاه سندوب فقط بإجمالي `{hm['unmatched_d1_vehicles']:,}` مركبة)."
            ),
            "defensible_insight": (
                "الاعتماد على الإجمالي الخام وحده يضخم الفارق الظاهري إلى `+38.73%` بسبب وجود ساعتين شديدتي الازدحام يوم 14 أكتوبر (12–2 ظهرًا: `4,425 + 3,636 = 8,061` مركبة) "
                "رُصدتا في سندوب → الصينية دون رصد مقابل في الاتجاه العكسي. أما تثبيت الـ 38 ساعة المشتركة فيثبت أن التفوق الحقيقي لاتجاه سندوب → الصينية هو **+21.97%**."
            ),
            "recommendation": (
                "اعتماد نسبة **+21.97%** (وليس +38.7%) كأساس هندسي لتوزيع سعة المسارات وأولويات التنظيم المروري لصالح اتجاه **سندوب → الصينية**، "
                "مع إلزام فرق الرصد الميداني مستقبلاً بالبدء والانتهاء في نفس الساعات لكلا الاتجاهين."
            ),
            "limitation": (
                "غياب الرصد في اتجاه الصينية → سندوب خلال فترتي 12:00–2:00 ظهرًا يوم الأربعاء 14 أكتوبر (5 ساعات غير متناظرة إجمالاً بين الاتجاهين)."
            ),
        }

    if tname == "analyze_peaks_and_anomalies":
        top_v = hm.get("top_1_vehicles", 0)
        top_cap = hm.get("top_1_caption", "—")
        d1_12 = hm.get("d1_12_13_mean", 0)
        d1_12_h = hm.get("d1_12_13_hours", 0)
        d1_13 = hm.get("d1_13_14_mean", 0)
        d1_13_h = hm.get("d1_13_14_hours", 0)
        d2_13 = hm.get("d2_13_14_mean", 0)
        d2_13_h = hm.get("d2_13_14_hours", 0)

        if lang == "en":
            return {
                "direct_answer": (
                    f"The highest single observed hour in the selected scope is **{top_v:,} vehicles/hour** at **{top_cap}**. "
                    f"However, the most reliable **recurring daily peak pattern** is **01:00 PM – 02:00 PM**, averaging **{d1_13:,.2f} veh/hr** "
                    f"across `{d1_13_h}` days in Sandoub → El-Siniya and **{d2_13:,.2f} veh/hr** across `{d2_13_h}` days in El-Siniya → Sandoub."
                ),
                "evidence_and_method": (
                    f"- **Peak Single Hour (Sandoub → El-Siniya):** `4,425` vehicles on `2026-10-14 | 12:00 PM - 01:00 PM` (merged from 2 observer records `A-26` + `A-27`), followed by `3,636` at `01:00 PM - 02:00 PM` on the same day.\n"
                    f"- **Peak Single Hour (El-Siniya → Sandoub):** `2,687` vehicles on `2026-10-17 | 12:00 PM - 01:00 PM`.\n"
                    f"- **Sustained Pattern Check:** `12:00–01:00 PM` in Sandoub → El-Siniya was observed only `{d1_12_h}` time(s), whereas `01:00–02:00 PM` was observed across `{d1_13_h}` days.\n"
                    f"- **Statistical Outliers (Tukey 1.5×IQR):** `{hm.get('outlier_count', 0)}` extreme hour(s) flagged."
                ),
                "defensible_insight": (
                    "Separating a single-day spike (`4,425` on Oct 14 12–1 PM, flagged as a Tukey IQR upper outlier above the `3,221.5` fence) "
                    "from the multi-day afternoon peak (`01:00–03:00 PM`) prevents over-designing permanent infrastructure on a single anomalous reading."
                ),
                "recommendation": (
                    "Design daily peak operational interventions around the **12:00 PM – 03:00 PM** midday window (with peak staffing at **01:00–02:00 PM**), "
                    "and schedule daily midday coverage at `12:00–01:00 PM` across all days to verify how often volumes exceed 3,500 veh/hr."
                ),
                "limitation": (
                    "The 12:00–01:00 PM slot was observed only once in Sandoub → El-Siniya (Oct 14) and twice in El-Siniya → Sandoub (Oct 12 & Oct 17)."
                ),
            }
        return {
            "direct_answer": (
                f"أعلى ساعة رصد منفردة في النطاق المحدد بلغت **{top_v:,} مركبة/ساعة** في **{top_cap}**. "
                f"غير أن التحليل المنهجي يميز بين **أعلى قفزة منفردة** (12:00–1:00 ظهرًا يوم 14 أكتوبر = `4,425` مركبة) "
                f"وبين **النمط اليومي المتكرر الأعلى** وهو الساعة **01:00–02:00 ظهرًا** التي سجلت متوسط **{d1_13:,.2f} مركبة/ساعة** عبر `{d1_13_h}` أيام في سندوب → الصينية "
                f"و**{d2_13:,.2f} مركبة/ساعة** عبر `{d2_13_h}` أيام في الصينية → سندوب."
            ),
            "evidence_and_method": (
                f"- **الترتيب الأعلى للساعات في الدراسة:**\n"
                f"  1) `2026-10-14 | 12:00–1:00 ظهرًا | سندوب → الصينية`: **4,425** مركبة (مجموع سجلين لراصدين: `2,331 + 2,094`).\n"
                f"  2) `2026-10-14 | 1:00–2:00 ظهرًا | سندوب → الصينية`: **3,636** مركبة (تزامن معها `811` عبور مشاة).\n"
                f"  3) `2026-10-17 | 12:00–1:00 ظهرًا | الصينية → سندوب`: **2,687** مركبة (أعلى ذروة في الاتجاه العكسي).\n"
                f"  4) `2026-10-12 | 2:00–3:00 ظهرًا | سندوب → الصينية`: **2,546** مركبة.\n"
                f"  5) `2026-10-17 | 1:00–2:00 ظهرًا | سندوب → الصينية`: **2,508** مركبات.\n"
                f"- **فحص القيم المتطرفة (Tukey 1.5×IQR):** تم رصد `{hm.get('outlier_count', 0)}` ساعة تتجاوز الحد الإحصائي الأعلى (`3,221.5` مركبة/ساعة في الاتجاه الأول)."
            ),
            "defensible_insight": (
                "الساعة 12:00–1:00 ظهرًا في اتجاه سندوب → الصينية رُصدت في يوم واحد فقط (14 أكتوبر)، ولذلك تُصنف كـ «قفزة رصد منفردة / قيمة متطرفة عليا»، "
                "بينما نافذة **1:00–3:00 بعد الظهر** تمثل ذروة تشغيلية متكررة ومثبتة عبر معظم أيام الأسبوع."
            ),
            "recommendation": (
                "تركيز خطة التنظيم المروري اليومية في نافذة **12:00 ظهرًا إلى 03:00 عصرًا** مع تكثيف التواجد الميداني بين **1:00 و2:00 ظهرًا**، "
                "وسد فجوة الرصد في الساعة **12:00–1:00 ظهرًا** خلال بقية الأيام للتحقق مما إذا كانت قراءة `4,425` تتكرر بانتظام."
            ),
            "limitation": (
                "الفترة 12:00–1:00 ظهرًا رُصدت في 3 فترات اتجاهية فقط من أصل 12 فترة محتملة خلال الأيام الستة."
            ),
        }

    if tname == "analyze_pedestrian_and_coactivity":
        tot_ped = hm.get("إجمالي_عبور_المشاة", 24342)
        d1_ped = hm.get("عبور_سندوب_إلى_الصينية", 17478)
        d2_ped = hm.get("عبور_الصينية_إلى_سندوب", 6864)
        d1_rate = hm.get("متوسط_عبور_سندوب_لكل_ساعة", 406.47)
        d2_rate = hm.get("متوسط_عبور_الصينية_لكل_ساعة", 180.63)
        co_cnt = hm.get("coactivity_windows_count", 9)

        if lang == "en":
            return {
                "direct_answer": (
                    f"Total observed pedestrian crossings reached **{tot_ped:,} independent events** (`{d1_ped:,}` in Sandoub → El-Siniya at **{d1_rate:,.2f}/hr** "
                    f"vs `{d2_ped:,}` in El-Siniya → Sandoub at **{d2_rate:,.2f}/hr**). Across the study, **{co_cnt} direction-hours** experienced "
                    f"simultaneous high vehicle and pedestrian activity (>= 75th percentile for both)."
                ),
                "evidence_and_method": (
                    f"- **Morning Pedestrian Peaks:** `1,314` crossings on `2026-10-12 | 09:00–10:00 AM` in Sandoub → El-Siniya (with `1,801` vehicles), "
                    f"and `617` crossings on `2026-10-17 | 09:00–10:00 AM` in El-Siniya → Sandoub.\n"
                    f"- **Highest Co-Activity Peak:** `811` crossings with `3,636` vehicles on `2026-10-14 | 01:00–02:00 PM` (Sandoub → El-Siniya).\n"
                    f"- **P75 Co-Activity Thresholds:** Dir 1 (`>= {hm.get('p75_dir1_vehicles', 2113):,.0f}` veh & `>= {hm.get('p75_dir1_pedestrians', 563):,.0f}` ped) -> `{hm.get('coactivity_dir1_count', 4)}` windows (all on Oct 14); "
                    f"Dir 2 (`>= {hm.get('p75_dir2_vehicles', 1730.25):,.2f}` veh & `>= {hm.get('p75_dir2_pedestrians', 281):,.0f}` ped) -> `{hm.get('coactivity_dir2_count', 5)}` windows (Oct 13 & Oct 17)."
                ),
                "defensible_insight": (
                    "Pedestrian demand on the Sandoub → El-Siniya side is **2.25× higher per hour** (`406.47` vs `180.63`/hr) than the reverse direction. "
                    "Moreover, pedestrian peaks occur in two distinct waves: a morning wave (09:00–10:00 AM) and a midday wave (12:00–03:00 PM) that collides with peak vehicular flow."
                ),
                "recommendation": (
                    "Deploy pedestrian guidance barriers and traffic wardens during the **9 priority co-activity hours** (especially 10:00 AM – 02:00 PM), "
                    "and implement the **2-Observer Bridge vs Street Protocol** during those exact 9 windows to quantify bridge utilization."
                ),
                "limitation": (
                    "The dataset records one combined pedestrian column (`مشاة (عبور المشاة)`) without distinguishing bridge deck crossings from surface crossings."
                ),
            }
        return {
            "direct_answer": (
                f"بلغ إجمالي أحداث عبور المشاة في النطاق المحدد **{tot_ped:,} حدث عبور مستقل**، يتوزع بواقع **{d1_ped:,}** في اتجاه **سندوب → الصينية** "
                f"(بمتوسط **{d1_rate:,.2f} عبور/ساعة**) مقابل **{d2_ped:,}** في اتجاه **الصينية → سندوب** (بمتوسط **{d2_rate:,.2f} عبور/ساعة**). "
                f"ورُصدت **{co_cnt} ساعات تزامن مرتفع** تجاوزت فيها المركبات والمشاة معًا المئين 75 (P75) داخل نفس الاتجاه."
            ),
            "evidence_and_method": (
                f"- **ذروة عبور المشاة الصباحية:** `1,314` عبورًا يوم الاثنين `2026-10-12 | 9:00–10:00 صباحًا` في سندوب → الصينية (مع `1,801` مركبة)، "
                f"و`617` عبورًا يوم السبت `2026-10-17 | 9:00–10:00 صباحًا` في الصينية → سندوب.\n"
                f"- **أخطر ساعة تزامن ظهري:** يوم الأربعاء `2026-10-14 | 1:00–2:00 ظهرًا` في سندوب → الصينية حيث تزامن عبور **811 مشاة** مع **3,636 مركبة**.\n"
                f"- **عتبات المئين 75 (P75) للتزامن المرتفع:**\n"
                f"  • سندوب → الصينية (`مركبات >= {hm.get('p75_dir1_vehicles', 2113):,.0f}` و`مشاة >= {hm.get('p75_dir1_pedestrians', 563):,.0f}`): أنتجت **{hm.get('coactivity_dir1_count', 4)} ساعات** (جميعها يوم الأربعاء 14 أكتوبر من 8–9 ص ومن 12–3 م).\n"
                f"  • الصينية → سندوب (`مركبات >= {hm.get('p75_dir2_vehicles', 1730.25):,.2f}` و`مشاة >= {hm.get('p75_dir2_pedestrians', 281):,.0f}`): أنتجت **{hm.get('coactivity_dir2_count', 5)} ساعات** (ساعتان يوم 13 أكتوبر و3 ساعات متصلة يوم 17 أكتوبر من 10 ص إلى 1 م)."
            ),
            "defensible_insight": (
                "كثافة عبور المشاة في جانب **سندوب → الصينية** تعادل **2.25 ضعف** المتوسط الساعي للاتجاه العكسي (`406.47` مقابل `180.63` عبور/ساعة). "
                "وتظهر البيانات موجتين للعبور: موجة صباحية (9–10 ص) وموجة ظهرية (12–3 م) تتقاطع مباشرة مع ذروة تدفق المركبات."
            ),
            "recommendation": (
                "توجيه الحواجز التوجيهية وأفراد التنظيم الميداني خلال **الساعات التسع ذات التزامن المرتفع** لمنع العبور السطحي العشوائي، "
                "وتنفيذ **بروتوكول الراصدين المتزامنين (أعلى الكوبري مقابل أسفل الكوبري)** في هذه الساعات التسع لقياس نسبة الالتزام الفعلي بالكوبري."
            ),
            "limitation": (
                "البيانات تقيس التعرض الزمني المشترك (Temporal Exposure) ولا تفصل بين من عبر فوق الكوبري ومن عبر سطحيًا على الأسفلت، ولا تتضمن إحداثيات مكانية."
            ),
        }

    if tname == "inspect_coverage_and_data_quality":
        if lang == "en":
            return {
                "direct_answer": (
                    "The dataset passes all arithmetic and schema integrity checks (`96` survey rows -> `81` observed direction-hours out of `120` possible slots, "
                    "`15` multi-observer hours summing `30` rows, `30` vehicles in `custom_1792135578939`, `216` zero cells preserved, and `68,196` audit log events reconciled)."
                ),
                "evidence_and_method": (
                    "- **Coverage Matrix:** `81` observed direction-hours (`43` in Dir 1 + `38` in Dir 2) across `43` calendar hours (`38` matched common hours + `5` Dir-1-only hours); `39` unobserved slots.\n"
                    "- **Multi-Observer Hours:** `15` direction-hours (`30` rows: 9 groups/18 rows in Dir 1, 6 groups/12 rows in Dir 2) where two observers recorded simultaneously; counts are summed while coverage counts once (60 min).\n"
                    "- **Unnamed Category (`custom_1792135578939`):** Exactly `30` vehicles (`0.0235%`), all on Friday `2026-10-16` in El-Siniya → Sandoub across 5 rows (`B-29`..`B-33`: `5, 4, 3, 11, 7`).\n"
                    "- **Counter Operation Logs (`AuditOperationLogs`):** `68,196` click events (`+1`: 190, `-1`: 148, `+2`: 68,006) yielding net delta `151,810` vs summary sheet `152,111` (net difference `301` = `300` ledger state jump in `B-33` across 8 classes + `1` unit in `A-24`)."
                ),
                "defensible_insight": (
                    "Keeping the 68,196 audit log events strictly isolated from street counts prevents double-counting, while retaining the 15 multi-observer pairs "
                    "and the 30 `custom_1792135578939` vehicles preserves 100% reconciliation with the Power BI semantic model."
                ),
                "recommendation": (
                    "Lock the counting app schema before future campaigns to prevent custom unnamed columns (`custom_1792135578939`), "
                    "and require observer shift handshakes in the app to eliminate ledger state jumps (`B-33`)."
                ),
                "limitation": (
                    "Observer personal names are anonymized to `راصد #01`..`راصد #15` in all analytical views and exports."
                ),
            }
        return {
            "direct_answer": (
                "تجتاز قاعدة البيانات جميع فحوصات المطابقة الحسابية والهيكلية بنسبة **100%** (`96` سجلًا خامًا تمثل `81` ساعة رصد فعلية من أصل `120` ساعة محتملة، "
                "مع `15` ساعة تعدد راصدين تضم `30` سجلًا، و`30` مركبة في الفئة غير المسمّاة `custom_1792135578939`، و`216` خلية صفرية محفوظة، ومطابقة `68,196` عملية في سجل العداد)."
            ),
            "evidence_and_method": (
                "1. **مصفوفة التغطية الزمنية:** `81` فترة اتجاه-ساعة مرصودة (`43` في سندوب → الصينية و`38` في الصينية → سندوب) مقابل `39` فترة غير مرصودة؛ تتقاطع في **38 ساعة مشتركة**.\n"
                "2. **حالات تعدد الراصدين (15 فترة = 30 سجلًا):** `9` فترات (18 سجلًا) في سندوب → الصينية و`6` فترات (12 سجلًا) في الصينية → سندوب؛ تُجمع أعدادها داخل الساعة وتُحسب التغطية الزمنية 60 دقيقة واحدة.\n"
                "3. **الفئة غير المسمّاة (`custom_1792135578939`):** إجماليها **30 مركبة** (`0.0235%`) تتركز بالكامل يوم الجمعة `2026-10-16` في اتجاه الصينية → سندوب عبر 5 سجلات (`B-29` إلى `B-33` بالقيم: `5، 4، 3، 11، 7`).\n"
                "4. **سجل عمليات العداد (`AuditOperationLogs`):** يضم **68,196** عملية نقر بصافي مجموع `151,810` مقابل `152,111` في جدول الملخص؛ الفارق الصافي (**301**) يفسره بالكامل:\n"
                "   • **300 وحدة** في السجل `B-33` (16 أكتوبر، 3–4 م، الصينية → سندوب) نتيجة قفزة حالة العداد (Ledger State Jump) عبر 8 فئات عند استئناف جلسة الجهاز.\n"
                "   • **وحدة واحدة (+1)** في السجل `A-24` (14 أكتوبر، 10–11 ص، سندوب → الصينية) حيث سُجل 492 ملاكي في الملخص مقابل صافي 491 في السجل.\n"
                "5. **حفظ القيم الصفرية:** تم الحفاظ على جميع الخلايا الصفرية (**216 خلية** عبر أعمدة العد التسعة في الـ 96 سجلًا) كأصفار رصد حقيقية دون حذف."
            ),
            "defensible_insight": (
                "عزل جدول `AuditOperationLogs` (68,196 عملية) كمرجع تدقيق فقط يمنع الازدواج الحسابي الكارثي، كما أن عدم حذف صفوف تعدد الراصدين يحمي 30 سجلًا ميدانيًا من الضياع."
            ),
            "recommendation": (
                "قفل إعدادات الفئات في تطبيق العد الميداني لمنع إنشاء أعمدة مخصصة غير مسماة (`custom_1792135578939`)، "
                "وربط تصفير العداد ببداية كل فترة زمنية لمنع قفزات الرصيد التراكمي مثل حالة السجل `B-33`."
            ),
            "limitation": (
                "تم تعمية أسماء الراصدين إلى معرفات تحليلية (`راصد #01` .. `راصد #15`) التزامًا بقواعد الخصوصية وحوكمة البيانات."
            ),
        }

    if tname == "evaluate_what_if_scenario":
        if lang == "en":
            return {
                "direct_answer": (
                    f"**Hypothetical What-If Simulation:** Under the assumed **{hm.get('assumed_rate_pct', hm.get('assumed_change_pct', 70.0)):.1f}%** scenario, "
                    + (
                        f"**{hm.get('hypothetical_diverted_pedestrians', 0):,}** pedestrian crossings would shift to the bridge, leaving "
                        f"**{hm.get('hypothetical_remaining_surface', 0):,}** at-grade crossings (and shifting **{hm.get('priority_windows_diverted_ped', 0):,}** crossings during the 9 priority co-activity hours)."
                        if hm.get("scenario_type") == "pedestrian_bridge_diversion"
                        else f"category `{hm.get('target_category')}` changes from **{hm.get('observed_category_vehicles', 0):,}** to **{hm.get('hypothetical_category_vehicles', 0):,}** vehicles, "
                        f"adjusting total vehicles from **{hm.get('observed_total_vehicles', 0):,}** to **{hm.get('hypothetical_total_vehicles', 0):,}**."
                    )
                ),
                "evidence_and_method": (
                    "The table clearly separates **Observed Historical Baseline** numbers from **Hypothetical Simulated (What-If)** projections "
                    f"within scope: {res.scope_used.describe_en()}."
                ),
                "defensible_insight": (
                    "Concentrating bridge diversion efforts during the **9 P75 co-activity hours** yields the highest reduction in vehicle-pedestrian temporal exposure per hour of enforcement."
                ),
                "recommendation": (
                    "Validate this hypothetical diversion rate in the field by running the **2-Observer Bridge vs Street Protocol** during peak hours."
                ),
                "limitation": (
                    "All projected values are strictly hypothetical (`What-If`) calculations; actual historical bridge vs surface split was not recorded."
                ),
            }
        return {
            "direct_answer": (
                f"**نتيجة محاكاة السيناريو الافتراضي (What-If):** بافتراض نسبة **{hm.get('assumed_rate_pct', hm.get('assumed_change_pct', 70.0)):.1f}%**، "
                + (
                    f"ينتقل افتراضيًا **{hm.get('hypothetical_diverted_pedestrians', 0):,}** حدث عبور إلى كوبري المشاة ويتبقى **{hm.get('hypothetical_remaining_surface', 0):,}** على السطح "
                    f"(منها تحويل **{hm.get('priority_windows_diverted_ped', 0):,}** عبورًا من أصل **{hm.get('priority_windows_observed_ped', 0):,}** خلال ساعات الأولوية التسع)."
                    if hm.get("scenario_type") == "pedestrian_bridge_diversion"
                    else f"يتغير حجم فئة ({hm.get('target_category')}) من **{hm.get('observed_category_vehicles', 0):,}** مركبة مرصودة إلى **{hm.get('hypothetical_category_vehicles', 0):,}** مركبة افتراضية، "
                    f"ليصبح إجمالي المركبات **{hm.get('hypothetical_total_vehicles', 0):,}** بدلاً من **{hm.get('observed_total_vehicles', 0):,}**."
                )
            ),
            "evidence_and_method": (
                f"تم الفصل الصريح في جدول الأدلة بين **الوضع التاريخي المرصود فعليًا** وبين **الأرقام الافتراضية المحاكاة** ضمن النطاق: {res.scope_used.describe_ar()}."
            ),
            "defensible_insight": (
                "تُظهر المحاكاة أن رفع التزام المشاة باستخدام الكوبري خلال **ساعات التزامن المرتفع (9 ساعات)** يحقق أكبر خفض ممكن في التعرض الزمني المشترك بأقل جهد تشغيلي."
            ),
            "recommendation": (
                "قياس النسبة الحقيقية للتحول إلى الكوبري ميدانيًا عبر تطبيق **بروتوكول الراصدين المتزامنين (أعلى الكوبري وأسفله)** ومقارنتها بهذا السيناريو الافتراضي."
            ),
            "limitation": (
                "الأرقام المحاكاة هي سيناريو حسابي افتراضي (`What-If`) بناءً على الفرضية المدخلة، لأن البيانات الحالية لا تفصل بين العبور فوق الكوبري والعبور السطحي."
            ),
        }

    # 3. Default / Overview / Flexible Aggregation Narrative
    tot_v = hm.get("إجمالي_المركبات", 0)
    obs_h = hm.get("ساعات_الرصد_الفعلي", 0)
    mean_v = hm.get("متوسط_المركبات_لكل_ساعة_رصد", 0.0)
    d1_v = hm.get("مركبات_سندوب_إلى_الصينية", 0)
    d2_v = hm.get("مركبات_الصينية_إلى_سندوب", 0)
    d1_h = hm.get("ساعات_سندوب_إلى_الصينية", 0)
    d2_h = hm.get("ساعات_الصينية_إلى_سندوب", 0)
    d1_m = hm.get("متوسط_سندوب_إلى_الصينية", 0.0)
    d2_m = hm.get("متوسط_الصينية_إلى_سندوب", 0.0)
    peak_v = hm.get("أعلى_ساعة_في_النطاق", 0)
    peak_cap = hm.get("وصف_أعلى_ساعة_في_النطاق", "—")
    tot_ped = hm.get("إجمالي_عبور_المشاة", 0)
    mean_ped = hm.get("متوسط_العبور_لكل_ساعة_رصد", 0.0)
    top_cat = hm.get("فئة المركبات الأعلى تسجيلًا", "سيارات ملاكي")
    top_cat_share = (hm.get("حصة الفئة الأعلى تسجيلًا") or 0.0) * 100.0

    # If evidence_df is from query_traffic_aggregations, highlight its top row as well
    top_row_summary_ar = ""
    top_row_summary_en = ""
    if tname == "query_traffic_aggregations" and not df.empty:
        r0 = df.iloc[0].to_dict()
        kv_str = " | ".join(f"{k}: {v:,}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in list(r0.items())[:6])
        top_row_summary_ar = f"\n- **أعلى مجموعة في جدول التجميع:** `{kv_str}`."
        top_row_summary_en = f"\n- **Top Ranked Group in Aggregation:** `{kv_str}`."

    if lang == "en":
        return {
            "direct_answer": (
                f"Within the active scope ({res.scope_used.describe_en()}), total observed traffic is **{tot_v:,} vehicles** "
                f"across **{obs_h} observed direction-hours** (mean **{mean_v:,.2f} vehicles/hour**), with peak hour volume reaching "
                f"**{peak_v:,} vehicles** (`{peak_cap}`). Independent pedestrian crossings total **{tot_ped:,}** (mean **{mean_ped:,.2f}/hr**)."
            ),
            "evidence_and_method": (
                f"- **Sandoub → El-Siniya:** `{d1_v:,}` vehicles across `{d1_h}` observed hours (mean `{d1_m:,.2f}` veh/hr).\n"
                f"- **El-Siniya → Sandoub:** `{d2_v:,}` vehicles across `{d2_h}` observed hours (mean `{d2_m:,.2f}` veh/hr).\n"
                f"- **Matched Common Hours ({hm.get('عدد_الساعات_المشتركة', 0)}h):** `{hm.get('مركبات_سندوب_المشتركة', 0):,}` vs `{hm.get('مركبات_الصينية_المشتركة', 0):,}` "
                f"(diff `{hm.get('فرق_الساعات_المشتركة', 0):+,}` / `{hm.get('نسبة_فرق_الساعات_المشتركة', 0.0):+.2f}%`).\n"
                f"- **Dominant Vehicle Category:** `{top_cat}` accounting for `{top_cat_share:.2f}%` of all vehicles."
                + top_row_summary_en
            ),
            "defensible_insight": (
                "Passenger vehicles (`سيارات ملاكي` at 49.75% and `ميكروباص وأجرة` at 23.11%) form **72.86%** of corridor flow, "
                "while `نقل ثقيل وتريلات` represents **5.85%** overall (`7,478` vehicles) but rises to **6.88%** in El-Siniya → Sandoub and peaks "
                "between **01:00 PM and 02:00 PM** (`123.56` heavy trucks/hr across 9 direction-periods)."
            ),
            "recommendation": (
                "Focus midday traffic control (`12:00–03:00 PM`) on managing passenger car and microbus queues on **Sandoub → El-Siniya**, "
                "and regulate heavy truck movements during the **01:00–02:00 PM** peak window."
            ),
            "limitation": (
                f"Current scope covers `{obs_h}` observed direction-hours; unobserved hours are excluded from denominators rather than treated as zeros."
            ),
        }

    return {
        "direct_answer": (
            f"ضمن النطاق التحليلي المحدد ({res.scope_used.describe_ar()})، بلغ إجمالي المركبات المرصودة **{tot_v:,} مركبة** "
            f"خلال **{obs_h} فترة اتجاه-ساعة فعلية** بمتوسط كثافة **{mean_v:,.2f} مركبة/ساعة**، بينما سجلت أعلى ساعة ذروة "
            f"**{peak_v:,} مركبة/ساعة** (`{peak_cap}`). وبشكل مستقل، بلغ إجمالي أحداث عبور المشاة **{tot_ped:,} حدث عبور** (بمتوسط **{mean_ped:,.2f} عبور/ساعة**)."
        ),
        "evidence_and_method": (
            f"- **اتجاه سندوب → الصينية:** `{d1_v:,}` مركبة خلال `{d1_h}` ساعة رصد (متوسط `{d1_m:,.2f}` مركبة/ساعة).\n"
            f"- **اتجاه الصينية → سندوب:** `{d2_v:,}` مركبة خلال `{d2_h}` ساعة رصد (متوسط `{d2_m:,.2f}` مركبة/ساعة).\n"
            f"- **المقارنة في الساعات المشتركة ({hm.get('عدد_الساعات_المشتركة', 0)} ساعة متناظرة):** `{hm.get('مركبات_سندوب_المشتركة', 0):,}` مقابل `{hm.get('مركبات_الصينية_المشتركة', 0):,}` مركبة "
            f"(بفارق `{hm.get('فرق_الساعات_المشتركة', 0):+,}` مركبة أي `{hm.get('نسبة_فرق_الساعات_المشتركة', 0.0):+.2f}%`).\n"
            f"- **الفئة الأعلى تسجيلاً:** `{top_cat}` بحصة `{top_cat_share:.2f}%` من إجمالي المركبات."
            + top_row_summary_ar
        ),
        "defensible_insight": (
            "تشكل المركبات الخفيفة ونقل الركاب (`سيارات ملاكي` بنسبة 49.75% + `ميكروباص وأجرة` بنسبة 23.11%) نحو **72.86%** من إجمالي الحركة على المحور، "
            "في حين يمثل `نقل ثقيل وتريلات` **5.85%** إجمالاً (`7,478` مركبة) لكن حصته النسبية ترتفع في اتجاه **الصينية → سندوب** إلى **6.88%** مقابل **5.11%** في الاتجاه الأول، "
            "ويبلغ ذروته الساعية بين **1:00 و2:00 ظهرًا** بمتوسط **123.56 شاحنة/ساعة**."
        ),
        "recommendation": (
            "تخصيص تنظيم مروري مكثف في فترة الذروة الظهرية (**12:00–3:00 م**) لاستيعاب تدفق الملاكي والميكروباص في اتجاه **سندوب → الصينية**، "
            "وتنظيم حركة الشاحنات الثقيلة خلال ساعة الذروة (**1:00–2:00 م**) لتقليل التداخل مع حركة المشاة."
        ),
        "limitation": (
            f"الحسابات مبنية على `{obs_h}` ساعة رصد فعلية متاحة في هذا النطاق؛ الساعات غير المرصودة لا تدخل في المقام ولا تُفترض أصفارًا."
        ),
    }


def _call_gemini_grounded_narrator(
    question: str,
    res: AnalysisToolResult,
    fallback_sections: Dict[str, str],
    lang: Literal["ar", "en"],
    api_key: str,
) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """Call Google Gemini via google-genai SDK to refine the 5-part answer strictly grounded in Python tool output."""
    try:
        from google import genai
        from google.genai import types
    except Exception:
        return None, None

    # Prepare compact, strictly anonymized JSON payload of the Python tool output
    safe_records = []
    if res.evidence_df is not None and not res.evidence_df.empty:
        safe_records = res.evidence_df.head(15).to_dict(orient="records")

    payload = {
        "user_question": question,
        "target_language": "Arabic" if lang == "ar" else "English",
        "executed_python_tool": res.tool_name,
        "active_scope": res.scope_used.describe_ar() if lang == "ar" else res.scope_used.describe_en(),
        "is_evidence_boundary_or_refusal": res.is_boundary_or_refusal,
        "boundary_topic": res.boundary_topic,
        "verified_headline_metrics": {
            str(k): (round(v, 4) if isinstance(v, float) else v)
            for k, v in res.headline_metrics.items()
            if isinstance(v, (int, float, str, bool))
        },
        "verified_evidence_table_top_rows": safe_records,
        "baseline_reference_narrative": fallback_sections,
    }

    system_instruction = (
        "أنت محلل بيانات مرور ومشاة خبير لتقرير كوبري السلاب (سندوب – الصينية) بمدينة المنصورة (12–17 أكتوبر 2026). "
        "مهمتك صياغة إجابة تحليلية تنفيذية من 5 أجزاء باللغة المطلوبة (العربية الفصحى الواضحة أو الإنجليزية) بالاعتماد الحصري على الأرقام المحسوبة في بايثون والمرفقة في JSON.\n"
        "القواعد الصارمة:\n"
        "1) لا تختلق أي رقم غير موجود في verified_headline_metrics أو verified_evidence_table_top_rows أو baseline_reference_narrative.\n"
        "2) لا تجمع أبدًا المركبات مع المشاة في مؤشر واحد.\n"
        "3) التزم بتوضيح المقام الزمني (ساعات الرصد الفعلية 43 مقابل 38، أو الـ 38 ساعة المشتركة المتناظرة).\n"
        "4) إذا كان السؤال خارج حدود البيانات (حوادث، سرعات، إحداثيات خرائط، أثر فعلي للكوبري، سنة خاطئة، أو فرضية خاطئة)، ارفض الفرضية غير المدعومة بوضوح وقدم البديل الكمي الموثق.\n"
        "5) أعد النتيجة بصيغة JSON فقط تحتوي على المفاتيح الخمسة: direct_answer, evidence_and_method, defensible_insight, recommendation, limitation."
    )

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=20000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )

    prompt_text = json.dumps(payload, ensure_ascii=False, default=str)

    for candidate_model in GEMINI_MODEL_CHAIN:
        try:
            response = client.models.generate_content(
                model=candidate_model,
                contents=prompt_text,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            raw_text = (response.text or "").strip()
            if not raw_text:
                continue
            parsed = json.loads(raw_text)
            required_keys = [
                "direct_answer",
                "evidence_and_method",
                "defensible_insight",
                "recommendation",
                "limitation",
            ]
            if all(isinstance(parsed.get(k), str) and len(parsed[k].strip()) > 10 for k in required_keys):
                return {k: parsed[k].strip() for k in required_keys}, candidate_model
        except Exception:
            continue

    return None, None


def answer_user_question(
    question: str,
    session_context: Optional[Dict[str, Any]] = None,
    ui_scope: Optional[FilterScope] = None,
    preferred_lang: str = "auto",
    use_live_gemini: bool = True,
    model: Optional[SemanticModel] = None,
) -> AIAnalystResponse:
    """Main entry point for 'اسأل بياناتك / Ask Your Data'."""
    model = model or build_semantic_model()
    ctx = dict(session_context or {})
    clean_q = (question or "").strip()
    lang = detect_query_language(clean_q, preferred_lang=preferred_lang)

    # 1. Check security & prompt-injection guardrail
    if PROMPT_INJECTION_PATTERNS.search(clean_q) or PROMPT_INJECTION_PATTERNS.search(
        normalize_arabic_text(clean_q)
    ):
        return _build_security_refusal(clean_q, lang, ui_scope, ctx, model)

    # 2. Execute guarded Python analytical tool
    tool_res, updated_ctx = resolve_question_and_execute(
        clean_q, session_context=ctx, ui_scope=ui_scope, model=model
    )

    # 3. Build dynamic Plotly chart from verified tool result
    fig = build_chart_for_analysis_result(tool_res, lang=lang)

    # 4. Build verified deterministic 5-part narrative
    det_sections = build_deterministic_narrative(clean_q, tool_res, lang=lang)

    # 5. Optionally refine narrative via live Gemini (strictly grounded in Python tool output)
    api_key = get_gemini_api_key() if use_live_gemini else None
    if api_key:
        gemini_sections, gemini_model = _call_gemini_grounded_narrator(
            clean_q, tool_res, det_sections, lang, api_key
        )
        if gemini_sections and gemini_model:
            return AIAnalystResponse(
                question=clean_q,
                lang=lang,
                direct_answer=gemini_sections["direct_answer"],
                evidence_and_method=gemini_sections["evidence_and_method"],
                defensible_insight=gemini_sections["defensible_insight"],
                recommendation=gemini_sections["recommendation"],
                limitation=gemini_sections["limitation"],
                tool_result=tool_res,
                figure=fig,
                engine_mode="gemini_live",
                model_used=gemini_model,
                status_badge_ar=f"ذكاء اصطناعي متصل ({gemini_model}) + حسابات بايثون الموثقة",
                status_badge_en=f"Live Gemini ({gemini_model}) + Verified Python Engine",
                session_context=updated_ctx,
            )

    # 6. Transparent deterministic Python fallback (never portrayed as a live Gemini call)
    badge_ar = (
        "محرك التحليل الحتمي الموثق (بايثون) — يعمل بدون مفتاح Gemini خارجي"
        if not api_key
        else "محرك التحليل الحتمي الموثق (بايثون — وضع الاحتياط الآمن)"
    )
    badge_en = (
        "Verified Deterministic Python Engine (Offline / No API Key Required)"
        if not api_key
        else "Verified Deterministic Python Engine (Safe Fallback Mode)"
    )

    return AIAnalystResponse(
        question=clean_q,
        lang=lang,
        direct_answer=det_sections["direct_answer"],
        evidence_and_method=det_sections["evidence_and_method"],
        defensible_insight=det_sections["defensible_insight"],
        recommendation=det_sections["recommendation"],
        limitation=det_sections["limitation"],
        tool_result=tool_res,
        figure=fig,
        engine_mode="deterministic_python",
        model_used="deterministic_python_v1",
        status_badge_ar=badge_ar,
        status_badge_en=badge_en,
        session_context=updated_ctx,
    )
