"""Configuration, constants, color palette, and analytical contract definitions."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
INPUTS_DIR: Path = PROJECT_ROOT / "inputs"
CACHE_DIR: Path = PROJECT_ROOT / "data_cache"
MANIFEST_PATH: Path = PROJECT_ROOT / "input_manifest.json"

SURVEY_XLSX_PATH: Path = INPUTS_DIR / "Traffic_Survey_Combined.xlsx"
POWERBI_ZIP_PATH: Path = INPUTS_DIR / "Traffic_Arabic_PowerBI_Enhanced.zip"
PROFILING_XLSX_PATH: Path = INPUTS_DIR / "Traffic_Data_Profiling.xlsx"
ASSESSMENT_PDF_PATH: Path = INPUTS_DIR / "Traffic_Data_Assessment_Report.pdf"
STORY_PDF_PATH: Path = INPUTS_DIR / "Traffic_Client_Story_AR.pdf"

# Study metadata
STUDY_LOCATION_AR = "كوبري السلاب (سندوب – الصينية)"
STUDY_LOCATION_EN = "El-henawy / El-Salab Bridge (Sandoub – El-Siniya)"
STUDY_DATES_AR = "12 إلى 17 أكتوبر 2026"
STUDY_DATES_EN = "12–17 October 2026"
STUDY_YEAR = 2026
STUDY_MONTH = 10
STUDY_DAYS = [12, 13, 14, 15, 16, 17]
STUDY_DATES_ISO: List[str] = [f"2026-10-{d:02d}" for d in STUDY_DAYS]


# Direction mappings
DIRECTION_SOURCE_TO_KEY: Dict[str, int] = {
    "سندوب - الصينية": 1,
    "الصينية - سندوب": 2,
    "طريق سندوب - الصينية": 1,
    "طريق الصينية - سندوب": 2,
    "من سندوب إلى الصينية": 1,
    "من الصينية إلى سندوب": 2,
    "سندوب → الصينية": 1,
    "الصينية → سندوب": 2,
}

DIRECTION_KEY_TO_AR: Dict[int, str] = {
    1: "من سندوب إلى الصينية",
    2: "من الصينية إلى سندوب",
}

DIRECTION_KEY_TO_SHORT_AR: Dict[int, str] = {
    1: "سندوب → الصينية",
    2: "الصينية → سندوب",
}

DIRECTION_KEY_TO_EN: Dict[int, str] = {
    1: "Sandoub → El-Siniya",
    2: "El-Siniya → Sandoub",
}

# Vehicle categories (exact order from Power BI DimVehicleCategory)
VEHICLE_CATEGORIES_SPEC: List[Dict[str, object]] = [
    {
        "order": 1,
        "source_col": "سيارات ملاكي",
        "name_ar": "سيارات ملاكي",
        "name_en": "Private Cars",
    },
    {
        "order": 2,
        "source_col": "ميكروباص وأجرة",
        "name_ar": "ميكروباص وأجرة",
        "name_en": "Microbus & Taxi",
    },
    {
        "order": 3,
        "source_col": "أتوبيسات نقل",
        "name_ar": "أتوبيسات نقل",
        "name_en": "Transport Buses",
    },
    {
        "order": 4,
        "source_col": "نقل خفيف وسوزوكي",
        "name_ar": "نقل خفيف وسوزوكي",
        "name_en": "Light Transport & Suzuki",
    },
    {
        "order": 5,
        "source_col": "نقل ثقيل وتريلات",
        "name_ar": "نقل ثقيل وتريلات",
        "name_en": "Heavy Transport & Trailers",
    },
    {
        "order": 6,
        "source_col": "موتوسيكلات ودراجات",
        "name_ar": "موتوسيكلات ودراجات",
        "name_en": "Motorcycles & Bicycles",
    },
    {
        "order": 7,
        "source_col": "توك توك وتروسيكل",
        "name_ar": "توك توك وتروسيكل",
        "name_en": "Tuk-Tuk & Tricycle",
    },
    {
        "order": 8,
        "source_col": "custom_1792135578939",
        "name_ar": "فئة غير مسمّاة",
        "name_en": "Unnamed Category (custom_1792135578939)",
    },
]

SOURCE_VEHICLE_COLS: List[str] = [str(c["source_col"]) for c in VEHICLE_CATEGORIES_SPEC]
DISPLAY_VEHICLE_CATS_AR: List[str] = [str(c["name_ar"]) for c in VEHICLE_CATEGORIES_SPEC]
VEHICLE_COL_TO_DISPLAY_AR: Dict[str, str] = {
    str(c["source_col"]): str(c["name_ar"]) for c in VEHICLE_CATEGORIES_SPEC
}
VEHICLE_COL_TO_DISPLAY_AR["مشاة (عبور المشاة)"] = "عبور المشاة (مؤشر مستقل)"


# Color palette reconciled with Power BI Enhanced Report
COLORS: Dict[str, str] = {
    "dir1": "#0F766E",          # Teal for من سندوب إلى الصينية
    "dir1_light": "#14B8A6",
    "dir2": "#334155",          # Slate for من الصينية إلى سندوب
    "dir2_light": "#64748B",
    "pedestrian": "#7C3AED",    # Purple for عبور المشاة
    "pedestrian_light": "#A78BFA",
    "coactivity": "#D97706",    # Amber for high co-activity windows
    "matched": "#0284C7",       # Sky blue for matched comparison
    "bg_card": "#FFFFFF",
    "border": "#E2E8F0",
    "text_primary": "#0F172A",
    "text_secondary": "#475569",
    "success": "#059669",
    "warning": "#D97706",
    "danger": "#DC2626",
}

CATEGORY_PALETTE: List[str] = [
    "#0F766E",  # سيارات ملاكي
    "#0284C7",  # ميكروباص وأجرة
    "#2563EB",  # أتوبيسات نقل
    "#0D9488",  # نقل خفيف وسوزوكي
    "#D97706",  # نقل ثقيل وتريلات
    "#7C3AED",  # موتوسيكلات ودراجات
    "#475569",  # توك توك وتروسيكل
    "#94A3B8",  # فئة غير مسمّاة
]

# Gemini candidate models in priority order (verified with google-genai SDK)
GEMINI_MODEL_CHAIN: List[str] = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]


def get_user_gemini_config_path() -> Path:
    """Return per-user local configuration file path on the client's Windows PC (never shipped in package)."""
    override_dir = os.environ.get("BRIDGE_TRAFFIC_USER_DATA_DIR", "").strip()
    if override_dir:
        base_dir = Path(override_dir)
    else:
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        base_dir = Path(local_app_data) if local_app_data else (Path.home() / "AppData" / "Local")
    return base_dir / "BridgeTrafficAI" / "user_gemini_config.json"


def load_saved_gemini_api_key() -> Optional[str]:
    """Read Gemini API key saved locally by the client via the GUI configurator or app sidebar."""
    cfg_path = get_user_gemini_config_path()
    try:
        if cfg_path.exists():
            # utf-8-sig handles BOM if written by Windows .NET tools
            raw = cfg_path.read_text(encoding="utf-8-sig").strip()
            if raw:
                data = json.loads(raw)
                key = str(data.get("gemini_api_key", "")).strip()
                if key and key != "replace_locally_do_not_commit":
                    return key
    except Exception:
        pass
    return None


def save_gemini_api_key_locally(api_key: str) -> Path:
    """Persist the client's Gemini API key locally in %LOCALAPPDATA%\\BridgeTrafficAI."""
    cfg_path = get_user_gemini_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"gemini_api_key": api_key.strip()}
    cfg_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return cfg_path


def clear_saved_gemini_api_key() -> None:
    """Remove any locally saved Gemini API key on this PC."""
    cfg_path = get_user_gemini_config_path()
    try:
        if cfg_path.exists():
            cfg_path.unlink()
    except Exception:
        pass


def get_gemini_api_key() -> Optional[str]:
    """Retrieve Gemini API key safely from session, local user GUI config, environment, or secrets."""
    try:
        import streamlit as st
        if "gemini_api_key_session" in st.session_state:
            s_key = str(st.session_state["gemini_api_key_session"]).strip()
            if s_key and s_key != "replace_locally_do_not_commit":
                return s_key
    except Exception:
        pass

    saved_key = load_saved_gemini_api_key()
    if saved_key:
        return saved_key

    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key and env_key != "replace_locally_do_not_commit":
        return env_key

    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            sec_key = str(st.secrets["GEMINI_API_KEY"]).strip()
            if sec_key and sec_key != "replace_locally_do_not_commit":
                return sec_key
    except Exception:
        pass
    return None
