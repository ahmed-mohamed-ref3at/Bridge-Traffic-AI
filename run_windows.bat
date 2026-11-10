@echo off
chcp 65001 >nul
echo ============================================================================
echo   Bridge Traffic ^& Pedestrian AI Data Analyst (Streamlit + Gemini)
echo   كوبري السلاب (سندوب - الصينية) | 12 - 17 أكتوبر 2026
echo ============================================================================
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

python -m streamlit run app.py --server.headless=false
pause
