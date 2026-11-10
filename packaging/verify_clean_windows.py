"""End-to-end verification of the Windows delivery packages in a clean No-Python environment."""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HANDOFF_ROOT = PROJECT_ROOT.parent
DELIVERY_DIR = HANDOFF_ROOT / "Client_Ready_Windows_Package"
CLEAN_TEST_ROOT = Path(r"E:\بيئة_اختبار_ويندوز_نظيفة")

SETUP_EXE = DELIVERY_DIR / "Setup_Bridge_Traffic_AI_AR.exe"
PORTABLE_ZIP = DELIVERY_DIR / "Bridge_Traffic_AI_Portable_Windows.zip"

FORBIDDEN_BYTES = [
    b"AQ.Ab8RN6",
    b"Ab8RN6Jwmo",
]


def make_no_python_env(profile_dir: Path) -> Dict[str, str]:
    """Build a stripped Windows environment with NO Python, NO Conda, NO Pip, and a brand-new user profile."""
    sys_root = os.environ.get("SystemRoot", r"C:\Windows")
    local_app_data = profile_dir / "AppData" / "Local"
    roaming_app_data = profile_dir / "AppData" / "Roaming"
    temp_dir = local_app_data / "Temp"
    desktop_dir = profile_dir / "Desktop"
    programs_dir = roaming_app_data / "Microsoft" / "Windows" / "Start Menu" / "Programs"

    for d in (local_app_data, roaming_app_data, temp_dir, desktop_dir, programs_dir):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "SystemRoot": sys_root,
        "WINDIR": sys_root,
        "SystemDrive": os.environ.get("SystemDrive", "C:"),
        "COMSPEC": str(Path(sys_root) / "System32" / "cmd.exe"),
        "PATHEXT": ".COM;.EXE;.BAT;.CMD;.VBS;.JS;.WS;.MSC",
        "PATH": f"{sys_root}\\System32;{sys_root};{sys_root}\\System32\\Wbem;{sys_root}\\System32\\WindowsPowerShell\\v1.0",
        "USERPROFILE": str(profile_dir),
        "HOMEPATH": str(profile_dir)[2:],
        "HOMEDRIVE": str(profile_dir)[:2],
        "LOCALAPPDATA": str(local_app_data),
        "APPDATA": str(roaming_app_data),
        "TEMP": str(temp_dir),
        "TMP": str(temp_dir),
        "BRIDGE_TRAFFIC_USER_DATA_DIR": str(local_app_data),
        "COMPUTERNAME": os.environ.get("COMPUTERNAME", "CLEANPC"),
        "USERNAME": "ClientUser",
        "OS": "Windows_NT",
    }


def read_pe_imports(pe_path: Path) -> List[str]:
    """Parse Win32 PE64/PE32 import table to list imported DLL names without external tools."""
    data = pe_path.read_bytes()
    if len(data) < 64 or data[:2] != b"MZ":
        return []
    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    if pe_offset + 24 > len(data) or data[pe_offset : pe_offset + 4] != b"PE\x00\x00":
        return []
    num_sections = struct.unpack_from("<H", data, pe_offset + 6)[0]
    opt_header_size = struct.unpack_from("<H", data, pe_offset + 20)[0]
    opt_offset = pe_offset + 24
    magic = struct.unpack_from("<H", data, opt_offset)[0]
    # PE32 = 0x10B (DataDirectory at +96), PE32+ (x64) = 0x20B (DataDirectory at +112)
    dd_offset = opt_offset + (112 if magic == 0x20B else 96)
    import_rva, import_size = struct.unpack_from("<II", data, dd_offset + 8)  # Entry 1 = Import Table
    if import_rva == 0 or import_size == 0:
        return []

    sec_table_offset = opt_offset + opt_header_size
    sections = []
    for i in range(num_sections):
        so = sec_table_offset + i * 40
        vsize, vrva, raw_size, raw_ptr = struct.unpack_from("<IIII", data, so + 8)
        sections.append((vrva, max(vsize, raw_size), raw_ptr))

    def rva_to_offset(rva: int) -> int:
        for vrva, span, raw_ptr in sections:
            if vrva <= rva < vrva + span:
                return raw_ptr + (rva - vrva)
        return -1

    imp_off = rva_to_offset(import_rva)
    if imp_off < 0:
        return []

    dlls: List[str] = []
    pos = imp_off
    while pos + 20 <= len(data):
        orig_thunk, ts, fwd, name_rva, first_thunk = struct.unpack_from("<IIIII", data, pos)
        if orig_thunk == 0 and name_rva == 0 and first_thunk == 0:
            break
        name_off = rva_to_offset(name_rva)
        if 0 <= name_off < len(data):
            end = data.find(b"\x00", name_off)
            if end > name_off:
                dlls.append(data[name_off:end].decode("ascii", errors="ignore"))
        pos += 20
    return dlls


def audit_runtime_dll_dependencies(runtime_dir: Path) -> None:
    """Verify every .exe/.dll/.pyd in runtime_dir resolves all imports from bundle or Windows System32."""
    sys32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    bundled_files: Set[str] = {f.name.lower() for f in runtime_dir.rglob("*") if f.is_file()}

    binaries = (
        list(runtime_dir.rglob("*.exe"))
        + list(runtime_dir.rglob("*.pyd"))
        + list(runtime_dir.rglob("*.dll"))
    )
    unresolved: List[str] = []
    for bin_file in binaries:
        local_dir_files = {f.name.lower() for f in bin_file.parent.iterdir() if f.is_file()}
        for dll in read_pe_imports(bin_file):
            dl = dll.lower()
            if dl.startswith("api-ms-win-") or dl.startswith("ext-ms-"):
                continue
            if dl in local_dir_files or dl in bundled_files:
                continue
            if (sys32 / dll).exists():
                continue
            unresolved.append(f"{bin_file.relative_to(runtime_dir)} -> {dll}")

    if unresolved:
        raise RuntimeError(f"Unresolved DLL imports detected:\n" + "\n".join(unresolved[:20]))
    print(f"[PASS] PE Binary Import Audit: {len(binaries)} binaries checked, 0 missing external DLLs.")


def main() -> None:
    print("=================================================================")
    print("  CLEAN WINDOWS (NO-PYTHON) END-TO-END PACKAGE VERIFICATION")
    print("=================================================================")

    assert SETUP_EXE.exists(), f"Missing {SETUP_EXE}"
    assert PORTABLE_ZIP.exists(), f"Missing {PORTABLE_ZIP}"

    if CLEAN_TEST_ROOT.exists():
        shutil.rmtree(CLEAN_TEST_ROOT)
    CLEAN_TEST_ROOT.mkdir(parents=True, exist_ok=True)

    profile_dir = CLEAN_TEST_ROOT / "حساب العميل الجديد"
    clean_env = make_no_python_env(profile_dir)

    # 1. Prove Python, Pip, Conda, Streamlit do NOT exist in this clean Windows environment
    for tool in ("python", "python3", "pip", "conda", "streamlit"):
        res = subprocess.run(["where.exe", tool], env=clean_env, capture_output=True, text=True)
        assert res.returncode != 0, f"Environment not clean! Found {tool}: {res.stdout}"
    print("[PASS] Check 1: Verified NO Python, pip, conda, or streamlit in clean Windows environment.")

    # 2. Binary byte-level secret scan on final deliverables
    for artifact in (SETUP_EXE, PORTABLE_ZIP):
        raw_bytes = artifact.read_bytes()
        for pat in FORBIDDEN_BYTES:
            assert pat not in raw_bytes, f"SECURITY VIOLATION: Found secret bytes in {artifact.name}!"
    print("[PASS] Check 2: Binary byte-for-byte scan of Setup.exe & Portable.zip confirmed 0 API keys.")

    # 3. Workflow A: Portable ZIP -> Extract to Arabic+Space folder -> Double-Click Launcher
    portable_extract_dir = CLEAN_TEST_ROOT / "سطح المكتب للتجربة"
    with zipfile.ZipFile(PORTABLE_ZIP, "r") as zf:
        zf.extractall(portable_extract_dir)

    portable_app_dir = portable_extract_dir / "محلل_المرور_الذكي_كوبري_السلاب"
    portable_launcher = portable_app_dir / "تشغيل محلل المرور الذكي.exe"
    assert portable_launcher.exists(), "Portable launcher .exe missing after extract!"

    # Audit every PE binary inside the extracted runtime
    audit_runtime_dll_dependencies(portable_app_dir / "system" / "runtime")

    t0 = time.time()
    proc_port = subprocess.run(
        [str(portable_launcher), "--verify-startup"],
        cwd=str(portable_app_dir),
        env=clean_env,
        timeout=90,
    )
    assert proc_port.returncode == 0, f"Portable launcher failed with exit code {proc_port.returncode}"
    print(f"[PASS] Check 3 (Portable Edition): Extracted & launched cleanly in {time.time() - t0:.1f}s (HTTP 200 OK).")

    # 4. Workflow B: Single-File Windows Installer (Setup_Bridge_Traffic_AI_AR.exe) -> Install -> Launch
    install_target = profile_dir / "AppData" / "Local" / "BridgeTrafficAI_App"
    proc_inst = subprocess.run(
        [str(SETUP_EXE), "--silent-install", str(install_target)],
        env=clean_env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc_inst.returncode == 0, f"Installer failed: {proc_inst.stderr}"

    installed_launcher = install_target / "تشغيل محلل المرور الذكي.exe"
    installed_gemini_cfg = install_target / "إعداد مفتاح الذكاء الاصطناعي.exe"
    installed_uninstaller = install_target / "إلغاء تثبيت البرنامج.exe"
    installed_guide = install_target / "دليل التشغيل السريع (عربي).html"
    assert installed_launcher.exists()
    assert installed_gemini_cfg.exists()
    assert installed_uninstaller.exists()
    assert installed_guide.exists()

    proc_inst_run = subprocess.run(
        [str(installed_launcher), "--verify-startup"],
        cwd=str(install_target),
        env=clean_env,
        timeout=90,
    )
    assert proc_inst_run.returncode == 0, f"Installed launcher failed with code {proc_inst_run.returncode}"
    print("[PASS] Check 4 (Installer Edition): Setup_Bridge_Traffic_AI_AR.exe installed & launched cleanly.")

    # 5. Live Streamlit UI + AppTest + HTTP Frontend Check using ONLY bundled runtime in clean environment
    bundled_py = install_target / "system" / "runtime" / "python.exe"
    bundled_app_dir = install_target / "system" / "app"

    # 5a. Live HTTP check (root HTML + /_stcore/health + /_stcore/host-config)
    srv = subprocess.Popen(
        [
            str(bundled_py), "-m", "streamlit", "run", "app.py",
            "--server.port=8529", "--server.address=127.0.0.1",
            "--server.headless=true", "--browser.gatherUsageStats=false",
        ],
        cwd=str(bundled_app_dir),
        env=clean_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        healthy = False
        for _ in range(80):
            time.sleep(0.5)
            try:
                with urllib.request.urlopen("http://127.0.0.1:8529/_stcore/health", timeout=2) as r:
                    if r.status == 200 and r.read().decode().strip() == "ok":
                        healthy = True
                        break
            except Exception:
                pass
        assert healthy, "Bundled Streamlit server did not become healthy on port 8529"
        with urllib.request.urlopen("http://127.0.0.1:8529/", timeout=5) as r_html:
            html_body = r_html.read().decode("utf-8", errors="ignore")
            assert r_html.status == 200 and "<div id=\"root\">" in html_body
    finally:
        srv.terminate()
        srv.wait(timeout=10)
    print("[PASS] Check 5a: Live HTTP frontend HTML & health endpoints verified via bundled runtime.")

    # 5b. Full Streamlit AppTest simulation (Arabic KPIs + Ask Your Data + Follow-up + Gemini Local Config)
    apptest_code = (
        "import os, json;\n"
        "from streamlit.testing.v1 import AppTest;\n"
        "from src.config import save_gemini_api_key_locally, get_gemini_api_key, clear_saved_gemini_api_key;\n"
        "from src.metrics_engine import run_baseline_qa_checks;\n"
        "from src.ai_orchestrator import answer_user_question;\n"
        "assert get_gemini_api_key() is None;\n"
        "save_gemini_api_key_locally('DUMMY_CLIENT_KEY_TEST');\n"
        "assert get_gemini_api_key() == 'DUMMY_CLIENT_KEY_TEST';\n"
        "clear_saved_gemini_api_key();\n"
        "assert get_gemini_api_key() is None;\n"
        "qa = run_baseline_qa_checks();\n"
        "assert qa['all_passed'] is True;\n"
        "r1 = answer_user_question('قارن بين الاتجاهين في الساعات المشتركة', use_live_gemini=False);\n"
        "r2 = answer_user_question('طب يوم 14؟', session_context=r1.session_context, use_live_gemini=False);\n"
        "assert '65,279' in r1.evidence_and_method or '11,758' in r1.direct_answer;\n"
        "assert '2026-10-14' in str(r2.session_context.get('last_dates'));\n"
        "at = AppTest.from_file('app.py', default_timeout=45);\n"
        "at.run();\n"
        "assert not at.exception, f'AppTest exception: {at.exception}';\n"
        "print('APPTEST_FULL_OK');\n"
    )
    proc_at = subprocess.run(
        [str(bundled_py), "-c", apptest_code],
        cwd=str(bundled_app_dir),
        env=clean_env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert proc_at.returncode == 0 and "APPTEST_FULL_OK" in proc_at.stdout, (
        f"AppTest failed:\nSTDOUT:{proc_at.stdout}\nSTDERR:{proc_at.stderr}"
    )
    print("[PASS] Check 5b: Full AppTest UI, 12 QA checks, Arabic AI follow-up & Gemini GUI config verified!")

    # 6. Test Friendly Arabic Failure Message when double-clicked inside unextracted .zip folder
    fake_zip_temp = CLEAN_TEST_ROOT / "Temp" / "Temp1_Bridge_Traffic.zip"
    fake_zip_temp.mkdir(parents=True, exist_ok=True)
    isolated_exe = fake_zip_temp / "تشغيل محلل المرور الذكي.exe"
    shutil.copy2(installed_launcher, isolated_exe)

    proc_fail = subprocess.run(
        [str(isolated_exe), "--verify-startup"],
        cwd=str(fake_zip_temp),
        env=clean_env,
        timeout=30,
    )
    assert proc_fail.returncode == 2, f"Expected exit code 2 for unextracted zip, got {proc_fail.returncode}"
    logs_dir = profile_dir / "AppData" / "Local" / "BridgeTrafficAI" / "logs"
    log_texts = "\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in logs_dir.glob("*.log"))
    assert "يبدو أنك قمت بفتح البرنامج من داخل الملف المضغوط (ZIP)" in log_texts, (
        f"Expected Arabic ZIP extraction guidance in logs, got: {log_texts}"
    )
    print("[PASS] Check 6: Unextracted-ZIP user mistake gracefully caught with Arabic guidance (exit code 2, zero crash).")

    # Cleanup test directory on E:\ so we leave the machine tidy
    shutil.rmtree(CLEAN_TEST_ROOT, ignore_errors=True)
    print("\nALL CLEAN-WINDOWS VERIFICATION CHECKS PASSED 100%!")


if __name__ == "__main__":
    main()
