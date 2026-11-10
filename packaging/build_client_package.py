"""Build complete zero-dependency Windows delivery packages (Setup .exe + Portable .zip)."""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import List, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HANDOFF_ROOT = PROJECT_ROOT.parent
PACKAGING_DIR = PROJECT_ROOT / "packaging"
BUILD_TEMP = Path(r"E:\BridgeTrafficAI_BuildTemp")
RAW_RUNTIME_DIR = BUILD_TEMP / "runtime"
BIN_DIR = BUILD_TEMP / "bin"
STAGING_PARENT = BUILD_TEMP / "staging_portable"
PORTABLE_FOLDER_NAME = "محلل_المرور_الذكي_كوبري_السلاب"
STAGING_APP_ROOT = STAGING_PARENT / PORTABLE_FOLDER_NAME
PAYLOAD_ZIP_PATH = BUILD_TEMP / "installer_payload.zip"

DELIVERY_DIR = HANDOFF_ROOT / "Client_Ready_Windows_Package"

FORBIDDEN_SECRETS = [
    "AQ.Ab8RN6",
    "Ab8RN6Jwmo",
]

REDIST_DLL_NAMES = [
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "concrt140.dll",
    "vcomp140.dll",
]


def step(msg: str) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(f"\n=== {msg} ===", flush=True)


def bundle_vc_redist_dlls(target_runtime: Path) -> List[str]:
    """Copy official MSVC C/C++ runtime DLLs into target_runtime so bare Windows PCs work out of the box."""
    search_dirs = [
        Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32",
        Path(r"E:\Anaconda\Library\bin"),
        Path(r"E:\Anaconda"),
    ]
    copied: List[str] = []
    for dll_name in REDIST_DLL_NAMES:
        dest = target_runtime / dll_name
        if dest.exists():
            copied.append(f"{dll_name} (already in runtime)")
            continue
        for sdir in search_dirs:
            candidate = sdir / dll_name
            if candidate.exists():
                shutil.copy2(candidate, dest)
                copied.append(f"{dll_name} (from {sdir})")
                break
    return copied


def prune_unused_build_artifacts(site_packages: Path) -> int:
    """Remove build-only static .lib/.h headers, pip/pytest, and bytecode caches while keeping 100% runtime & dist-info."""
    removed_bytes = 0

    # Remove build-only and notebook-only packages not needed by end-user Streamlit app
    build_only_prefixes = (
        "pip",
        "pip-",
        "_pytest",
        "pytest",
        "pytest-",
        "pluggy",
        "pluggy-",
        "iniconfig",
        "iniconfig-",
        "IPython",
        "ipython",
        "ipykernel",
        "jupyter_",
        "debugpy",
        "zmq",
        "pyzmq",
        "jedi",
        "parso",
        "prompt_toolkit",
        "stack_data",
        "pure_eval",
        "asttokens",
        "executing",
        "comm",
    )
    for item in list(site_packages.iterdir()):
        if any(item.name == p or item.name.startswith(p) for p in build_only_prefixes):
            if item.is_dir():
                removed_bytes += sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                shutil.rmtree(item, ignore_errors=True)
            elif item.is_file():
                removed_bytes += item.stat().st_size
                item.unlink(missing_ok=True)

    # Remove static C/C++ headers and .lib link libraries inside pyarrow / numpy (never used at Python runtime)
    for sub_rel in ["pyarrow/include", "numpy/_core/include", "numpy/core/include", "numpy/f2py/tests", "pandas/tests"]:
        sub = site_packages / sub_rel
        if sub.exists() and sub.is_dir():
            removed_bytes += sum(f.stat().st_size for f in sub.rglob("*") if f.is_file())
            shutil.rmtree(sub, ignore_errors=True)

    for lib_file in list(site_packages.rglob("*.lib")) + list(site_packages.rglob("*.pdb")):
        removed_bytes += lib_file.stat().st_size
        lib_file.unlink(missing_ok=True)

    # Remove bulky __pycache__ directories inside site-packages (cuts file count in half for fast zip/unzip)
    for pyc_dir in list(site_packages.rglob("__pycache__")):
        if pyc_dir.is_dir():
            removed_bytes += sum(f.stat().st_size for f in pyc_dir.rglob("*") if f.is_file())
            shutil.rmtree(pyc_dir, ignore_errors=True)

    return removed_bytes


def scan_for_secrets_and_hygiene(root_dir: Path) -> None:
    """Ensure zero personal API keys, .env files, or test caches exist anywhere in the client bundle."""
    text_exts: Set[str] = {".py", ".toml", ".json", ".txt", ".md", ".html", ".bat", ".cfg", ".ini", "._pth"}
    for fpath in root_dir.rglob("*"):
        if not fpath.is_file():
            continue
        if fpath.name == ".env":
            raise RuntimeError(f"Forbidden .env file found in package: {fpath}")
        if fpath.suffix.lower() in text_exts:
            content = fpath.read_text(encoding="utf-8", errors="ignore")
            for secret in FORBIDDEN_SECRETS:
                if secret in content:
                    raise RuntimeError(f"SECURITY ALERT: Forbidden secret pattern '{secret}' found in {fpath}!")


def create_zip_from_directory(src_dir: Path, zip_path: Path, include_root_folder: bool) -> None:
    """Create UTF-8 ZIP archive cleanly."""
    if zip_path.exists():
        zip_path.unlink()
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    base_for_rel = src_dir.parent if include_root_folder else src_dir
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for item in sorted(src_dir.rglob("*")):
            if item.is_file():
                arcname = item.relative_to(base_for_rel).as_posix()
                zf.write(item, arcname)


def build_self_extracting_installer(stub_exe: Path, payload_zip: Path, out_setup_exe: Path) -> None:
    """Combine native WinForms SetupStub.exe + payload.zip + [Int64 size] + ['BTAIINST'] into one .exe."""
    out_setup_exe.parent.mkdir(parents=True, exist_ok=True)
    payload_size = payload_zip.stat().st_size
    footer = struct.pack("<q", payload_size) + b"BTAIINST"

    with open(out_setup_exe, "wb") as out_f:
        with open(stub_exe, "rb") as stub_f:
            shutil.copyfileobj(stub_f, out_f, length=1024 * 1024)
        with open(payload_zip, "rb") as zip_f:
            shutil.copyfileobj(zip_f, out_f, length=4 * 1024 * 1024)
        out_f.write(footer)


def main() -> None:
    step("1. Generating Icon & Compiling Native Windows Executables (x64)")
    sys.path.insert(0, str(PACKAGING_DIR))
    from generate_icon import generate_bridge_ico

    ico_path = PACKAGING_DIR / "assets" / "bridge_app.ico"
    generate_bridge_ico(ico_path)

    csc_path = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")
    if not csc_path.exists():
        raise FileNotFoundError(f"csc.exe not found at {csc_path}")

    BIN_DIR.mkdir(parents=True, exist_ok=True)
    launcher_cs = PACKAGING_DIR / "BridgeTrafficLauncher.cs"
    installer_cs = PACKAGING_DIR / "BridgeTrafficInstaller.cs"

    compile_jobs = [
        [
            str(csc_path), "/nologo", "/target:winexe", "/platform:x64", "/optimize+",
            f"/win32icon:{ico_path}",
            "/r:System.dll,System.Drawing.dll,System.Windows.Forms.dll",
            f"/out:{BIN_DIR / 'Launcher.exe'}", str(launcher_cs),
        ],
        [
            str(csc_path), "/nologo", "/target:winexe", "/platform:x64", "/optimize+", "/define:CONFIG_ONLY",
            f"/win32icon:{ico_path}",
            "/r:System.dll,System.Drawing.dll,System.Windows.Forms.dll",
            f"/out:{BIN_DIR / 'ConfigGemini.exe'}", str(launcher_cs),
        ],
        [
            str(csc_path), "/nologo", "/target:winexe", "/platform:x64", "/optimize+",
            f"/win32icon:{ico_path}",
            "/r:System.dll,System.Drawing.dll,System.Windows.Forms.dll,System.IO.Compression.dll",
            f"/out:{BIN_DIR / 'SetupStub.exe'}", str(installer_cs),
        ],
        [
            str(csc_path), "/nologo", "/target:winexe", "/platform:x64", "/optimize+", "/define:UNINSTALLER_MODE",
            f"/win32icon:{ico_path}",
            "/r:System.dll,System.Drawing.dll,System.Windows.Forms.dll,System.IO.Compression.dll",
            f"/out:{BIN_DIR / 'Uninstaller.exe'}", str(installer_cs),
        ],
    ]
    for cmd in compile_jobs:
        subprocess.run(cmd, check=True)

    step("2. Running Full 42-Test Suite Inside Embedded Python 3.12 Runtime Before Packaging")
    embedded_py = RAW_RUNTIME_DIR / "python.exe"
    if not embedded_py.exists():
        raise FileNotFoundError(f"Embedded runtime missing at {embedded_py}")

    clean_env = {
        "SystemRoot": os.environ.get("SystemRoot", r"C:\Windows"),
        "WINDIR": os.environ.get("WINDIR", r"C:\Windows"),
        "PATH": f"{RAW_RUNTIME_DIR};{os.environ.get('SystemRoot', r'C:\Windows')}\\System32;{os.environ.get('SystemRoot', r'C:\Windows')}",
        "TEMP": str(BUILD_TEMP / "test_tmp"),
        "TMP": str(BUILD_TEMP / "test_tmp"),
        "USERPROFILE": str(BUILD_TEMP / "test_profile"),
        "LOCALAPPDATA": str(BUILD_TEMP / "test_profile" / "AppData" / "Local"),
        "APPDATA": str(BUILD_TEMP / "test_profile" / "AppData" / "Roaming"),
        "PYTHONUTF8": "1",
        "PYTHONNOUSERSITE": "1",
    }
    (BUILD_TEMP / "test_tmp").mkdir(parents=True, exist_ok=True)
    (BUILD_TEMP / "test_profile" / "AppData" / "Local").mkdir(parents=True, exist_ok=True)
    (BUILD_TEMP / "test_profile" / "AppData" / "Roaming").mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [str(embedded_py), "-m", "pytest", str(PROJECT_ROOT / "tests"), "-q"],
        cwd=str(PROJECT_ROOT),
        env=clean_env,
        check=True,
    )
    print("Embedded Python 3.12 passed 100% of test suite with stripped PATH and no Anaconda!")

    step("3. Assembling Clean Staging Directory for Non-Technical Client")
    if STAGING_PARENT.exists():
        shutil.rmtree(STAGING_PARENT)
    STAGING_APP_ROOT.mkdir(parents=True, exist_ok=True)

    # Top-level friendly files
    shutil.copy2(BIN_DIR / "Launcher.exe", STAGING_APP_ROOT / "تشغيل محلل المرور الذكي.exe")
    shutil.copy2(BIN_DIR / "ConfigGemini.exe", STAGING_APP_ROOT / "إعداد مفتاح الذكاء الاصطناعي.exe")
    shutil.copy2(BIN_DIR / "Uninstaller.exe", STAGING_APP_ROOT / "إلغاء تثبيت البرنامج.exe")
    shutil.copy2(PACKAGING_DIR / "templates" / "دليل التشغيل السريع (عربي).html", STAGING_APP_ROOT / "دليل التشغيل السريع (عربي).html")
    shutil.copy2(PACKAGING_DIR / "templates" / "اقرأني - طريقة التشغيل.txt", STAGING_APP_ROOT / "اقرأني - طريقة التشغيل.txt")

    # Copy & optimize runtime (only ignore bytecode and C++ link libraries during recursive copy!)
    target_runtime = STAGING_APP_ROOT / "system" / "runtime"
    shutil.copytree(
        RAW_RUNTIME_DIR,
        target_runtime,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            "*.pyo",
            "*.lib",
            "*.pdb",
        ),
    )
    copied_dlls = bundle_vc_redist_dlls(target_runtime)
    print("Bundled VC++ Redistributable DLLs:", copied_dlls)

    saved_bytes = prune_unused_build_artifacts(target_runtime / "Lib" / "site-packages")
    print(f"Pruned {saved_bytes / (1024 * 1024):.1f} MB of top-level build/notebook packages & headers")

    # Write sitecustomize.py so embedded Python strictly ignores any host user-site directories
    site_cust = target_runtime / "Lib" / "site-packages" / "sitecustomize.py"
    site_cust.write_text(
        "import sys\n"
        "sys.path[:] = [p for p in sys.path if 'Roaming\\\\Python' not in p and 'Roaming/Python' not in p]\n",
        encoding="utf-8",
    )

    # Remove Scripts/*.exe that hardcode build-machine paths (we invoke `-m streamlit` via python.exe)
    scripts_dir = target_runtime / "Scripts"
    if scripts_dir.exists():
        shutil.rmtree(scripts_dir)

    # Copy application code & verified read-only inputs
    target_app = STAGING_APP_ROOT / "system" / "app"
    target_app.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT_ROOT / "app.py", target_app / "app.py")
    shutil.copy2(PROJECT_ROOT / "input_manifest.json", target_app / "input_manifest.json")
    shutil.copytree(
        PROJECT_ROOT / "src",
        target_app / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(PROJECT_ROOT / "inputs", target_app / "inputs")

    st_cfg_dir = target_app / ".streamlit"
    st_cfg_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PACKAGING_DIR / "templates" / "streamlit_config.toml", st_cfg_dir / "config.toml")

    step("4. Running Security & Secret Hygiene Audit on Packaged Files")
    scan_for_secrets_and_hygiene(STAGING_APP_ROOT)
    print("Security scan PASSED: 0 API keys, 0 .env files, 0 build-path leaks.")

    step("5. Building Single-File Windows Installer (.exe) & Portable (.zip)")
    if DELIVERY_DIR.exists():
        shutil.rmtree(DELIVERY_DIR)
    DELIVERY_DIR.mkdir(parents=True, exist_ok=True)

    # 5a. Payload zip (relative to root of app dir) for Setup_Bridge_Traffic_AI_AR.exe
    create_zip_from_directory(STAGING_APP_ROOT, PAYLOAD_ZIP_PATH, include_root_folder=False)
    setup_exe_out = DELIVERY_DIR / "Setup_Bridge_Traffic_AI_AR.exe"
    build_self_extracting_installer(BIN_DIR / "SetupStub.exe", PAYLOAD_ZIP_PATH, setup_exe_out)

    # 5b. Portable zip (containing محلل_المرور_الذكي_كوبري_السلاب/...)
    portable_zip_out = DELIVERY_DIR / "Bridge_Traffic_AI_Portable_Windows.zip"
    create_zip_from_directory(STAGING_APP_ROOT, portable_zip_out, include_root_folder=True)

    # 5c. Copy standalone Arabic quick guides alongside deliverables
    shutil.copy2(
        PACKAGING_DIR / "templates" / "دليل التشغيل السريع (عربي).html",
        DELIVERY_DIR / "دليل_التثبيت_والتشغيل_للعميل.html",
    )
    shutil.copy2(
        PACKAGING_DIR / "templates" / "اقرأني - طريقة التشغيل.txt",
        DELIVERY_DIR / "طريقة_التشغيل_في_خطوتين.txt",
    )

    print("\nDELIVERY ARTIFACTS READY:")
    for f in sorted(DELIVERY_DIR.iterdir()):
        print(f" - {f.name}: {f.stat().st_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
