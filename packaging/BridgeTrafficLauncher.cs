using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Windows.Forms;

namespace BridgeTrafficAI
{
    internal static class NativeJobObject
    {
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
        private static extern IntPtr CreateJobObject(IntPtr lpJobAttributes, string lpName);

        [DllImport("kernel32.dll")]
        private static extern bool SetInformationJobObject(IntPtr hJob, int infoType, IntPtr lpJobObjectInfo, uint cbJobObjectInfoLength);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

        [StructLayout(LayoutKind.Sequential)]
        private struct JOBOBJECT_BASIC_LIMIT_INFORMATION
        {
            public long PerProcessUserTimeLimit;
            public long PerJobUserTimeLimit;
            public uint LimitFlags;
            public UIntPtr MinimumWorkingSetSize;
            public UIntPtr MaximumWorkingSetSize;
            public uint ActiveProcessLimit;
            public UIntPtr Affinity;
            public uint PriorityClass;
            public uint SchedulingClass;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct IO_COUNTERS
        {
            public ulong ReadOperationCount;
            public ulong WriteOperationCount;
            public ulong OtherOperationCount;
            public ulong ReadTransferCount;
            public ulong WriteTransferCount;
            public ulong OtherTransferCount;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        {
            public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
            public IO_COUNTERS IoInfo;
            public UIntPtr ProcessMemoryLimit;
            public UIntPtr JobMemoryLimit;
            public UIntPtr PeakProcessMemoryUsed;
            public UIntPtr PeakJobMemoryUsed;
        }

        private const int JobObjectExtendedLimitInformation = 9;
        private const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;

        private static IntPtr _jobHandle = IntPtr.Zero;

        public static void AttachKillOnClose(Process childProcess)
        {
            try
            {
                if (_jobHandle == IntPtr.Zero)
                {
                    _jobHandle = CreateJobObject(IntPtr.Zero, null);
                    if (_jobHandle != IntPtr.Zero)
                    {
                        JOBOBJECT_EXTENDED_LIMIT_INFORMATION info = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
                        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                        int length = Marshal.SizeOf(typeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION));
                        IntPtr ptr = Marshal.AllocHGlobal(length);
                        try
                        {
                            Marshal.StructureToPtr(info, ptr, false);
                            SetInformationJobObject(_jobHandle, JobObjectExtendedLimitInformation, ptr, (uint)length);
                        }
                        finally
                        {
                            Marshal.FreeHGlobal(ptr);
                        }
                    }
                }
                if (_jobHandle != IntPtr.Zero && childProcess != null && !childProcess.HasExited)
                {
                    AssignProcessToJobObject(_jobHandle, childProcess.Handle);
                }
            }
            catch
            {
                // Best-effort kernel cleanup attachment; explicit Kill() also runs on FormClosing.
            }
        }
    }

    internal static class AppConfigPaths
    {
        public static string GetUserDataDir()
        {
            string overrideDir = Environment.GetEnvironmentVariable("BRIDGE_TRAFFIC_USER_DATA_DIR");
            if (!string.IsNullOrEmpty(overrideDir) && overrideDir.Trim().Length > 0)
            {
                return Path.Combine(overrideDir.Trim(), "BridgeTrafficAI");
            }
            string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            if (string.IsNullOrEmpty(localAppData))
            {
                localAppData = Path.GetTempPath();
            }
            return Path.Combine(localAppData, "BridgeTrafficAI");
        }

        public static string GetGeminiConfigFile()
        {
            return Path.Combine(GetUserDataDir(), "user_gemini_config.json");
        }

        public static string GetLogsDir()
        {
            string dir = Path.Combine(GetUserDataDir(), "logs");
            if (!Directory.Exists(dir))
            {
                Directory.CreateDirectory(dir);
            }
            return dir;
        }

        public static string GetActiveServerFile()
        {
            string dir = GetUserDataDir();
            if (!Directory.Exists(dir))
            {
                Directory.CreateDirectory(dir);
            }
            return Path.Combine(dir, "active_server_port.txt");
        }

        public static string LoadSavedGeminiKey()
        {
            try
            {
                string path = GetGeminiConfigFile();
                if (!File.Exists(path)) return "";
                string raw = File.ReadAllText(path, Encoding.UTF8);
                int idx = raw.IndexOf("\"gemini_api_key\"");
                if (idx < 0) return "";
                int colon = raw.IndexOf(':', idx);
                if (colon < 0) return "";
                int firstQuote = raw.IndexOf('"', colon + 1);
                if (firstQuote < 0) return "";
                StringBuilder sb = new StringBuilder();
                bool escaped = false;
                for (int i = firstQuote + 1; i < raw.Length; i++)
                {
                    char c = raw[i];
                    if (escaped)
                    {
                        sb.Append(c);
                        escaped = false;
                    }
                    else if (c == '\\')
                    {
                        escaped = true;
                    }
                    else if (c == '"')
                    {
                        break;
                    }
                    else
                    {
                        sb.Append(c);
                    }
                }
                return sb.ToString().Trim();
            }
            catch
            {
                return "";
            }
        }

        public static void SaveGeminiKey(string apiKey)
        {
            string dir = GetUserDataDir();
            if (!Directory.Exists(dir))
            {
                Directory.CreateDirectory(dir);
            }
            string escaped = (apiKey ?? "").Trim().Replace("\\", "\\\\").Replace("\"", "\\\"");
            string json = "{\r\n  \"gemini_api_key\": \"" + escaped + "\"\r\n}\r\n";
            File.WriteAllText(GetGeminiConfigFile(), json, new UTF8Encoding(false));
        }

        public static void ClearGeminiKey()
        {
            string path = GetGeminiConfigFile();
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    internal sealed class GeminiConfigDialog : Form
    {
        private TextBox _keyBox;
        private CheckBox _showCheck;
        private Label _statusLabel;

        public GeminiConfigDialog()
        {
            this.Text = "إعداد مفتاح الذكاء الاصطناعي Gemini (اختياري)";
            this.Size = new Size(560, 355);
            this.FormBorderStyle = FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.StartPosition = FormStartPosition.CenterScreen;
            this.RightToLeft = RightToLeft.Yes;
            this.RightToLeftLayout = true;
            this.BackColor = Color.FromArgb(248, 250, 252);
            this.Font = new Font("Segoe UI", 9.5f, FontStyle.Regular);

            try
            {
                this.Icon = Icon.ExtractAssociatedIcon(Application.ExecutablePath);
            }
            catch { }

            Panel header = new Panel();
            header.Dock = DockStyle.Top;
            header.Height = 64;
            header.BackColor = Color.FromArgb(15, 118, 110);

            Label title = new Label();
            title.Text = "إعداد مفتاح الذكاء الاصطناعي Google Gemini (اختياري)";
            title.ForeColor = Color.White;
            title.Font = new Font("Segoe UI", 12f, FontStyle.Bold);
            title.AutoSize = false;
            title.Dock = DockStyle.Fill;
            title.TextAlign = ContentAlignment.MiddleRight;
            title.Padding = new Padding(16, 0, 16, 0);
            header.Controls.Add(title);
            this.Controls.Add(header);

            Label info = new Label();
            info.Text = "ملاحظة مطمئنة: يعمل البرنامج بكامل دقّته الحسابية وجميع لوحاته التفاعلية وميزة «اسأل بياناتك» تلقائيًا بدون الحاجة لأي مفتاح.\r\nإذا توفر لديك مفتاح Google Gemini ورغبت في تفعيل الصياغة اللغوية الحية، الصقه أدناه واضغط «حفظ المفتاح».";
            info.Location = new Point(22, 78);
            info.Size = new Size(505, 62);
            info.ForeColor = Color.FromArgb(51, 65, 85);
            this.Controls.Add(info);

            Label promptLbl = new Label();
            promptLbl.Text = "مفتاح Gemini API:";
            promptLbl.Font = new Font("Segoe UI", 9.5f, FontStyle.Bold);
            promptLbl.Location = new Point(22, 146);
            promptLbl.Size = new Size(200, 24);
            this.Controls.Add(promptLbl);

            _keyBox = new TextBox();
            _keyBox.Location = new Point(125, 174);
            _keyBox.Size = new Size(402, 28);
            _keyBox.RightToLeft = RightToLeft.No;
            _keyBox.UseSystemPasswordChar = true;
            _keyBox.Font = new Font("Consolas", 10f);
            _keyBox.Text = AppConfigPaths.LoadSavedGeminiKey();
            this.Controls.Add(_keyBox);

            Button pasteBtn = new Button();
            pasteBtn.Text = "لصق";
            pasteBtn.Location = new Point(22, 172);
            pasteBtn.Size = new Size(95, 30);
            pasteBtn.BackColor = Color.White;
            pasteBtn.FlatStyle = FlatStyle.System;
            pasteBtn.Click += delegate
            {
                if (Clipboard.ContainsText())
                {
                    _keyBox.Text = Clipboard.GetText().Trim();
                }
            };
            this.Controls.Add(pasteBtn);

            _showCheck = new CheckBox();
            _showCheck.Text = "إظهار أحرف المفتاح للتأكد منه";
            _showCheck.Location = new Point(22, 208);
            _showCheck.Size = new Size(260, 24);
            _showCheck.CheckedChanged += delegate
            {
                _keyBox.UseSystemPasswordChar = !_showCheck.Checked;
            };
            this.Controls.Add(_showCheck);

            _statusLabel = new Label();
            _statusLabel.Location = new Point(22, 236);
            _statusLabel.Size = new Size(505, 24);
            _statusLabel.Font = new Font("Segoe UI", 9f, FontStyle.Bold);
            UpdateStatusText();
            this.Controls.Add(_statusLabel);

            Button saveBtn = new Button();
            saveBtn.Text = "حفظ وتفعيل المفتاح";
            saveBtn.Location = new Point(22, 268);
            saveBtn.Size = new Size(175, 36);
            saveBtn.BackColor = Color.FromArgb(15, 118, 110);
            saveBtn.ForeColor = Color.White;
            saveBtn.FlatStyle = FlatStyle.Flat;
            saveBtn.Font = new Font("Segoe UI", 9.5f, FontStyle.Bold);
            saveBtn.Click += OnSaveClick;
            this.Controls.Add(saveBtn);

            Button clearBtn = new Button();
            clearBtn.Text = "مسح المفتاح المحفوظ";
            clearBtn.Location = new Point(207, 268);
            clearBtn.Size = new Size(165, 36);
            clearBtn.BackColor = Color.White;
            clearBtn.FlatStyle = FlatStyle.System;
            clearBtn.Click += OnClearClick;
            this.Controls.Add(clearBtn);

            Button closeBtn = new Button();
            closeBtn.Text = "إغلاق";
            closeBtn.Location = new Point(382, 268);
            closeBtn.Size = new Size(145, 36);
            closeBtn.FlatStyle = FlatStyle.System;
            closeBtn.Click += delegate { this.Close(); };
            this.Controls.Add(closeBtn);
        }

        private void UpdateStatusText()
        {
            string saved = AppConfigPaths.LoadSavedGeminiKey();
            if (!string.IsNullOrEmpty(saved))
            {
                _statusLabel.ForeColor = Color.FromArgb(5, 150, 105);
                _statusLabel.Text = "الحالة الحالية: يوجد مفتاح محفوظ ومفعّل على هذا الجهاز.";
            }
            else
            {
                _statusLabel.ForeColor = Color.FromArgb(71, 85, 105);
                _statusLabel.Text = "الحالة الحالية: يعمل البرنامج بالوضع المدمج الموثق (بدون مفتاح خارجي).";
            }
        }

        private void OnSaveClick(object sender, EventArgs e)
        {
            string val = _keyBox.Text.Trim();
            if (string.IsNullOrEmpty(val))
            {
                MessageBox.Show(
                    "يرجى لصق مفتاح Gemini في المربع المخصص أولاً، أو اضغط «إغلاق» للاستمرار بالوضع المدمج بدون مفتاح.",
                    "تنبيه",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Information,
                    MessageBoxDefaultButton.Button1,
                    MessageBoxOptions.RightAlign | MessageBoxOptions.RtlReading
                );
                return;
            }
            try
            {
                AppConfigPaths.SaveGeminiKey(val);
                UpdateStatusText();
                MessageBox.Show(
                    "تم حفظ مفتاح الذكاء الاصطناعي على جهازك بنجاح!\r\nسيتم استخدامه تلقائيًا في البرنامج دون الحاجة لإدخاله مرة أخرى.",
                    "تم الحفظ بنجاح",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Information,
                    MessageBoxDefaultButton.Button1,
                    MessageBoxOptions.RightAlign | MessageBoxOptions.RtlReading
                );
                this.DialogResult = DialogResult.OK;
                this.Close();
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    "تعذر حفظ المفتاح على القرص:\r\n" + ex.Message,
                    "خطأ في الحفظ",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error,
                    MessageBoxDefaultButton.Button1,
                    MessageBoxOptions.RightAlign | MessageBoxOptions.RtlReading
                );
            }
        }

        private void OnClearClick(object sender, EventArgs e)
        {
            try
            {
                AppConfigPaths.ClearGeminiKey();
                _keyBox.Text = "";
                UpdateStatusText();
                MessageBox.Show(
                    "تم مسح المفتاح المحفوظ، وسيعمل البرنامج بمحركه التحليلي المدمج الموثق.",
                    "تم المسح",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Information,
                    MessageBoxDefaultButton.Button1,
                    MessageBoxOptions.RightAlign | MessageBoxOptions.RtlReading
                );
            }
            catch { }
        }
    }

    internal sealed class LauncherMainForm : Form
    {
        private readonly string _baseDir;
        private readonly bool _verifyOnlyMode;
        private Process _serverProcess;
        private int _selectedPort = 8501;
        private string _serverUrl = "";
        private string _logFilePath = "";
        private StringBuilder _stderrBuffer = new StringBuilder();
        private System.Windows.Forms.Timer _pollTimer;
        private int _pollTicks = 0;
        private bool _browserOpened = false;

        private Label _statusHeading;
        private Label _statusDetail;
        private ProgressBar _progressBar;
        private Button _openBrowserBtn;
        private Button _geminiBtn;
        private Button _guideBtn;
        private Button _stopBtn;

        public int ExitVerificationCode { get; private set; }

        public LauncherMainForm(string baseDir, bool verifyOnlyMode)
        {
            _baseDir = baseDir;
            _verifyOnlyMode = verifyOnlyMode;
            ExitVerificationCode = 0;

            this.Text = "محلل بيانات المرور والمشاة الذكي — كوبري السلاب";
            this.Size = new Size(600, 365);
            this.FormBorderStyle = FormBorderStyle.FixedSingle;
            this.MaximizeBox = false;
            this.StartPosition = FormStartPosition.CenterScreen;
            this.RightToLeft = RightToLeft.Yes;
            this.RightToLeftLayout = true;
            this.BackColor = Color.FromArgb(248, 250, 252);
            this.Font = new Font("Segoe UI", 9.5f, FontStyle.Regular);

            try
            {
                this.Icon = Icon.ExtractAssociatedIcon(Application.ExecutablePath);
            }
            catch { }

            BuildUi();
            this.Shown += OnFormShown;
            this.FormClosing += OnFormClosing;
        }

        private void BuildUi()
        {
            Panel banner = new Panel();
            banner.Dock = DockStyle.Top;
            banner.Height = 76;
            banner.BackColor = Color.FromArgb(15, 118, 110);

            Label mainTitle = new Label();
            mainTitle.Text = "محلل بيانات المرور والمشاة الذكي — كوبري السلاب";
            mainTitle.ForeColor = Color.White;
            mainTitle.Font = new Font("Segoe UI", 13f, FontStyle.Bold);
            mainTitle.Location = new Point(20, 12);
            mainTitle.Size = new Size(545, 28);
            banner.Controls.Add(mainTitle);

            Label subTitle = new Label();
            subTitle.Text = "لوحة المؤشرات التفاعلية والمحلل الذكي ثنائي اللغة (العربية / الإنجليزية)";
            subTitle.ForeColor = Color.FromArgb(204, 251, 241);
            subTitle.Font = new Font("Segoe UI", 9.5f, FontStyle.Regular);
            subTitle.Location = new Point(20, 42);
            subTitle.Size = new Size(545, 22);
            banner.Controls.Add(subTitle);

            this.Controls.Add(banner);

            Panel card = new Panel();
            card.Location = new Point(20, 92);
            card.Size = new Size(545, 130);
            card.BackColor = Color.White;
            card.BorderStyle = BorderStyle.FixedSingle;

            _statusHeading = new Label();
            _statusHeading.Text = "جاري تجهيز البرنامج وفتح لوحة البيانات...";
            _statusHeading.Font = new Font("Segoe UI", 11f, FontStyle.Bold);
            _statusHeading.ForeColor = Color.FromArgb(15, 23, 42);
            _statusHeading.Location = new Point(16, 14);
            _statusHeading.Size = new Size(510, 28);
            card.Controls.Add(_statusHeading);

            _statusDetail = new Label();
            _statusDetail.Text = "يتم الآن تشغيل المحرك التحليلي المحلي؛ ستفتح الواجهة تلقائيًا في متصفح الإنترنت خلال ثوانٍ قليلة.";
            _statusDetail.ForeColor = Color.FromArgb(71, 85, 105);
            _statusDetail.Location = new Point(16, 46);
            _statusDetail.Size = new Size(510, 42);
            card.Controls.Add(_statusDetail);

            _progressBar = new ProgressBar();
            _progressBar.Style = ProgressBarStyle.Marquee;
            _progressBar.MarqueeAnimationSpeed = 25;
            _progressBar.Location = new Point(16, 94);
            _progressBar.Size = new Size(510, 20);
            card.Controls.Add(_progressBar);

            this.Controls.Add(card);

            _openBrowserBtn = new Button();
            _openBrowserBtn.Text = "فتح البرنامج في المتصفح";
            _openBrowserBtn.Location = new Point(20, 238);
            _openBrowserBtn.Size = new Size(265, 38);
            _openBrowserBtn.BackColor = Color.FromArgb(15, 118, 110);
            _openBrowserBtn.ForeColor = Color.White;
            _openBrowserBtn.FlatStyle = FlatStyle.Flat;
            _openBrowserBtn.Font = new Font("Segoe UI", 10f, FontStyle.Bold);
            _openBrowserBtn.Enabled = false;
            _openBrowserBtn.Click += delegate { OpenBrowserSafe(_serverUrl); };
            this.Controls.Add(_openBrowserBtn);

            _geminiBtn = new Button();
            _geminiBtn.Text = "إعداد مفتاح Gemini (اختياري)";
            _geminiBtn.Location = new Point(300, 238);
            _geminiBtn.Size = new Size(265, 38);
            _geminiBtn.FlatStyle = FlatStyle.System;
            _geminiBtn.Click += delegate
            {
                using (GeminiConfigDialog dlg = new GeminiConfigDialog())
                {
                    dlg.ShowDialog(this);
                }
            };
            this.Controls.Add(_geminiBtn);

            _guideBtn = new Button();
            _guideBtn.Text = "دليل الاستخدام السريع";
            _guideBtn.Location = new Point(20, 284);
            _guideBtn.Size = new Size(265, 34);
            _guideBtn.FlatStyle = FlatStyle.System;
            _guideBtn.Click += OnOpenGuideClick;
            this.Controls.Add(_guideBtn);

            _stopBtn = new Button();
            _stopBtn.Text = "إيقاف البرنامج وإغلاقه";
            _stopBtn.Location = new Point(300, 284);
            _stopBtn.Size = new Size(265, 34);
            _stopBtn.FlatStyle = FlatStyle.System;
            _stopBtn.Click += delegate { this.Close(); };
            this.Controls.Add(_stopBtn);
        }

        private void OnOpenGuideClick(object sender, EventArgs e)
        {
            string htmlGuide = Path.Combine(_baseDir, "دليل التشغيل السريع (عربي).html");
            string txtGuide = Path.Combine(_baseDir, "اقرأني - طريقة التشغيل.txt");
            if (File.Exists(htmlGuide))
            {
                OpenBrowserSafe(htmlGuide);
            }
            else if (File.Exists(txtGuide))
            {
                OpenBrowserSafe(txtGuide);
            }
            else
            {
                MessageBox.Show(
                    "البرنامج يعمل مباشرة في متصفح الإنترنت لديك.\r\nاضغط على «فتح البرنامج في المتصفح» لاستعراض لوحات البيانات أو طرح أي سؤال بالعربية في تبويب «اسأل بياناتك».",
                    "دليل الاستخدام السريع",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Information,
                    MessageBoxDefaultButton.Button1,
                    MessageBoxOptions.RightAlign | MessageBoxOptions.RtlReading
                );
            }
        }

        private void OnFormShown(object sender, EventArgs e)
        {
            Application.DoEvents();
            StartBackendWorkflow();
        }

        private void StartBackendWorkflow()
        {
            try
            {
                _logFilePath = Path.Combine(AppConfigPaths.GetLogsDir(), "startup_" + DateTime.Now.ToString("yyyyMMdd_HHmmss_fff") + ".log");
            }
            catch { }

            string runtimeDir = Path.Combine(_baseDir, Path.Combine("system", "runtime"));
            string appDir = Path.Combine(_baseDir, Path.Combine("system", "app"));
            string pythonwExe = Path.Combine(runtimeDir, "pythonw.exe");
            string pythonExe = Path.Combine(runtimeDir, "python.exe");
            string appPy = Path.Combine(appDir, "app.py");
            string surveyXlsx = Path.Combine(appDir, Path.Combine("inputs", "Traffic_Survey_Combined.xlsx"));

            // 1. Check if launched directly inside unextracted Windows Explorer ZIP preview
            string lowerBase = _baseDir.ToLowerInvariant();
            bool looksLikeZipTemp = lowerBase.Contains(".zip") || lowerBase.Contains("\\temp\\temp") || lowerBase.Contains("\\inetcache\\");
            if (!File.Exists(pythonExe) || !File.Exists(appPy))
            {
                string msg;
                if (looksLikeZipTemp)
                {
                    msg = "تنبيه هام قبل التشغيل:\r\n\r\n"
                        + "يبدو أنك قمت بفتح البرنامج من داخل الملف المضغوط (ZIP) مباشرةً دون فك الضغط.\r\n\r\n"
                        + "طريقة التشغيل الصحيحة في خطوة واحدة:\r\n"
                        + "1) اضغط بزر الفأرة الأيمن على ملف (ZIP) واختر «استخراج الكل...» (Extract All).\r\n"
                        + "2) افتح المجلد الجديد المستخرج واضغط مرتين على أيقونة «تشغيل محلل المرور الذكي».";
                }
                else
                {
                    msg = "تعذر العثور على ملفات المحرك المدمج للبرنامج داخل المجلد:\r\n"
                        + _baseDir + "\r\n\r\n"
                        + "يرجى التأكد من فك ضغط جميع ملفات الحزمة كاملةً دون حذف مجلد (system)، أو إعادة تشغيل ملف التثبيت (Setup).";
                }
                ShowFriendlyFailureAndExit("يرجى استخراج ملفات البرنامج أولاً", msg, 2);
                return;
            }

            if (!File.Exists(surveyXlsx))
            {
                string msg = "لم يتم العثور على ملفات البيانات الأساسية للدراسة داخل مجلد (system\\app\\inputs).\r\n"
                    + "يرجى التأكد من اكتمال استخراج جميع الملفات أو إعادة تثبيت الحزمة.";
                ShowFriendlyFailureAndExit("ملفات البيانات غير مكتملة", msg, 3);
                return;
            }

            // 2. Check if an instance is already running and healthy
            int existingPort = CheckExistingHealthyServer();
            if (existingPort > 0)
            {
                _selectedPort = existingPort;
                _serverUrl = "http://127.0.0.1:" + _selectedPort.ToString();
                if (!_verifyOnlyMode)
                {
                    OpenBrowserSafe(_serverUrl);
                }
                this.Close();
                return;
            }

            // 3. Pick a free local port (preferring 8501)
            _selectedPort = FindAvailablePort(8501, 8599);
            if (_selectedPort <= 0)
            {
                ShowFriendlyFailureAndExit(
                    "تعذر حجز منفذ اتصال محلي",
                    "لم يتمكن البرنامج من فتح منفذ اتصال محلي على هذا الجهاز (المنافذ 8501-8599 مشغولة أو محجوبة بواسطة برنامج الحماية).\r\nيرجى إغلاق أي نسخ قديمة أو السماح للبرنامج في جدار الحماية ثم المحاولة مجددًا.",
                    4
                );
                return;
            }
            _serverUrl = "http://127.0.0.1:" + _selectedPort.ToString();

            // 4. Ensure Streamlit onboarding email prompt is disabled
            EnsureStreamlitCredentialsConfigured();

            // 5. Launch bundled Python silently (using python.exe with CreateNoWindow=true so stderr is reliably captured)
            try
            {
                ProcessStartInfo psi = new ProcessStartInfo();
                psi.FileName = pythonExe;
                psi.Arguments = "-m streamlit run \"app.py\" --server.port=" + _selectedPort.ToString()
                    + " --server.address=127.0.0.1 --server.headless=true --browser.gatherUsageStats=false --global.developmentMode=false";
                psi.WorkingDirectory = appDir;
                psi.UseShellExecute = false;
                psi.CreateNoWindow = true;
                psi.RedirectStandardOutput = true;
                psi.RedirectStandardError = true;
                psi.RedirectStandardInput = true;
                psi.StandardOutputEncoding = Encoding.UTF8;
                psi.StandardErrorEncoding = Encoding.UTF8;

                // Clean isolated environment so host Python/Conda variables can NEVER interfere
                psi.EnvironmentVariables.Remove("PYTHONHOME");
                psi.EnvironmentVariables.Remove("PYTHONPATH");
                psi.EnvironmentVariables.Remove("VIRTUAL_ENV");
                psi.EnvironmentVariables.Remove("CONDA_PREFIX");
                psi.EnvironmentVariables.Remove("CONDA_DEFAULT_ENV");

                string sysRoot = Environment.GetEnvironmentVariable("SystemRoot");
                if (string.IsNullOrEmpty(sysRoot)) sysRoot = @"C:\Windows";
                string cleanPath = runtimeDir + ";"
                    + Path.Combine(runtimeDir, "Scripts") + ";"
                    + Path.Combine(sysRoot, "System32") + ";"
                    + sysRoot;
                psi.EnvironmentVariables["PATH"] = cleanPath;
                psi.EnvironmentVariables["PYTHONUTF8"] = "1";
                psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
                psi.EnvironmentVariables["PYTHONNOUSERSITE"] = "1";
                psi.EnvironmentVariables["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false";

                string savedGemini = AppConfigPaths.LoadSavedGeminiKey();
                if (!string.IsNullOrEmpty(savedGemini))
                {
                    psi.EnvironmentVariables["GEMINI_API_KEY"] = savedGemini;
                }

                _serverProcess = new Process();
                _serverProcess.StartInfo = psi;
                _serverProcess.OutputDataReceived += OnProcessDataReceived;
                _serverProcess.ErrorDataReceived += OnProcessDataReceived;

                if (!_serverProcess.Start())
                {
                    ShowFriendlyFailureAndExit(
                        "تعذر بدء تشغيل المحرك التحليلي",
                        "لم يسمح نظام التشغيل ببدء عملية التحليل المدمجة. يرجى التحقق من عدم حجب البرنامج بواسطة مضاد الفيروسات.",
                        5
                    );
                    return;
                }

                NativeJobObject.AttachKillOnClose(_serverProcess);
                try { _serverProcess.StandardInput.Close(); } catch { }
                _serverProcess.BeginOutputReadLine();
                _serverProcess.BeginErrorReadLine();

                try
                {
                    File.WriteAllText(AppConfigPaths.GetActiveServerFile(), _selectedPort.ToString(), Encoding.ASCII);
                }
                catch { }

                _pollTimer = new System.Windows.Forms.Timer();
                _pollTimer.Interval = 400;
                _pollTimer.Tick += OnPollTimerTick;
                _pollTimer.Start();
            }
            catch (Exception ex)
            {
                WriteLogLine("Exception starting process: " + ex.ToString());
                ShowFriendlyFailureAndExit(
                    "حدث خطأ غير متوقع أثناء التشغيل",
                    "تعذر تشغيل محرك التحليل المدمج:\r\n" + ex.Message + "\r\n\r\nتم حفظ تفاصيل التشخيص في:\r\n" + _logFilePath,
                    6
                );
            }
        }

        private void OnProcessDataReceived(object sender, DataReceivedEventArgs e)
        {
            if (e == null || string.IsNullOrEmpty(e.Data)) return;
            lock (_stderrBuffer)
            {
                if (_stderrBuffer.Length < 16000)
                {
                    _stderrBuffer.AppendLine(e.Data);
                }
            }
            WriteLogLine(e.Data);
        }

        private void WriteLogLine(string line)
        {
            try
            {
                if (!string.IsNullOrEmpty(_logFilePath))
                {
                    File.AppendAllText(_logFilePath, line + Environment.NewLine, Encoding.UTF8);
                }
            }
            catch { }
        }

        private void OnPollTimerTick(object sender, EventArgs e)
        {
            _pollTicks++;

            // Check if server process crashed early
            if (_serverProcess != null && _serverProcess.HasExited)
            {
                _pollTimer.Stop();
                string details;
                lock (_stderrBuffer)
                {
                    details = _stderrBuffer.ToString();
                }
                WriteLogLine("Process exited early with code " + _serverProcess.ExitCode.ToString());
                string msg = "توقف محرك البرنامج أثناء بدء التشغيل (رمز التوقف: " + _serverProcess.ExitCode.ToString() + ").\r\n\r\n"
                    + "الأسباب المحتملة:\r\n"
                    + "• قيام برنامج الحماية (Antivirus) بحجب التشغيل المؤقت للملفات المستخرجة.\r\n"
                    + "• نقص في استخراج بعض ملفات الحزمة.\r\n\r\n"
                    + "تم حفظ تقرير التشخيص الفني في:\r\n" + _logFilePath;
                if (!string.IsNullOrEmpty(details))
                {
                    string preview = details.Length > 350 ? details.Substring(details.Length - 350) : details;
                    msg += "\r\n\r\nملخص فني:\r\n" + preview;
                }
                ShowFriendlyFailureAndExit("تعذر إكمال بدء تشغيل البرنامج", msg, 7);
                return;
            }

            if (IsServerHealthy(_selectedPort))
            {
                _pollTimer.Stop();
                OnServerReady();
                return;
            }

            // Timeout after ~60 seconds (150 ticks * 400ms)
            if (_pollTicks > 150)
            {
                _pollTimer.Stop();
                ShowFriendlyFailureAndExit(
                    "استغرق بدء التشغيل وقتًا أطول من المعتاد",
                    "لم يستجب الخادم المحلي للبرنامج خلال المهلة المحددة (60 ثانية).\r\n"
                    + "يرجى التأكد من عدم قيام جدار الحماية بحجب الاتصال المحلي (127.0.0.1) ثم أعد المحاولة.\r\n\r\n"
                    + "ملف التشخيص:\r\n" + _logFilePath,
                    8
                );
            }
        }

        private void OnServerReady()
        {
            _progressBar.Style = ProgressBarStyle.Continuous;
            _progressBar.Value = 100;
            _statusHeading.ForeColor = Color.FromArgb(5, 150, 105);
            _statusHeading.Text = "البرنامج يعمل الآن بنجاح — تم فتح الواجهة في المتصفح";
            _statusDetail.Text = "يمكنك الآن استخدام البرنامج في متصفح الإنترنت على العنوان (" + _serverUrl + ").\r\nاترك هذه النافذة مفتوحة (أو مصغّرة في شريط المهام) طوال فترة استخدامك للبرنامج.";
            _openBrowserBtn.Enabled = true;

            if (_verifyOnlyMode)
            {
                ExitVerificationCode = 0;
                WriteLogLine("VERIFY_READY_OK:" + _serverUrl);
                this.Close();
                return;
            }

            if (!_browserOpened)
            {
                _browserOpened = true;
                OpenBrowserSafe(_serverUrl);
            }
        }

        private static int CheckExistingHealthyServer()
        {
            try
            {
                string path = AppConfigPaths.GetActiveServerFile();
                if (!File.Exists(path)) return -1;
                string txt = File.ReadAllText(path, Encoding.ASCII).Trim();
                int port;
                if (int.TryParse(txt, out port) && port >= 1024 && port <= 65535)
                {
                    if (IsServerHealthy(port))
                    {
                        return port;
                    }
                }
            }
            catch { }
            return -1;
        }

        private static bool IsServerHealthy(int port)
        {
            try
            {
                HttpWebRequest req = (HttpWebRequest)WebRequest.Create("http://127.0.0.1:" + port.ToString() + "/_stcore/health");
                req.Method = "GET";
                req.Timeout = 800;
                req.ReadWriteTimeout = 800;
                req.Proxy = null;
                using (HttpWebResponse resp = (HttpWebResponse)req.GetResponse())
                {
                    if (resp.StatusCode == HttpStatusCode.OK)
                    {
                        using (StreamReader sr = new StreamReader(resp.GetResponseStream(), Encoding.UTF8))
                        {
                            string body = sr.ReadToEnd();
                            return body != null && body.Trim().Equals("ok", StringComparison.OrdinalIgnoreCase);
                        }
                    }
                }
            }
            catch { }
            return false;
        }

        private static int FindAvailablePort(int startPort, int endPort)
        {
            for (int p = startPort; p <= endPort; p++)
            {
                TcpListener listener = null;
                try
                {
                    listener = new TcpListener(IPAddress.Loopback, p);
                    listener.Start();
                    listener.Stop();
                    return p;
                }
                catch
                {
                    if (listener != null)
                    {
                        try { listener.Stop(); } catch { }
                    }
                }
            }
            return -1;
        }

        private static void EnsureStreamlitCredentialsConfigured()
        {
            try
            {
                string userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
                if (string.IsNullOrEmpty(userProfile)) return;
                string stDir = Path.Combine(userProfile, ".streamlit");
                if (!Directory.Exists(stDir))
                {
                    Directory.CreateDirectory(stDir);
                }
                string credFile = Path.Combine(stDir, "credentials.toml");
                if (!File.Exists(credFile))
                {
                    File.WriteAllText(credFile, "[general]\r\nemail = \"\"\r\n", new UTF8Encoding(false));
                }
            }
            catch { }
        }

        private static void OpenBrowserSafe(string target)
        {
            if (string.IsNullOrEmpty(target)) return;
            try
            {
                ProcessStartInfo psi = new ProcessStartInfo();
                psi.FileName = target;
                psi.UseShellExecute = true;
                Process.Start(psi);
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    "البرنامج يعمل الآن بنجاح، ولكن تعذر فتح المتصفح تلقائيًا.\r\nيرجى فتح أي متصفح إنترنت والدخول إلى العنوان التالي:\r\n\r\n" + target + "\r\n\r\n(" + ex.Message + ")",
                    "انسخ الرابط لفتح البرنامج",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Information,
                    MessageBoxDefaultButton.Button1,
                    MessageBoxOptions.RightAlign | MessageBoxOptions.RtlReading
                );
            }
        }

        private void ShowFriendlyFailureAndExit(string title, string arabicMessage, int code)
        {
            ExitVerificationCode = code;
            WriteLogLine("FAILURE [" + code.ToString() + "] " + title + ": " + arabicMessage);
            if (!_verifyOnlyMode)
            {
                MessageBox.Show(
                    this,
                    arabicMessage,
                    title,
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Warning,
                    MessageBoxDefaultButton.Button1,
                    MessageBoxOptions.RightAlign | MessageBoxOptions.RtlReading
                );
            }
            this.Close();
        }

        private void OnFormClosing(object sender, FormClosingEventArgs e)
        {
            if (_pollTimer != null)
            {
                _pollTimer.Stop();
            }
            if (_serverProcess != null)
            {
                try
                {
                    if (!_serverProcess.HasExited)
                    {
                        _serverProcess.Kill();
                        _serverProcess.WaitForExit(2000);
                    }
                }
                catch { }
                try
                {
                    string activeFile = AppConfigPaths.GetActiveServerFile();
                    if (File.Exists(activeFile))
                    {
                        File.Delete(activeFile);
                    }
                }
                catch { }
            }
        }
    }

    internal static class Program
    {
        [STAThread]
        public static int Main(string[] args)
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

#if CONFIG_ONLY
            using (GeminiConfigDialog dlg = new GeminiConfigDialog())
            {
                Application.Run(dlg);
            }
            return 0;
#else
            bool configOnly = false;
            bool verifyOnly = false;
            if (args != null)
            {
                for (int i = 0; i < args.Length; i++)
                {
                    string a = (args[i] ?? "").Trim().ToLowerInvariant();
                    if (a == "--configure-gemini") configOnly = true;
                    if (a == "--verify-startup") verifyOnly = true;
                }
            }

            if (configOnly)
            {
                using (GeminiConfigDialog dlg = new GeminiConfigDialog())
                {
                    Application.Run(dlg);
                }
                return 0;
            }

            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            using (LauncherMainForm form = new LauncherMainForm(baseDir, verifyOnly))
            {
                Application.Run(form);
                return form.ExitVerificationCode;
            }
#endif
        }
    }
}
