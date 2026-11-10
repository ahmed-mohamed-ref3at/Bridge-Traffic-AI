using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Windows.Forms;
using Microsoft.Win32;

namespace BridgeTrafficSetup
{
    internal sealed class BoundedReadStream : Stream
    {
        private readonly Stream _inner;
        private readonly long _start;
        private readonly long _length;
        private long _pos;

        public BoundedReadStream(Stream inner, long start, long length)
        {
            _inner = inner;
            _start = start;
            _length = length;
            _pos = 0;
            _inner.Seek(_start, SeekOrigin.Begin);
        }

        public override bool CanRead { get { return true; } }
        public override bool CanSeek { get { return true; } }
        public override bool CanWrite { get { return false; } }
        public override long Length { get { return _length; } }

        public override long Position
        {
            get { return _pos; }
            set { Seek(value, SeekOrigin.Begin); }
        }

        public override void Flush() { }

        public override int Read(byte[] buffer, int offset, int count)
        {
            long remaining = _length - _pos;
            if (remaining <= 0) return 0;
            if (count > remaining) count = (int)remaining;
            _inner.Seek(_start + _pos, SeekOrigin.Begin);
            int read = _inner.Read(buffer, offset, count);
            _pos += read;
            return read;
        }

        public override long Seek(long offset, SeekOrigin origin)
        {
            long target = _pos;
            if (origin == SeekOrigin.Begin) target = offset;
            else if (origin == SeekOrigin.Current) target = _pos + offset;
            else if (origin == SeekOrigin.End) target = _length + offset;

            if (target < 0) target = 0;
            if (target > _length) target = _length;
            _pos = target;
            _inner.Seek(_start + _pos, SeekOrigin.Begin);
            return _pos;
        }

        public override void SetLength(long value) { throw new NotSupportedException(); }
        public override void Write(byte[] buffer, int offset, int count) { throw new NotSupportedException(); }
    }

    internal static class ShortcutHelper
    {
        public static void CreateShortcut(string shortcutPath, string targetExe, string workingDir, string description)
        {
            try
            {
                string parent = Path.GetDirectoryName(shortcutPath);
                if (!string.IsNullOrEmpty(parent) && !Directory.Exists(parent))
                {
                    Directory.CreateDirectory(parent);
                }
                Type shellType = Type.GetTypeFromProgID("WScript.Shell");
                if (shellType == null) return;
                object shell = Activator.CreateInstance(shellType);
                object shortcut = shellType.InvokeMember("CreateShortcut", BindingFlags.InvokeMethod, null, shell, new object[] { shortcutPath });
                Type scType = shortcut.GetType();
                scType.InvokeMember("TargetPath", BindingFlags.SetProperty, null, shortcut, new object[] { targetExe });
                scType.InvokeMember("WorkingDirectory", BindingFlags.SetProperty, null, shortcut, new object[] { workingDir });
                scType.InvokeMember("Description", BindingFlags.SetProperty, null, shortcut, new object[] { description });
                scType.InvokeMember("IconLocation", BindingFlags.SetProperty, null, shortcut, new object[] { targetExe + ",0" });
                scType.InvokeMember("Save", BindingFlags.InvokeMethod, null, shortcut, null);
            }
            catch
            {
                // Non-fatal if shell COM is restricted in headless sandbox
            }
        }
    }

#if !UNINSTALLER_MODE
    internal sealed class InstallerWizardForm : Form
    {
        private const string MAGIC_SIG = "BTAIINST";
        private TextBox _pathBox;
        private CheckBox _desktopCheck;
        private CheckBox _startMenuCheck;
        private CheckBox _launchAfterCheck;
        private ProgressBar _progressBar;
        private Label _statusLabel;
        private Button _installBtn;
        private Button _browseBtn;
        private Button _cancelBtn;
        private bool _isInstalling = false;

        public InstallerWizardForm()
        {
            this.Text = "تثبيت محلل بيانات المرور والمشاة الذكي — كوبري السلاب";
            this.Size = new Size(620, 430);
            this.FormBorderStyle = FormBorderStyle.FixedDialog;
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
            this.FormClosing += delegate(object s, FormClosingEventArgs e)
            {
                if (_isInstalling)
                {
                    e.Cancel = true;
                }
            };
        }

        private static string GetDefaultInstallPath()
        {
            string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            if (string.IsNullOrEmpty(localAppData))
            {
                localAppData = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "AppData\\Local");
            }
            return Path.Combine(localAppData, "BridgeTrafficAI_App");
        }

        private void BuildUi()
        {
            Panel banner = new Panel();
            banner.Dock = DockStyle.Top;
            banner.Height = 82;
            banner.BackColor = Color.FromArgb(15, 118, 110);

            Label h1 = new Label();
            h1.Text = "معالج تثبيت محلل بيانات المرور والمشاة الذكي";
            h1.ForeColor = Color.White;
            h1.Font = new Font("Segoe UI", 13.5f, FontStyle.Bold);
            h1.Location = new Point(22, 14);
            h1.Size = new Size(560, 30);
            banner.Controls.Add(h1);

            Label h2 = new Label();
            h2.Text = "نسخة ويندوز الكاملة — تشمل المحرك المدمج وجميع لوحات البيانات بدون أي إعدادات تقنية";
            h2.ForeColor = Color.FromArgb(204, 251, 241);
            h2.Font = new Font("Segoe UI", 9.5f, FontStyle.Regular);
            h2.Location = new Point(22, 46);
            h2.Size = new Size(560, 24);
            banner.Controls.Add(h2);

            this.Controls.Add(banner);

            Label pathLbl = new Label();
            pathLbl.Text = "مجلد تثبيت البرنامج على جهازك:";
            pathLbl.Font = new Font("Segoe UI", 9.5f, FontStyle.Bold);
            pathLbl.Location = new Point(22, 98);
            pathLbl.Size = new Size(300, 24);
            this.Controls.Add(pathLbl);

            _pathBox = new TextBox();
            _pathBox.Location = new Point(135, 126);
            _pathBox.Size = new Size(445, 28);
            _pathBox.RightToLeft = RightToLeft.No;
            _pathBox.Text = GetDefaultInstallPath();
            this.Controls.Add(_pathBox);

            _browseBtn = new Button();
            _browseBtn.Text = "تغيير المسار...";
            _browseBtn.Location = new Point(22, 124);
            _browseBtn.Size = new Size(105, 30);
            _browseBtn.FlatStyle = FlatStyle.System;
            _browseBtn.Click += OnBrowseClick;
            this.Controls.Add(_browseBtn);

            Panel optsPanel = new Panel();
            optsPanel.Location = new Point(22, 168);
            optsPanel.Size = new Size(558, 102);
            optsPanel.BackColor = Color.White;
            optsPanel.BorderStyle = BorderStyle.FixedSingle;

            _desktopCheck = new CheckBox();
            _desktopCheck.Text = "إنشاء أيقونة اختصار على سطح المكتب (محلل المرور الذكي - كوبري السلاب)";
            _desktopCheck.Checked = true;
            _desktopCheck.Location = new Point(16, 12);
            _desktopCheck.Size = new Size(520, 24);
            optsPanel.Controls.Add(_desktopCheck);

            _startMenuCheck = new CheckBox();
            _startMenuCheck.Text = "إضافة البرنامج إلى قائمة ابدأ (Start Menu)";
            _startMenuCheck.Checked = true;
            _startMenuCheck.Location = new Point(16, 40);
            _startMenuCheck.Size = new Size(520, 24);
            optsPanel.Controls.Add(_startMenuCheck);

            _launchAfterCheck = new CheckBox();
            _launchAfterCheck.Text = "تشغيل البرنامج وفتح الواجهة تلقائيًا فور انتهاء التثبيت";
            _launchAfterCheck.Checked = true;
            _launchAfterCheck.Font = new Font("Segoe UI", 9.5f, FontStyle.Bold);
            _launchAfterCheck.Location = new Point(16, 68);
            _launchAfterCheck.Size = new Size(520, 24);
            optsPanel.Controls.Add(_launchAfterCheck);

            this.Controls.Add(optsPanel);

            _statusLabel = new Label();
            _statusLabel.Text = "اضغط على «تثبيت وتشغيل الآن» لبدء التثبيت التلقائي السريع.";
            _statusLabel.Location = new Point(22, 282);
            _statusLabel.Size = new Size(558, 24);
            _statusLabel.ForeColor = Color.FromArgb(51, 65, 85);
            this.Controls.Add(_statusLabel);

            _progressBar = new ProgressBar();
            _progressBar.Location = new Point(22, 310);
            _progressBar.Size = new Size(558, 22);
            _progressBar.Minimum = 0;
            _progressBar.Maximum = 100;
            _progressBar.Value = 0;
            this.Controls.Add(_progressBar);

            _installBtn = new Button();
            _installBtn.Text = "تثبيت وتشغيل الآن";
            _installBtn.Location = new Point(22, 344);
            _installBtn.Size = new Size(260, 40);
            _installBtn.BackColor = Color.FromArgb(15, 118, 110);
            _installBtn.ForeColor = Color.White;
            _installBtn.FlatStyle = FlatStyle.Flat;
            _installBtn.Font = new Font("Segoe UI", 10.5f, FontStyle.Bold);
            _installBtn.Click += OnInstallClick;
            this.Controls.Add(_installBtn);

            _cancelBtn = new Button();
            _cancelBtn.Text = "إلغاء";
            _cancelBtn.Location = new Point(450, 344);
            _cancelBtn.Size = new Size(130, 40);
            _cancelBtn.FlatStyle = FlatStyle.System;
            _cancelBtn.Click += delegate { if (!_isInstalling) this.Close(); };
            this.Controls.Add(_cancelBtn);
        }

        private void OnBrowseClick(object sender, EventArgs e)
        {
            using (FolderBrowserDialog fbd = new FolderBrowserDialog())
            {
                fbd.Description = "اختر المجلد الذي ترغب في تثبيت البرنامج بداخله:";
                fbd.SelectedPath = _pathBox.Text;
                if (fbd.ShowDialog(this) == DialogResult.OK && !string.IsNullOrEmpty(fbd.SelectedPath))
                {
                    _pathBox.Text = Path.Combine(fbd.SelectedPath, "BridgeTrafficAI_App");
                }
            }
        }

        private void OnInstallClick(object sender, EventArgs e)
        {
            string targetDir = _pathBox.Text.Trim();
            if (string.IsNullOrEmpty(targetDir))
            {
                MessageBox.Show(
                    "يرجى تحديد مسار صالح لتثبيت البرنامج.",
                    "تنبيه",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Warning,
                    MessageBoxDefaultButton.Button1,
                    MessageBoxOptions.RightAlign | MessageBoxOptions.RtlReading
                );
                return;
            }

            _isInstalling = true;
            _installBtn.Enabled = false;
            _browseBtn.Enabled = false;
            _cancelBtn.Enabled = false;
            _pathBox.ReadOnly = true;

            bool createDesktop = _desktopCheck.Checked;
            bool createStartMenu = _startMenuCheck.Checked;
            bool launchNow = _launchAfterCheck.Checked;

            Thread worker = new Thread(delegate()
            {
                try
                {
                    PerformExtractionAndSetup(
                        Application.ExecutablePath,
                        targetDir,
                        createDesktop,
                        createStartMenu,
                        delegate(int pct, string msg)
                        {
                            try
                            {
                                this.BeginInvoke((MethodInvoker)delegate
                                {
                                    if (pct >= 0 && pct <= 100) _progressBar.Value = pct;
                                    _statusLabel.Text = msg;
                                });
                            }
                            catch { }
                        }
                    );

                    this.BeginInvoke((MethodInvoker)delegate
                    {
                        _isInstalling = false;
                        _progressBar.Value = 100;
                        _statusLabel.ForeColor = Color.FromArgb(5, 150, 105);
                        _statusLabel.Text = "اكتمل تثبيت البرنامج بنجاح 100%!";

                        string launcherExe = Path.Combine(targetDir, "تشغيل محلل المرور الذكي.exe");
                        if (launchNow && File.Exists(launcherExe))
                        {
                            try
                            {
                                ProcessStartInfo psi = new ProcessStartInfo();
                                psi.FileName = launcherExe;
                                psi.WorkingDirectory = targetDir;
                                psi.UseShellExecute = true;
                                Process.Start(psi);
                            }
                            catch { }
                        }
                        else
                        {
                            MessageBox.Show(
                                this,
                                "تم تثبيت «محلل بيانات المرور والمشاة الذكي» بنجاح!\r\nيمكنك تشغيله في أي وقت من الأيقونة الموجودة على سطح المكتب.",
                                "اكتمل التثبيت بنجاح",
                                MessageBoxButtons.OK,
                                MessageBoxIcon.Information,
                                MessageBoxDefaultButton.Button1,
                                MessageBoxOptions.RightAlign | MessageBoxOptions.RtlReading
                            );
                        }
                        this.Close();
                    });
                }
                catch (Exception ex)
                {
                    this.BeginInvoke((MethodInvoker)delegate
                    {
                        _isInstalling = false;
                        _installBtn.Enabled = true;
                        _browseBtn.Enabled = true;
                        _cancelBtn.Enabled = true;
                        _pathBox.ReadOnly = false;
                        _statusLabel.ForeColor = Color.FromArgb(220, 38, 38);
                        _statusLabel.Text = "حدث خطأ أثناء التثبيت.";
                        MessageBox.Show(
                            this,
                            "تعذر إكمال التثبيت في المسار المحدد:\r\n" + ex.Message + "\r\n\r\nيرجى التأكد من توفر مساحة كافية على القرص وإغلاق أي نسخة مفتوحة سابقًا من البرنامج.",
                            "تنبيه أثناء التثبيت",
                            MessageBoxButtons.OK,
                            MessageBoxIcon.Error,
                            MessageBoxDefaultButton.Button1,
                            MessageBoxOptions.RightAlign | MessageBoxOptions.RtlReading
                        );
                    });
                }
            });
            worker.IsBackground = true;
            worker.Start();
        }

        internal delegate void ProgressCallback(int percent, string message);

        internal static void PerformExtractionAndSetup(
            string installerExePath,
            string targetDir,
            bool createDesktopShortcut,
            bool createStartMenuShortcut,
            ProgressCallback onProgress)
        {
            if (onProgress != null) onProgress(2, "جاري فحص حزمة التثبيت المدمجة...");

            using (FileStream fs = new FileStream(installerExePath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                if (fs.Length < 32)
                {
                    throw new InvalidDataException("ملف التثبيت غير مكتمل.");
                }
                fs.Seek(-16, SeekOrigin.End);
                byte[] footer = new byte[16];
                int readFooter = fs.Read(footer, 0, 16);
                if (readFooter != 16)
                {
                    throw new InvalidDataException("تعذر قراءة ترويسة الحزمة المدمجة.");
                }
                long payloadLength = BitConverter.ToInt64(footer, 0);
                string magic = Encoding.ASCII.GetString(footer, 8, 8);
                if (magic != MAGIC_SIG || payloadLength <= 0 || payloadLength > fs.Length - 16)
                {
                    throw new InvalidDataException("حزمة البيانات المدمجة داخل ملف التثبيت غير صالحة أو تالفة أثناء التحميل.");
                }

                long payloadStart = fs.Length - 16 - payloadLength;
                if (!Directory.Exists(targetDir))
                {
                    Directory.CreateDirectory(targetDir);
                }

                string fullTargetRoot = Path.GetFullPath(targetDir);
                if (!fullTargetRoot.EndsWith(Path.DirectorySeparatorChar.ToString()))
                {
                    fullTargetRoot += Path.DirectorySeparatorChar;
                }

                using (BoundedReadStream subStream = new BoundedReadStream(fs, payloadStart, payloadLength))
                using (ZipArchive archive = new ZipArchive(subStream, ZipArchiveMode.Read, false, Encoding.UTF8))
                {
                    int total = archive.Entries.Count;
                    int index = 0;
                    foreach (ZipArchiveEntry entry in archive.Entries)
                    {
                        index++;
                        string relPath = entry.FullName.Replace('/', Path.DirectorySeparatorChar);
                        string destPath = Path.GetFullPath(Path.Combine(targetDir, relPath));
                        if (!destPath.StartsWith(fullTargetRoot, StringComparison.OrdinalIgnoreCase))
                        {
                            continue;
                        }

                        if (string.IsNullOrEmpty(entry.Name) && (entry.FullName.EndsWith("/") || entry.FullName.EndsWith("\\")))
                        {
                            if (!Directory.Exists(destPath))
                            {
                                Directory.CreateDirectory(destPath);
                            }
                            continue;
                        }

                        string parentDir = Path.GetDirectoryName(destPath);
                        if (!string.IsNullOrEmpty(parentDir) && !Directory.Exists(parentDir))
                        {
                            Directory.CreateDirectory(parentDir);
                        }

                        using (Stream inStream = entry.Open())
                        using (FileStream outStream = new FileStream(destPath, FileMode.Create, FileAccess.Write, FileShare.None))
                        {
                            inStream.CopyTo(outStream);
                        }

                        if (onProgress != null && (index % 25 == 0 || index == total))
                        {
                            int pct = 5 + (int)((index * 88L) / Math.Max(1, total));
                            onProgress(pct, "جاري استخراج ملفات البرنامج والمحرك التحليلي (" + index.ToString() + " / " + total.ToString() + ")...");
                        }
                    }
                }
            }

            if (onProgress != null) onProgress(95, "جاري إنشاء الاختصارات وتسجيل البرنامج...");

            string launcherExe = Path.Combine(targetDir, "تشغيل محلل المرور الذكي.exe");
            string shortcutTitle = "محلل المرور الذكي - كوبري السلاب.lnk";
            string desc = "تشغيل محلل بيانات المرور والمشاة الذكي — كوبري السلاب";

            if (createDesktopShortcut && File.Exists(launcherExe))
            {
                string desktopDir = Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory);
                if (!string.IsNullOrEmpty(desktopDir))
                {
                    ShortcutHelper.CreateShortcut(Path.Combine(desktopDir, shortcutTitle), launcherExe, targetDir, desc);
                }
            }

            if (createStartMenuShortcut && File.Exists(launcherExe))
            {
                string programsDir = Environment.GetFolderPath(Environment.SpecialFolder.Programs);
                if (!string.IsNullOrEmpty(programsDir))
                {
                    ShortcutHelper.CreateShortcut(Path.Combine(programsDir, shortcutTitle), launcherExe, targetDir, desc);
                }
            }

            // Register per-user uninstaller in HKCU (no Admin rights required)
            try
            {
                string uninstallerExe = Path.Combine(targetDir, "إلغاء تثبيت البرنامج.exe");
                using (RegistryKey key = Registry.CurrentUser.CreateSubKey(@"Software\Microsoft\Windows\CurrentVersion\Uninstall\BridgeTrafficAI"))
                {
                    if (key != null)
                    {
                        key.SetValue("DisplayName", "محلل بيانات المرور والمشاة الذكي — كوبري السلاب");
                        key.SetValue("DisplayIcon", launcherExe);
                        key.SetValue("InstallLocation", targetDir);
                        key.SetValue("Publisher", "Bridge Traffic Analytics");
                        key.SetValue("DisplayVersion", "1.0.0");
                        key.SetValue("NoModify", 1, RegistryValueKind.DWord);
                        key.SetValue("NoRepair", 1, RegistryValueKind.DWord);
                        if (File.Exists(uninstallerExe))
                        {
                            key.SetValue("UninstallString", "\"" + uninstallerExe + "\"");
                        }
                    }
                }
            }
            catch { }

            if (onProgress != null) onProgress(100, "اكتمل التثبيت بنجاح!");
        }
    }
#endif

    internal static class SetupProgram
    {
        [STAThread]
        public static int Main(string[] args)
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

#if UNINSTALLER_MODE
            string appDir = AppDomain.CurrentDomain.BaseDirectory.TrimEnd('\\', '/');
            DialogResult res = MessageBox.Show(
                "هل ترغب في إلغاء تثبيت «محلل بيانات المرور والمشاة الذكي» وحذف اختصاراته من هذا الجهاز؟",
                "تأكيد إلغاء التثبيت",
                MessageBoxButtons.YesNo,
                MessageBoxIcon.Question,
                MessageBoxDefaultButton.Button2,
                MessageBoxOptions.RightAlign | MessageBoxOptions.RtlReading
            );
            if (res != DialogResult.Yes) return 0;

            try
            {
                string shortcutTitle = "محلل المرور الذكي - كوبري السلاب.lnk";
                string desktopLnk = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), shortcutTitle);
                if (File.Exists(desktopLnk)) File.Delete(desktopLnk);

                string startLnk = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Programs), shortcutTitle);
                if (File.Exists(startLnk)) File.Delete(startLnk);

                Registry.CurrentUser.DeleteSubKeyTree(@"Software\Microsoft\Windows\CurrentVersion\Uninstall\BridgeTrafficAI", false);
            }
            catch { }

            // Schedule self-deletion of directory via cmd after process exits
            try
            {
                ProcessStartInfo psi = new ProcessStartInfo();
                psi.FileName = "cmd.exe";
                psi.Arguments = "/C ping 127.0.0.1 -n 3 > nul & rmdir /S /Q \"" + appDir + "\"";
                psi.CreateNoWindow = true;
                psi.UseShellExecute = false;
                Process.Start(psi);
            }
            catch { }

            MessageBox.Show(
                "تمت إزالة البرنامج واختصاراته بنجاح.",
                "اكتمل إلغاء التثبيت",
                MessageBoxButtons.OK,
                MessageBoxIcon.Information,
                MessageBoxDefaultButton.Button1,
                MessageBoxOptions.RightAlign | MessageBoxOptions.RtlReading
            );
            return 0;
#else
            // Support automated clean-environment verification via --silent-install <dir>
            if (args != null && args.Length >= 2 && args[0] == "--silent-install")
            {
                try
                {
                    string targetDir = args[1];
                    InstallerWizardForm.PerformExtractionAndSetup(
                        Application.ExecutablePath,
                        targetDir,
                        true,
                        true,
                        null
                    );
                    return 0;
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine("SILENT_INSTALL_ERROR: " + ex.ToString());
                    return 1;
                }
            }

            using (InstallerWizardForm wiz = new InstallerWizardForm())
            {
                Application.Run(wiz);
            }
            return 0;
#endif
        }
    }
}
