# shot_ui.ps1 -- capture a running Study Assistant window to a PNG.
# PrintWindow(h, hdc, 2) = PW_RENDERFULLCONTENT, so the capture works even
# when the window is partly covered by other windows.
#
#   powershell -ExecutionPolicy Bypass -File tools\shot_ui.ps1
#   powershell -ExecutionPolicy Bypass -File tools\shot_ui.ps1 -Title "Study Assistant"

param(
    # The dashboard's real title is the app's Chinese name, NOT "Study Assistant"
    # ("Study Assistant" is only the 220x120 eye widget). FindLargest() then picks
    # the biggest match, which is the dashboard.
    [string]$Title  = '桌面学习行为检测与专注陪伴助手',
    [string]$OutDir = 'E:\soft\workbuddy\StudyAssistant\out',
    [string]$Name   = 'live_shot'
)

Add-Type -AssemblyName System.Drawing

if (-not ('SaShot' -as [type])) {
    Add-Type -ReferencedAssemblies System.Drawing @'
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

public class SaShot
{
    [DllImport("user32.dll")] public static extern IntPtr FindWindow(string cls, string name);
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr dc, uint flags);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int cmd);
    [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h);

    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int Left, Top, Right, Bottom; }

    public static IntPtr Found = IntPtr.Zero;

    public static IntPtr FindBySubstring(string part)
    {
        IntPtr direct = FindWindow(null, part);
        if (direct != IntPtr.Zero) { Found = direct; return direct; }
        Found = IntPtr.Zero;
        EnumWindows(delegate(IntPtr h, IntPtr l) {
            int n = GetWindowTextLength(h);
            if (n <= 0) { return true; }
            System.Text.StringBuilder sb = new System.Text.StringBuilder(n + 2);
            GetWindowText(h, sb, sb.Capacity);
            string t = sb.ToString();
            if (t.IndexOf(part, StringComparison.OrdinalIgnoreCase) >= 0) {
                Found = h;
                return false;
            }
            return true;
        }, IntPtr.Zero);
        return Found;
    }

    public delegate bool EnumProc(IntPtr h, IntPtr l);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc cb, IntPtr l);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern int GetWindowTextLength(IntPtr h);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern int GetWindowText(IntPtr h, System.Text.StringBuilder s, int max);

    // The app owns several top-level windows that all carry the same title
    // (main window + frameless always-on-top eye widget + debug view).
    // Picking by area is the only reliable way to single out the dashboard.
    public static IntPtr FindLargest(string part)
    {
        IntPtr best = IntPtr.Zero;
        long bestArea = 0;

        EnumWindows(delegate(IntPtr h, IntPtr l) {
            int n = GetWindowTextLength(h);
            if (n <= 0) { return true; }
            System.Text.StringBuilder sb = new System.Text.StringBuilder(n + 2);
            GetWindowText(h, sb, sb.Capacity);
            string t = sb.ToString();
            if (t.IndexOf(part, StringComparison.OrdinalIgnoreCase) < 0) { return true; }

            RECT r;
            if (!GetWindowRect(h, out r)) { return true; }
            long area = (long)(r.Right - r.Left) * (r.Bottom - r.Top);
            if (area > bestArea) { bestArea = area; best = h; }
            return true;
        }, IntPtr.Zero);

        return best;
    }

    public static string Describe(IntPtr h)
    {
        RECT r;
        if (!GetWindowRect(h, out r)) { return "?"; }
        return "x=" + r.Left + " y=" + r.Top + " w=" + (r.Right - r.Left) + " h=" + (r.Bottom - r.Top);
    }

    public static string[] Titles()
    {
        System.Collections.Generic.List<string> res = new System.Collections.Generic.List<string>();
        EnumWindows(delegate(IntPtr h, IntPtr l) {
            int n = GetWindowTextLength(h);
            if (n <= 0) { return true; }
            System.Text.StringBuilder sb = new System.Text.StringBuilder(n + 2);
            GetWindowText(h, sb, sb.Capacity);
            if (sb.Length > 0) { res.Add(sb.ToString()); }
            return true;
        }, IntPtr.Zero);
        return res.ToArray();
    }

    public static bool Capture(IntPtr h, string path)
    {
        if (h == IntPtr.Zero) { return false; }
        if (IsIconic(h)) { ShowWindow(h, 9); System.Threading.Thread.Sleep(500); }
        SetForegroundWindow(h);
        System.Threading.Thread.Sleep(700);

        RECT r;
        if (!GetWindowRect(h, out r)) { return false; }
        int w = r.Right - r.Left;
        int ht = r.Bottom - r.Top;
        if (w <= 0 || ht <= 0) { return false; }

        using (Bitmap bmp = new Bitmap(w, ht, PixelFormat.Format32bppArgb))
        {
            using (Graphics g = Graphics.FromImage(bmp))
            {
                IntPtr hdc = g.GetHdc();
                PrintWindow(h, hdc, 2);
                g.ReleaseHdc(hdc);
            }
            bmp.Save(path, ImageFormat.Png);
        }
        return true;
    }
}
'@
}

if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir -Force | Out-Null }

$h = [SaShot]::FindLargest($Title)
if ($h -eq [IntPtr]::Zero) {
    Write-Output 'WINDOW_NOT_FOUND'
    Write-Output '--- current window titles ---'
    [SaShot]::Titles() | ForEach-Object { Write-Output $_ }
    exit 1
}
Write-Output ("WINDOW " + [SaShot]::Describe($h))

$png = Join-Path $OutDir ("$Name.png")
$ok = [SaShot]::Capture($h, $png)
if ($ok) {
    Write-Output "SAVED $png"
    $fi = Get-Item $png
    Write-Output ("SIZE_KB " + [int]($fi.Length / 1024))
} else {
    Write-Output 'CAPTURE_FAILED'
    exit 1
}
