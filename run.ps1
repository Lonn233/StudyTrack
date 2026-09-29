# Study Assistant launcher (ASCII only on purpose -- PowerShell 5.1 reads
# BOM-less UTF-8 as GBK and would fail to parse non-ASCII source).
#
#   powershell -ExecutionPolicy Bypass -File run.ps1
#   powershell -ExecutionPolicy Bypass -File run.ps1 -Headless -Seconds 30

param(
    [switch]$Headless,
    [double]$Seconds = 0,
    [switch]$NoDebug,
    [switch]$NoEyes,
    [switch]$Calibrate,
    [switch]$AutoCalibrate,
    [int]$Camera = -1,
    [double]$DumpEvery = 0
)

$ErrorActionPreference = 'Stop'

# --- 1) venv python (override with STUDYASSISTANT_PYTHON) ---
$py = $env:STUDYASSISTANT_PYTHON
if (-not $py) {
    $py = "C:\Users\Lonn\.workbuddy\binaries\python\envs\sg-demo\Scripts\python.exe"
}
if (-not (Test-Path $py)) {
    Write-Host "[ERROR] python not found: $py" -ForegroundColor Red
    Write-Host "        set `$env:STUDYASSISTANT_PYTHON to your interpreter" -ForegroundColor Yellow
    exit 2
}

# --- 2) required environment switches (see README section 9) ---
$env:CODEBUDDY_SAFE_DELETE_ENABLED = '0'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

# --- 3) assemble args ---
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$argv = @("$root\app.py")

if ($Headless)      { $argv += '--headless' }
if ($Seconds -gt 0) { $argv += @('--seconds', $Seconds) }
if ($NoDebug)       { $argv += '--no-debug' }
if ($NoEyes)        { $argv += '--no-eyes' }
if ($Calibrate)     { $argv += '--calibrate' }
if ($AutoCalibrate) { $argv += '--auto-calibrate' }
if ($Camera -ge 0)  { $argv += @('--camera', $Camera) }
if ($DumpEvery -gt 0) { $argv += @('--dump-every', $DumpEvery) }

Write-Host "[run] $py $($argv -join ' ')" -ForegroundColor Cyan

# The working directory must be the project root: model paths and the
# calibration file are resolved relative to it.
Set-Location $root

& $py @argv
exit $LASTEXITCODE
