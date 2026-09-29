# Run the full verification chain for Study Assistant (ASCII only).
#
#   powershell -ExecutionPolicy Bypass -File verify.ps1
#   powershell -ExecutionPolicy Bypass -File verify.ps1 -Frames 300

param(
    [int]$Frames = 200,
    [switch]$SkipPipeline
)

$ErrorActionPreference = 'Continue'

$py = $env:STUDYASSISTANT_PYTHON
if (-not $py) {
    $py = "C:\Users\Lonn\.workbuddy\binaries\python\envs\sg-demo\Scripts\python.exe"
}

$env:CODEBUDDY_SAFE_DELETE_ENABLED = '0'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$failed = @()

function Step($name, $scriptArgs) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor DarkGray
    Write-Host "  $name" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor DarkGray

    & $py @scriptArgs
    $code = $LASTEXITCODE

    if ($code -eq 0) {
        Write-Host "[OK] $name" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] $name (exit $code)" -ForegroundColor Red
        $script:failed += $name
    }
}

Step "1/3  stage1   camera / hands / object models" @("tests\stage1_check.py")

if (-not $SkipPipeline) {
    Step "2/3  pipeline frequency scheduling / contracts" @("tests\pipeline_check.py", "$Frames")
}

Step "3/3  acceptance  behaviour / state machine / reminders" @("tests\acceptance_test.py")

Write-Host ""
Write-Host ("=" * 70) -ForegroundColor DarkGray

if ($failed.Count -eq 0) {
    Write-Host "ALL CHECKS PASSED" -ForegroundColor Green
    exit 0
}

Write-Host "FAILED: $($failed -join ', ')" -ForegroundColor Red
exit 1
