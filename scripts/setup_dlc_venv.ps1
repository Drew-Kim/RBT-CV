param(
    [string]$EnvPath = ".venv-dlc"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $EnvPath)) {
    python -m venv $EnvPath
}

$python = Join-Path $EnvPath "Scripts\python.exe"

& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt
& $python -m pip install "deeplabcut[gui]"
& $python -c "import deeplabcut; print('DeepLabCut installed:', deeplabcut.__version__)"

Write-Host ""
Write-Host "Activate with:"
Write-Host "  .\$EnvPath\Scripts\Activate.ps1"
