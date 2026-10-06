[CmdletBinding()]
param([switch]$RepairStorage)
$ErrorActionPreference = 'Stop'
$cudaIdentity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
if ($cudaIdentity -match 'CodexSandbox') {
    Write-Host 'Run this script in PowerShell from your normal Windows account.' -ForegroundColor Yellow
    exit 1
}
if ($RepairStorage) {
    $cudaPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    if (-not $cudaPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Host 'Restoring the missing Windows storage device requires administrator PowerShell.' -ForegroundColor Yellow
        Write-Host 'Open Terminal (Admin), then run this same command with -RepairStorage.'
        exit 1
    }
}

$env:WSL_UTF8 = '1'
$cudaWsl = Join-Path $env:SystemRoot 'System32\wsl.exe'
$cudaDistro = 'CUDA-Ubuntu-24.04'
$cudaLogDir = Join-Path $PSScriptRoot 'logs'
New-Item -ItemType Directory -Path $cudaLogDir -Force | Out-Null
$cudaLauncherLog = Join-Path $cudaLogDir ('cuda-launcher-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
Start-Transcript -LiteralPath $cudaLauncherLog -Force | Out-Null
$cudaExitCode = 0

function Invoke-CudaWsl {
    param([string[]]$WslArguments)
    Write-Host ('> wsl ' + ($WslArguments -join ' ')) -ForegroundColor Cyan
    # Keep original stdout/stderr visible, including errors before Linux starts.
    & $cudaWsl @WslArguments
    $cudaCommandExit = $LASTEXITCODE
    if ($cudaCommandExit -ne 0) {
        throw ('WSL command failed with exit code ' + $cudaCommandExit + '. The original output is shown above.')
    }
}

try {
    Write-Host 'Preparing CUDA Toolkit 12.8 for CUDA-Ubuntu-24.04 / weltlaw.' -ForegroundColor Cyan
    Write-Host ('Launcher log: ' + $cudaLauncherLog)
    Write-Host ('Windows account: ' + $cudaIdentity)
    $cudaSource = Join-Path $PSScriptRoot 'setup-cuda.sh'
    if (-not (Test-Path -LiteralPath $cudaSource)) { throw 'Missing setup-cuda.sh beside this launcher.' }
    if ($RepairStorage) {
        & "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'Find-DeviceCleanup.ps1')
        if ($LASTEXITCODE -ne 0) { Write-Host 'Some cleanup diagnostics were unavailable. Continuing with the known device repair.' -ForegroundColor Yellow }
        Write-Host 'Restoring the missing Hyper-V storage device with the existing Microsoft driver...'
        & "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'Repair-WSL-Storage.ps1') -SkipSetup
        $cudaRepairExit = $LASTEXITCODE
        if ($cudaRepairExit -eq 3010) { throw 'Windows requires a restart to finish the repair. Save your work, restart, then rerun this command.' }
        if ($cudaRepairExit -ne 0) { throw ('Storage repair stopped with exit code ' + $cudaRepairExit + '. Keep the repair log shown above.') }
    }
    if (-not (Test-Path -LiteralPath 'HKLM:\SYSTEM\CurrentControlSet\Enum\ROOT\STORVSP\0000')) {
        throw 'Hyper-V device ROOT\STORVSP\0000 is missing. Run this launcher with -RepairStorage in administrator PowerShell.'
    }
    Invoke-CudaWsl -WslArguments @('--version')
    Invoke-CudaWsl -WslArguments @('--list','--verbose')
    Invoke-CudaWsl -WslArguments @('-d',$cudaDistro,'-u','weltlaw','--cd','~','--exec','/usr/bin/id','-un')
    Write-Host 'Ubuntu starts as weltlaw. Enter your Linux password if sudo asks.' -ForegroundColor Green
    # --cd accepts an absolute Windows directory; no captured wslpath output.
    Invoke-CudaWsl -WslArguments @('-d',$cudaDistro,'-u','weltlaw','--cd',$PSScriptRoot,'--exec','/bin/bash','-n','./setup-cuda.sh')
    Invoke-CudaWsl -WslArguments @('-d',$cudaDistro,'-u','weltlaw','--cd',$PSScriptRoot,'--exec','/bin/bash','./setup-cuda.sh')
    Write-Host 'CUDA setup and GPU calculation passed.' -ForegroundColor Green
    Write-Host 'Open Ubuntu: wsl -d CUDA-Ubuntu-24.04 --cd ~'
    Write-Host 'Your example project: ~/projects/cuda-starter (or the dated path printed by setup).'
} catch {
    Write-Host ('SETUP STOPPED: ' + $_.Exception.Message) -ForegroundColor Red
    Write-Host ('Log: ' + $cudaLauncherLog)
    $cudaExitCode = 1
} finally {
    Stop-Transcript | Out-Null
}
exit $cudaExitCode
