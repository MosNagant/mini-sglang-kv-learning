# Restore the Microsoft Hyper-V Virtual Disk Server deleted by device cleanup.
# Does not delete distributions or change physical disk drivers.
[CmdletBinding()]
param([switch]$SkipSetup)
$ErrorActionPreference = 'Stop'
$repairIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
$repairPrincipal = New-Object Security.Principal.WindowsPrincipal($repairIdentity)
if ($repairIdentity.Name -match 'CodexSandbox' -or -not $repairPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host 'Run this script from your normal administrator PowerShell.' -ForegroundColor Yellow
    exit 1
}
if (-not [Environment]::Is64BitProcess) { throw 'Use the 64-bit Windows PowerShell.' }
$repairLogDir = Join-Path $PSScriptRoot 'logs'
New-Item -ItemType Directory -Path $repairLogDir -Force | Out-Null
$repairStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$repairLog = Join-Path $repairLogDir ('repair-storage-' + $repairStamp + '.log')
Start-Transcript -LiteralPath $repairLog -Force | Out-Null
$repairContinueInstall = $false

try {
    Write-Host 'Restoring Microsoft Hyper-V Virtual Disk Server using the existing Windows driver.' -ForegroundColor Cyan
    $repairInf = Join-Path $env:SystemRoot 'INF\wstorvsp.inf'
    $repairDriver = Join-Path $env:SystemRoot 'System32\drivers\storvsp.sys'
    foreach ($repairFile in @($repairInf,$repairDriver)) {
        if (-not (Test-Path -LiteralPath $repairFile)) { throw ('Required Windows file is missing: ' + $repairFile) }
        $repairSignature = Get-AuthenticodeSignature -LiteralPath $repairFile
        if ($repairSignature.Status -ne 'Valid' -or $repairSignature.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation(?:,|$)') {
            throw ('Microsoft signature check failed for ' + $repairFile + ': ' + $repairSignature.Status)
        }
    }
    if ((Get-Content -LiteralPath $repairInf -Raw) -notmatch '(?im)^\s*HardwareIds\s*=\s*root\\storvsp\s*$') {
        throw 'The Windows INF does not declare the expected storage hardware ID.'
    }
    $repairInfHash = (Get-FileHash -LiteralPath $repairInf -Algorithm SHA256).Hash
    $repairSourceCandidates = @(Get-ChildItem -LiteralPath (Join-Path $env:SystemRoot 'System32\DriverStore\FileRepository') -Filter 'wstorvsp.inf_*' -Directory |
        ForEach-Object { Join-Path $_.FullName 'wstorvsp.inf' } |
        Where-Object { (Test-Path -LiteralPath $_) -and (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash -eq $repairInfHash })
    if ($repairSourceCandidates.Count -ne 1) { throw 'Could not identify one matching Windows Driver Store package.' }
    $repairSourceInf = $repairSourceCandidates[0]
    $repairSourceDriver = Join-Path (Split-Path -Parent $repairSourceInf) 'storvsp.sys'
    if ((Get-FileHash -LiteralPath $repairSourceDriver -Algorithm SHA256).Hash -ne (Get-FileHash -LiteralPath $repairDriver -Algorithm SHA256).Hash) {
        throw 'The Driver Store package does not match the current Microsoft storage driver.'
    }
    Write-Host ('Using Windows Driver Store package: ' + $repairSourceInf)

    $repairBefore = @(Get-CimInstance Win32_PnPEntity -Filter "Service='storvsp'")
    $repairBefore | Select-Object Name,PNPDeviceID,Present,ConfigManagerErrorCode,Service |
        ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $repairLogDir ('storage-before-' + $repairStamp + '.json')) -Encoding UTF8
    & "$env:SystemRoot\System32\reg.exe" export 'HKLM\SYSTEM\CurrentControlSet\Services\storvsp' (Join-Path $repairLogDir ('storvsp-service-before-' + $repairStamp + '.reg')) /y
    if ($LASTEXITCODE -ne 0) { throw 'Could not save the current storage driver registration.' }

    $repairRoot = 'HKLM:\SYSTEM\CurrentControlSet\Enum\ROOT\STORVSP'
    $repairRootDevices = @()
    if (Test-Path -LiteralPath $repairRoot) { $repairRootDevices = @(Get-ChildItem -LiteralPath $repairRoot) }
    if ($repairBefore.Count -gt 1 -or $repairRootDevices.Count -gt 1) { throw 'Multiple storage devices were found; stopped to avoid changing an unexpected device.' }
    if ($repairRootDevices.Count -eq 1) {
        $repairRootProperties = Get-ItemProperty -LiteralPath $repairRootDevices[0].PSPath
        if (@($repairRootProperties.HardwareID) -notcontains 'root\storvsp') { throw 'The existing device has an unexpected hardware ID.' }
    }
    $repairNeedsCreation = ($repairBefore.Count -eq 0 -and $repairRootDevices.Count -eq 0)
    $repairAlreadyHealthy = ($repairBefore.Count -eq 1 -and $repairBefore[0].ConfigManagerErrorCode -eq 0 -and $repairBefore[0].Present)

    if (-not $repairAlreadyHealthy) {
        Write-Host ('Create missing device registration: ' + $repairNeedsCreation)
        Add-Type -Path (Join-Path $PSScriptRoot 'Restore-HyperVStorage.cs')
        $repairNeedsReboot = [RestoreHyperVStorage]::Restore($repairSourceInf,$repairNeedsCreation)
        if ($repairNeedsReboot) {
            Write-Host 'DEVICE RESTORED. Windows requires a restart to finish loading the driver.' -ForegroundColor Yellow
            Write-Host 'Save your work, restart Windows, then run this same repair command again.'
            exit 3010
        }
    }

    $repairAfter = @()
    for ($repairAttempt=0; $repairAttempt -lt 10; $repairAttempt++) {
        $repairAfter = @(Get-CimInstance Win32_PnPEntity -Filter "Service='storvsp'")
        if ($repairAfter.Count -eq 1 -and $repairAfter[0].ConfigManagerErrorCode -eq 0 -and $repairAfter[0].Present) { break }
        Start-Sleep -Seconds 1
    }
    $repairAfter | Select-Object Name,PNPDeviceID,Present,ConfigManagerErrorCode,Service | Format-List
    $repairAfter | Select-Object Name,PNPDeviceID,Present,ConfigManagerErrorCode,Service |
        ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $repairLogDir ('storage-after-' + $repairStamp + '.json')) -Encoding UTF8
    $repairDriverState = Get-CimInstance Win32_SystemDriver -Filter "Name='storvsp'"
    $repairDriverState | Select-Object Name,State,StartMode,PathName | Format-List
    if ($repairAfter.Count -ne 1 -or $repairAfter[0].ConfigManagerErrorCode -ne 0 -or -not $repairAfter[0].Present -or $repairDriverState.State -ne 'Running') {
        throw 'Storage device registration was attempted, but the device/driver is not healthy yet. Keep this log for diagnosis.'
    }
    Write-Host 'STORAGE DEVICE OK.' -ForegroundColor Green
    if (-not $SkipSetup) {
        Write-Host 'Continuing with Ubuntu installation/verification...'
        $repairContinueInstall = $true
    }
} catch {
    Write-Host ('REPAIR STOPPED: ' + $_.Exception.Message) -ForegroundColor Red
    Write-Host ('Log: ' + $repairLog)
    exit 1
} finally {
    Stop-Transcript | Out-Null
}

if ($repairContinueInstall) {
    & "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'Install-WSL.ps1')
    exit $LASTEXITCODE
}
exit 0
