# Run from the user's own Windows account, in an administrator PowerShell.
# Official references: https://learn.microsoft.com/windows/wsl/install
# https://docs.nvidia.com/cuda/wsl-user-guide/
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$setupIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
$setupPrincipal = New-Object Security.Principal.WindowsPrincipal($setupIdentity)
if ($setupIdentity.Name -match 'CodexSandbox' -or -not $setupPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host 'Run Start-WSL-Setup.cmd as administrator from your normal Windows account.' -ForegroundColor Yellow
    exit 1
}

$setupDistro = 'CUDA-Ubuntu-24.04'
$setupLogDir = Join-Path $PSScriptRoot 'logs'
New-Item -ItemType Directory -Path $setupLogDir -Force | Out-Null
$setupStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$setupLog = Join-Path $setupLogDir ('setup-' + $setupStamp + '.log')
$env:WSL_UTF8 = '1'
Start-Transcript -LiteralPath $setupLog -Force | Out-Null

function Invoke-SetupWsl {
    param([string[]]$WslArguments)
    Write-Host ('> wsl ' + ($WslArguments -join ' ')) -ForegroundColor Cyan
    & "$env:SystemRoot\System32\wsl.exe" @WslArguments
    if ($LASTEXITCODE -ne 0) {
        throw ('WSL failed with exit code ' + $LASTEXITCODE + '. See the output above and the log.')
    }
}

function Get-SetupDistros {
    $key = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss'
    if (Test-Path $key) {
        Get-ChildItem $key | ForEach-Object { Get-ItemProperty $_.PSPath }
    }
}

function Install-SetupWslMsi {
    $packagePath = Join-Path $PSScriptRoot 'packages\wsl.2.7.14.0.x64.msi'
    $expectedHash = 'DB084E536279A59E90A26EC598D8AA8A4DFF8309F41D078FD06242953AC1EBCD'
    if (-not (Test-Path -LiteralPath $packagePath)) {
        throw ('Official WSL MSI is not ready yet: ' + $packagePath)
    }
    Write-Host 'Verifying the local Microsoft WSL installer...'
    if ((Get-FileHash -LiteralPath $packagePath -Algorithm SHA256).Hash -ne $expectedHash) {
        throw 'WSL MSI SHA-256 does not match the official Microsoft release.'
    }
    $packageSignature = Get-AuthenticodeSignature -LiteralPath $packagePath
    if ($packageSignature.Status -ne 'Valid' -or $packageSignature.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation(?:,|$)') {
        throw ('WSL installer signature verification failed: ' + $packageSignature.Status)
    }
    $msiLog = Join-Path $setupLogDir ('wsl-msi-' + $setupStamp + '.log')
    $msiArguments = '/i "' + $packagePath + '" /qn /norestart /L*v "' + $msiLog + '"'
    Write-Host 'Installing WSL 2.7.14 from the verified local MSI (no network download needed)...'
    $msiProcess = Start-Process -FilePath "$env:SystemRoot\System32\msiexec.exe" -ArgumentList $msiArguments -WindowStyle Hidden -Wait -PassThru
    if ($msiProcess.ExitCode -notin @(0,3010)) {
        throw ('WSL MSI installation failed: ' + $msiProcess.ExitCode + '. Log: ' + $msiLog)
    }
    (Get-Date).ToString('o') | Set-Content -LiteralPath (Join-Path $setupLogDir 'wsl-runtime-installed-at.txt') -Encoding ASCII
    return $msiProcess.ExitCode
}

function Save-SetupFailureEvents {
    foreach ($eventChannel in @('Microsoft-Windows-Hyper-V-Compute-Operational','Microsoft-Windows-Hyper-V-Worker-Admin')) {
        try {
            $eventLogPath = Join-Path $setupLogDir ('failure-' + $setupStamp + '-' + $eventChannel + '.json')
            Get-WinEvent -FilterHashtable @{LogName=$eventChannel;StartTime=(Get-Date).AddHours(-1)} -MaxEvents 40 -ErrorAction Stop |
                Select-Object TimeCreated,Id,LevelDisplayName,Message,@{Name='EventXml';Expression={$_.ToXml()}} |
                ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $eventLogPath -Encoding UTF8
        } catch {
            Write-Host ('Could not collect ' + $eventChannel + ': ' + $_.Exception.Message)
        }
    }
}

try {
    Write-Host 'WSL 2 + Ubuntu 24.04 LTS setup' -ForegroundColor Green
    Write-Host ('Windows account: ' + $setupIdentity.Name)
    Write-Host 'Existing distributions and their files will be preserved. No automatic restart.'
    $setupOs = Get-CimInstance Win32_OperatingSystem
    $setupComputer = Get-CimInstance Win32_ComputerSystem
    $setupCpu = Get-CimInstance Win32_Processor
    $setupOs | Select-Object Caption,Version,BuildNumber,OSArchitecture,LastBootUpTime | Format-List
    $setupCpu | Select-Object Name,VirtualizationFirmwareEnabled,SecondLevelAddressTranslationExtensions | Format-List
    $setupComputer | Select-Object HypervisorPresent | Format-List
    Get-SetupDistros | Select-Object DistributionName,Version,BasePath | Format-Table -AutoSize
    if ([int]$setupOs.BuildNumber -lt 19041) { throw 'This Windows build must be updated before installing modern WSL.' }
    if (-not $setupComputer.HypervisorPresent -and @($setupCpu | Where-Object { $_.VirtualizationFirmwareEnabled -eq $true }).Count -eq 0) {
        throw 'CPU virtualization is not available. Enable Intel VT-x/VMX or AMD SVM in BIOS/UEFI, then rerun this setup.'
    }

    $setupReboot = $false
    $setupFeature = Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
    Write-Host ('VirtualMachinePlatform: ' + $setupFeature.State)
    if ($setupFeature.State -eq 'EnablePending') {
        $setupReboot = $true
    } elseif ($setupFeature.State -ne 'Enabled') {
        $setupFeatureResult = Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -All -NoRestart
        $setupReboot = $true
    }

    # Check for a disabled hypervisor; do not silently rewrite the boot configuration.
    $setupBootConfig = (& "$env:SystemRoot\System32\bcdedit.exe" /enum '{current}' | Out-String)
    if ($LASTEXITCODE -ne 0) { throw 'Could not read Windows boot configuration.' }
    $setupBootConfig | Write-Host
    if ($setupBootConfig -match '(?im)^\s*hypervisorlaunchtype\s+Off\s*$') {
        throw 'Windows hypervisor boot is disabled. Run bcdedit /set hypervisorlaunchtype auto in administrator PowerShell, restart Windows, then rerun this setup.'
    }

    # Avoid the system launcher's download route, which returned HTTP 403 on this PC.
    $setupVersionOutput = @(& "$env:SystemRoot\System32\wsl.exe" --version)
    $setupVersionExit = $LASTEXITCODE
    $setupVersionMatch = [regex]::Match(($setupVersionOutput -join ' '), '\b(\d+\.\d+\.\d+(?:\.\d+)?)\b')
    $setupRuntimeReady = $false
    if ($setupVersionExit -eq 0 -and $setupVersionMatch.Success) {
        $setupRuntimeReady = ([version]$setupVersionMatch.Groups[1].Value -ge [version]'2.7.14')
    }
    if (-not $setupRuntimeReady) {
        $setupMsiExit = Install-SetupWslMsi
        $setupReboot = $true
    }
    $setupRuntimeMarker = Join-Path $setupLogDir 'wsl-runtime-installed-at.txt'
    if (Test-Path -LiteralPath $setupRuntimeMarker) {
        $setupRuntimeInstalledAt = [datetime]::Parse((Get-Content -LiteralPath $setupRuntimeMarker -Raw).Trim(),[Globalization.CultureInfo]::InvariantCulture,[Globalization.DateTimeStyles]::RoundtripKind)
        if ($setupOs.LastBootUpTime -lt $setupRuntimeInstalledAt) { $setupReboot = $true }
    }

    if ($setupReboot) {
        Write-Host ''
        Write-Host 'RESTART REQUIRED: Save your work and restart Windows yourself.' -ForegroundColor Yellow
        Write-Host 'After restarting, run Start-WSL-Setup.cmd as administrator again.' -ForegroundColor Yellow
        'Restart Windows, then rerun Start-WSL-Setup.cmd as administrator.' | Set-Content -LiteralPath (Join-Path $setupLogDir 'latest-status.txt') -Encoding UTF8
        exit 3010
    }

    Invoke-SetupWsl -WslArguments @('--version')
    Invoke-SetupWsl -WslArguments @('--set-default-version','2')
    $setupExisting = @(Get-SetupDistros | Where-Object DistributionName -eq $setupDistro)
    if ($setupExisting.Count -eq 0) {
        Write-Host ('Installing a separate distribution: ' + $setupDistro)
        Invoke-SetupWsl -WslArguments @('--install','-d','Ubuntu-24.04','--name',$setupDistro,'--no-launch','--web-download')
    } elseif ($setupExisting[0].Version -ne 2) {
        throw ('An existing ' + $setupDistro + ' uses WSL 1. Back it up before converting it; this script will not convert it automatically.')
    }

    Write-Host 'If Ubuntu asks, create your Linux username and password. Password input is invisible.' -ForegroundColor Yellow
    Invoke-SetupWsl -WslArguments @('-d',$setupDistro,'--exec','/bin/true')
    Invoke-SetupWsl -WslArguments @('--list','--verbose')
    Invoke-SetupWsl -WslArguments @('-d',$setupDistro,'--exec','uname','-r')
    Invoke-SetupWsl -WslArguments @('-d',$setupDistro,'--exec','/usr/lib/wsl/lib/nvidia-smi')
    Invoke-SetupWsl -WslArguments @('-d',$setupDistro,'--exec','getent','hosts','archive.ubuntu.com')

    'PASS: WSL 2 distribution starts, Linux kernel runs, NVIDIA GPU is visible, Ubuntu DNS resolves. CUDA Toolkit has not been installed or tested.' | Set-Content -LiteralPath (Join-Path $setupLogDir 'latest-status.txt') -Encoding UTF8
    Write-Host ''
    Write-Host 'PASS: WSL starts, NVIDIA GPU is visible, and Ubuntu DNS resolves.' -ForegroundColor Green
    Write-Host ('Open your CUDA environment with: wsl -d ' + $setupDistro + ' --cd ~')
    Write-Host 'CUDA Toolkit is not installed yet. Install only a compatible Toolkit, never a Linux NVIDIA driver in WSL.'
} catch {
    Write-Host ('SETUP STOPPED: ' + $_.Exception.Message) -ForegroundColor Red
    ('SETUP STOPPED: ' + $_.Exception.Message) | Set-Content -LiteralPath (Join-Path $setupLogDir 'latest-status.txt') -Encoding UTF8
    Save-SetupFailureEvents
    Write-Host ('Log: ' + $setupLog)
    exit 1
} finally {
    Stop-Transcript | Out-Null
}
