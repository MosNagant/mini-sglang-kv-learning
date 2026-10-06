# Read-only host diagnostics plus one minimal WSL system-VM launch attempt.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Continue'
$diagIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
$diagPrincipal = New-Object Security.Principal.WindowsPrincipal($diagIdentity)
if ($diagIdentity.Name -match 'CodexSandbox' -or -not $diagPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host 'Run this script from your normal administrator PowerShell.' -ForegroundColor Yellow
    exit 1
}
$diagDir = Join-Path $PSScriptRoot ('logs\diagnostics-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $diagDir -Force | Out-Null
Start-Transcript -LiteralPath (Join-Path $diagDir 'diagnostics.log') -Force | Out-Null
$env:WSL_UTF8 = '1'

function Save-DiagEvents {
    param([string]$Phase)
    $diagChannels = @(
        'Microsoft-Windows-Hyper-V-Compute-Admin',
        'Microsoft-Windows-Hyper-V-Compute-Operational',
        'Microsoft-Windows-Hyper-V-Worker-Admin',
        'Microsoft-Windows-Hyper-V-Hypervisor-Admin',
        'Microsoft-Windows-Host-Network-Service-Admin'
    )
    foreach ($diagChannel in $diagChannels) {
        try {
            $diagEvents = @(Get-WinEvent -FilterHashtable @{LogName=$diagChannel;StartTime=(Get-Date).AddHours(-2)} -MaxEvents 60 -ErrorAction Stop)
            $diagEvents | Select-Object TimeCreated,Id,LevelDisplayName,ProviderName,Message,@{Name='EventXml';Expression={$_.ToXml()}} |
                ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $diagDir ($Phase + '-' + $diagChannel + '.json')) -Encoding UTF8
            Write-Host ($Phase + ': ' + $diagChannel + ': ' + $diagEvents.Count + ' events saved.')
        } catch {
            Write-Host ($Phase + ': ' + $diagChannel + ': ' + $_.Exception.Message)
        }
    }
}

try {
    Write-Host 'Collecting WSL/HCS diagnostics. No distributions will be deleted or reinstalled.' -ForegroundColor Cyan
    Write-Host ('Account: ' + $diagIdentity.Name)
    Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,LastBootUpTime | Format-List
    Get-CimInstance Win32_ComputerSystem | Select-Object HypervisorPresent,Manufacturer,Model | Format-List
    Get-Service WslService,vmcompute,hns | Format-List Name,Status,StartType
    foreach ($diagFeature in @('VirtualMachinePlatform','Microsoft-Windows-Subsystem-Linux','Microsoft-Hyper-V-Hypervisor','Microsoft-Hyper-V-Services')) {
        Get-WindowsOptionalFeature -Online -FeatureName $diagFeature | Select-Object FeatureName,State | Format-List
    }
    & "$env:SystemRoot\System32\bcdedit.exe" /enum '{current}'
    $diagConfig = Join-Path $env:USERPROFILE '.wslconfig'
    if (Test-Path -LiteralPath $diagConfig) {
        Write-Host ('Existing config: ' + $diagConfig)
        Get-Content -LiteralPath $diagConfig
    } else { Write-Host 'No user .wslconfig file.' }
    [pscustomobject]@{TEMP=$env:TEMP;TempExists=(Test-Path -LiteralPath $env:TEMP);LOCALAPPDATA=$env:LOCALAPPDATA;LocalAppDataExists=(Test-Path -LiteralPath $env:LOCALAPPDATA)} | Format-List
    $diagReg = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss'
    if (Test-Path $diagReg) {
        Get-ItemProperty $diagReg | Format-List
        Get-ChildItem $diagReg | ForEach-Object {
            $diagDistro = Get-ItemProperty $_.PSPath
            $diagDistro | Select-Object DistributionName,Version,State,BasePath,VhdFileName,DefaultUid | Format-List
            if ($diagDistro.BasePath) {
                Write-Host ('BasePath exists: ' + (Test-Path -LiteralPath $diagDistro.BasePath))
                Get-ChildItem -LiteralPath $diagDistro.BasePath -Force -ErrorAction Continue | Select-Object Name,Length,Attributes | Format-Table -AutoSize
            }
        }
    }
    Get-ChildItem -LiteralPath 'C:\Program Files\WSL','C:\Program Files\WSL\tools' -Force | Select-Object FullName,Length,Attributes | Format-Table -AutoSize
    foreach ($diagRegPath in @('HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Virtualization','HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Virtualization\Containers')) {
        if (Test-Path $diagRegPath) { Get-ItemProperty $diagRegPath | Format-List }
    }
    Get-Item -LiteralPath 'C:\Windows\System32\vmcompute.exe','C:\Windows\System32\vmwp.exe','C:\Windows\System32\drivers\vmbus.sys','C:\Windows\System32\drivers\vmswitch.sys','C:\Windows\System32\drivers\vid.sys' -ErrorAction Continue |
        Select-Object FullName,Length,@{Name='FileVersion';Expression={$_.VersionInfo.FileVersion}} | Format-Table -AutoSize
    & "$env:SystemRoot\System32\wsl.exe" --version
    & "$env:SystemRoot\System32\wsl.exe" --list --verbose
    & "$env:SystemRoot\System32\hcsdiag.exe" list
    Save-DiagEvents -Phase 'before'

    Write-Host 'Attempting a minimal WSL system-VM boot (maximum 60 seconds)...' -ForegroundColor Cyan
    $diagStdout = Join-Path $diagDir 'system-boot.stdout.txt'
    $diagStderr = Join-Path $diagDir 'system-boot.stderr.txt'
    $diagBoot = Start-Process -FilePath "$env:SystemRoot\System32\wsl.exe" -ArgumentList @('--system','--exec','/bin/true') -WindowStyle Hidden -PassThru -RedirectStandardOutput $diagStdout -RedirectStandardError $diagStderr
    if ($diagBoot.WaitForExit(60000)) {
        $diagBoot.WaitForExit()
        Write-Host ('System boot exit code: ' + $diagBoot.ExitCode)
    } else {
        Stop-Process -Id $diagBoot.Id -ErrorAction SilentlyContinue
        Write-Host 'System boot timed out; stopped this diagnostic launcher.'
    }
    Get-Content -LiteralPath $diagStdout,$diagStderr -ErrorAction Continue
    Save-DiagEvents -Phase 'after'
    Write-Host ('DIAGNOSTICS READY: ' + $diagDir) -ForegroundColor Green
    Write-Host 'Tell Codex that diagnostics are ready. The saved logs can be read directly in this workspace.'
} finally {
    Stop-Transcript | Out-Null
}
