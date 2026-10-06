# Read-only investigation of the observed DevManView device removals.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$cleanupIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
$cleanupPrincipal = New-Object Security.Principal.WindowsPrincipal($cleanupIdentity)
if ($cleanupIdentity.Name -match 'CodexSandbox' -or -not $cleanupPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host 'Run the cleanup investigation in your own administrator PowerShell.' -ForegroundColor Yellow
    exit 1
}
$cleanupLogDir = Join-Path $PSScriptRoot 'logs'
New-Item -ItemType Directory -Path $cleanupLogDir -Force | Out-Null
$cleanupStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$cleanupReport = [ordered]@{CheckedAt=(Get-Date).ToString('o'); Notes=@(); BootEvents=@(); ProcessMatches=@(); StartupTasks=@(); RunEntries=@(); ScriptMatches=@()}
$cleanupScriptPaths = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)

function Add-CleanupScriptPath {
    param([string]$Command)
    if (-not $Command) { return }
    $expandedCommand = [Environment]::ExpandEnvironmentVariables($Command)
    # Only inspect explicitly referenced local script files, never execute them.
    foreach ($match in [regex]::Matches($expandedCommand,'(?i)(?:"(?<p>[A-Z]:\\[^"\r\n]+\.(?:bat|cmd|ps1|vbs))"|(?<p>[A-Z]:\\[^\s"\r\n]+\.(?:bat|cmd|ps1|vbs)))(?=\s|$|[";])')) {
        [void]$cleanupScriptPaths.Add($match.Groups['p'].Value)
    }
}

Write-Host 'Collecting read-only startup evidence. No startup tasks or settings will be changed.' -ForegroundColor Cyan
try {
    $cleanupReport.BootEvents = @(Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-Kernel-General';Id=12;StartTime=(Get-Date).AddDays(-3)} -MaxEvents 20 -ErrorAction Stop |
        Select-Object TimeCreated,Id,Message)
} catch { $cleanupReport.Notes += ('Boot events: ' + $_.Exception.Message) }
try {
    Select-String -LiteralPath (Join-Path $env:SystemRoot 'INF\setupapi.dev.log') -SimpleMatch '[Delete Device - ROOT\STORVSP\0000]' -Context 0,5 |
        Out-String -Width 240 | Set-Content -LiteralPath (Join-Path $cleanupLogDir ('storage-deletions-' + $cleanupStamp + '.txt')) -Encoding UTF8
} catch { $cleanupReport.Notes += ('Device log: ' + $_.Exception.Message) }

# Two observed removal windows. Save only events for DevManView itself.
foreach ($cleanupStart in @([datetime]'2026-09-29T11:54:00',[datetime]'2026-09-29T16:44:00')) {
    try {
        $cleanupEvents = @(Get-WinEvent -FilterHashtable @{LogName='Security';Id=4688;StartTime=$cleanupStart;EndTime=$cleanupStart.AddMinutes(3)} -MaxEvents 10000 -ErrorAction Stop)
        foreach ($cleanupEvent in $cleanupEvents) {
            $cleanupXml = [xml]$cleanupEvent.ToXml()
            $cleanupFields = @{}
            foreach ($cleanupData in $cleanupXml.Event.EventData.Data) { $cleanupFields[$cleanupData.Name] = $cleanupData.'#text' }
            if ($cleanupFields.NewProcessName -match '(?i)\\DevManView(?:64)?\.exe$') {
                $cleanupReport.ProcessMatches += [pscustomobject]@{TimeCreated=$cleanupEvent.TimeCreated;Image=$cleanupFields.NewProcessName;CommandLine=$cleanupFields.CommandLine;ParentImage=$cleanupFields.ParentProcessName;ParentProcessId=$cleanupFields.ProcessId}
            }
        }
    } catch { $cleanupReport.Notes += ('Process audit ' + $cleanupStart.ToString('s') + ': ' + $_.Exception.Message) }
}

try {
    foreach ($cleanupTask in Get-ScheduledTask -ErrorAction Stop) {
        $cleanupActions = @($cleanupTask.Actions | ForEach-Object { [pscustomobject]@{Execute=$_.Execute;Arguments=$_.Arguments;WorkingDirectory=$_.WorkingDirectory} })
        $cleanupActionText = @($cleanupActions | ForEach-Object { $_.Execute + ' ' + $_.Arguments }) -join ' '
        $cleanupTriggers = @($cleanupTask.Triggers | ForEach-Object { $_.CimClass.CimClassName })
        if ($cleanupActionText -match '(?i)DevManView' -or @($cleanupTriggers | Where-Object { $_ -match 'BootTrigger|LogonTrigger' }).Count -gt 0) {
            $cleanupReport.StartupTasks += [pscustomobject]@{TaskPath=$cleanupTask.TaskPath;TaskName=$cleanupTask.TaskName;State=[string]$cleanupTask.State;Triggers=$cleanupTriggers;Actions=$cleanupActions}
        }
        foreach ($cleanupAction in $cleanupActions) { Add-CleanupScriptPath ('"' + $cleanupAction.Execute + '" ' + $cleanupAction.Arguments) }
    }
} catch { $cleanupReport.Notes += ('Scheduled tasks: ' + $_.Exception.Message) }

foreach ($cleanupHive in @('HKLM:','HKCU:')) {
    foreach ($cleanupRunSuffix in @('SOFTWARE\Microsoft\Windows\CurrentVersion\Run','SOFTWARE\Microsoft\Windows\CurrentVersion\RunOnce','SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Run')) {
        $cleanupKey = $cleanupHive + '\' + $cleanupRunSuffix
        try {
            if (Test-Path -LiteralPath $cleanupKey) {
                $cleanupProperties = (Get-ItemProperty -LiteralPath $cleanupKey).PSObject.Properties | Where-Object { $_.Name -notlike 'PS*' }
                foreach ($cleanupProperty in $cleanupProperties) {
                    $cleanupReport.RunEntries += [pscustomobject]@{Key=$cleanupKey;Name=$cleanupProperty.Name;Command=[string]$cleanupProperty.Value}
                    Add-CleanupScriptPath ([string]$cleanupProperty.Value)
                }
            }
        } catch { $cleanupReport.Notes += ('Startup key ' + $cleanupKey + ': ' + $_.Exception.Message) }
    }
}
foreach ($cleanupFolder in @('C:\',(Join-Path $env:ProgramData 'Microsoft\Windows\Start Menu\Programs\Startup'),(Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\Startup'))) {
    if (Test-Path -LiteralPath $cleanupFolder) {
        Get-ChildItem -LiteralPath $cleanupFolder -File -ErrorAction SilentlyContinue | Where-Object { $_.Extension -in @('.bat','.cmd','.ps1','.vbs') } |
            ForEach-Object { [void]$cleanupScriptPaths.Add($_.FullName) }
    }
}
foreach ($cleanupScript in $cleanupScriptPaths) {
    try {
        if ((Test-Path -LiteralPath $cleanupScript -PathType Leaf) -and (Get-Item -LiteralPath $cleanupScript).Length -lt 2097152) {
            foreach ($cleanupMatch in Select-String -LiteralPath $cleanupScript -Pattern '(?i)DevManView') {
                $cleanupReport.ScriptMatches += [pscustomobject]@{Path=$cleanupScript;LineNumber=$cleanupMatch.LineNumber;Line=$cleanupMatch.Line}
            }
        }
    } catch { $cleanupReport.Notes += ('Script read ' + $cleanupScript + ': ' + $_.Exception.Message) }
}
$cleanupReportPath = Join-Path $cleanupLogDir ('device-cleanup-investigation-' + $cleanupStamp + '.json')
$cleanupReport | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $cleanupReportPath -Encoding UTF8
Write-Host ('Process matches: ' + $cleanupReport.ProcessMatches.Count + '; script matches: ' + $cleanupReport.ScriptMatches.Count + '; startup tasks for review: ' + $cleanupReport.StartupTasks.Count)
if ($cleanupReport.ProcessMatches.Count -gt 0) { $cleanupReport.ProcessMatches | Format-List }
if ($cleanupReport.ScriptMatches.Count -gt 0) { $cleanupReport.ScriptMatches | Format-Table -AutoSize }
Write-Host ('INVESTIGATION SAVED: ' + $cleanupReportPath)
Write-Host 'An empty process result can mean process auditing was not enabled; it does not rule out a startup caller.'
exit 0
