<#
.SYNOPSIS
  A1: records file-system activity with Process Monitor and counts disk writes by one process.

.DESCRIPTION
  Run in a second, elevated shell while the soak run is going (scripts/soak.py prints its
  pid). Process Monitor sees native code too (onnxruntime, PortAudio, CTranslate2, SAPI),
  which the Python audit hooks cannot.

  Counted as a write, for the given pid only and only for paths on a drive letter:
    - WriteFile
    - CreateFile whose desired access asks for write/append/delete, or whose disposition
      creates or overwrites a file
    - SetDispositionInformationFile (delete), SetRenameInformationFile,
      SetEndOfFileInformationFile, SetAllocationInformationFile
  Reads, DLL loads, pipes and console handles are not writes. Registry writes are listed
  for information but do not fail the check.

  The capture file and its CSV export are written by Process Monitor and this script, not by
  the demo, into procmon/ (ignored by git). Delete them after reading the report.

  Needs Process Monitor from Sysinternals:
  https://learn.microsoft.com/sysinternals/downloads/procmon

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts/procmon_check.ps1 -ProcessId 1234 -Minutes 60
#>
param(
    [Parameter(Mandatory = $true)][int]$ProcessId,
    [double]$Minutes = 60,
    [string]$ProcmonPath = '',
    [string]$OutDir = ''
)

$ErrorActionPreference = 'Stop'

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    throw 'Process Monitor needs an elevated shell: start PowerShell with "Run as administrator".'
}

if (-not $ProcmonPath) {
    $candidates = @('Procmon64.exe', 'Procmon.exe') | ForEach-Object {
        (Get-Command $_ -ErrorAction SilentlyContinue).Source
        Join-Path $PSScriptRoot $_
    }
    $ProcmonPath = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
}
if (-not $ProcmonPath -or -not (Test-Path $ProcmonPath)) {
    throw ('Process Monitor not found. Download it from ' +
        'https://learn.microsoft.com/sysinternals/downloads/procmon and pass -ProcmonPath.')
}

if (-not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) {
    throw "No process with pid $ProcessId. Start the soak run first; it prints its pid."
}

if (-not $OutDir) { $OutDir = Join-Path (Split-Path $PSScriptRoot -Parent) 'procmon' }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$pml = Join-Path $OutDir "a1-$stamp.pml"
$csv = Join-Path $OutDir "a1-$stamp.csv"

"Starting Process Monitor for {0} min, pid {1}" -f $Minutes, $ProcessId
& $ProcmonPath /AcceptEula /Quiet /Minimized /BackingFile $pml
& $ProcmonPath /WaitForIdle

$deadline = (Get-Date).AddMinutes($Minutes)
$exitedEarly = $false
while ((Get-Date) -lt $deadline) {
    if (-not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) { $exitedEarly = $true; break }
    Start-Sleep -Seconds 5
}

& $ProcmonPath /Terminate
Start-Sleep -Seconds 3
"Exporting the capture to CSV (this can take a few minutes for a long run)"
& $ProcmonPath /AcceptEula /Quiet /OpenLog $pml /SaveAs $csv
$waited = 0
while (-not (Test-Path $csv) -and $waited -lt 600) { Start-Sleep -Seconds 2; $waited += 2 }
& $ProcmonPath /Terminate
if (-not (Test-Path $csv)) { throw "Process Monitor did not write $csv." }

$writeOps = @(
    'WriteFile', 'SetDispositionInformationFile', 'SetDispositionInformationEx',
    'SetRenameInformationFile', 'SetRenameInformationEx',
    'SetEndOfFileInformationFile', 'SetAllocationInformationFile'
)
$createWrite = 'Desired Access: [^,]*(Write|Append|Delete)|Disposition: (Create|Overwrite|OverwriteIf|Supersede)'
$pidText = [string]$ProcessId

$events = 0
$writes = New-Object System.Collections.Generic.List[object]
$registry = New-Object System.Collections.Generic.List[object]
Import-Csv -Path $csv | Where-Object { $_.PID -eq $pidText } | ForEach-Object {
    $events++
    $op = $_.Operation
    $onDisk = $_.Path -match '^[A-Za-z]:\\'
    if ($onDisk -and ($writeOps -contains $op -or ($op -eq 'CreateFile' -and $_.Detail -match $createWrite))) {
        $writes.Add($_)
    } elseif ($op -like 'RegSetValue*' -or $op -eq 'RegCreateKey' -or $op -like 'RegDelete*') {
        $registry.Add($_)
    }
}

if ($writes.Count -gt 0) {
    'Disk writes by pid {0}:' -f $ProcessId
    $writes | Group-Object Operation, Path | Sort-Object Count -Descending |
        Select-Object Count, Name -First 50 | Format-Table -AutoSize | Out-String
}
if ($registry.Count -gt 0) {
    'Registry writes (information only): {0}' -f $registry.Count
    $registry | Group-Object Operation, Path | Sort-Object Count -Descending |
        Select-Object Count, Name -First 20 | Format-Table -AutoSize | Out-String
}

$verdict = if ($events -eq 0) { 'NOT RUN (no events for this pid; wrong pid or capture failed)' }
           elseif ($writes.Count -eq 0) { 'PASS' } else { 'FAIL' }
'A1 pid={0} events={1} disk_writes={2} registry_writes={3} exited_early={4} -> {5}' -f `
    $ProcessId, $events, $writes.Count, $registry.Count, $exitedEarly, $verdict
'Capture: {0}' -f $pml
'Export:  {0}' -f $csv
