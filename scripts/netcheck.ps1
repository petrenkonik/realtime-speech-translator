<#
.SYNOPSIS
  A2: lists the TCP connections and UDP endpoints owned by one process, once a minute.

.DESCRIPTION
  Run in a second shell while the soak run is going (scripts/soak.py prints its pid).
  An offline demo has none: the check passes when every sample shows 0 connections and
  0 endpoints. It reads only; it changes nothing on the machine.

.EXAMPLE
  powershell -File scripts/netcheck.ps1 -ProcessId 1234 -Minutes 60
#>
param(
    [Parameter(Mandatory = $true)][int]$ProcessId,
    [double]$Minutes = 60,
    [int]$IntervalSeconds = 60
)

$samples = 0
$withTraffic = 0
$deadline = (Get-Date).AddMinutes($Minutes)
while ((Get-Date) -lt $deadline) {
    if (-not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) { break }
    $tcp = @(Get-NetTCPConnection -OwningProcess $ProcessId -ErrorAction SilentlyContinue)
    $udp = @(Get-NetUDPEndpoint -OwningProcess $ProcessId -ErrorAction SilentlyContinue)
    $samples++
    if ($tcp.Count -gt 0 -or $udp.Count -gt 0) {
        $withTraffic++
        "{0} tcp={1} udp={2}" -f (Get-Date -Format 'HH:mm:ss'), $tcp.Count, $udp.Count
        $tcp | Format-Table LocalAddress, LocalPort, RemoteAddress, RemotePort, State -AutoSize | Out-String
        $udp | Format-Table LocalAddress, LocalPort -AutoSize | Out-String
    }
    Start-Sleep -Seconds $IntervalSeconds
}
"A2 samples={0} samples_with_sockets={1} -> {2}" -f $samples, $withTraffic, $(if ($samples -gt 0 -and $withTraffic -eq 0) { 'PASS' } else { 'FAIL' })
