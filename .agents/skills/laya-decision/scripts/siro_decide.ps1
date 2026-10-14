[CmdletBinding()]
param (
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Prompt,

    [switch]$Execute
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path "$ScriptDir\..\..\..\..").Path
$Binary = Join-Path $RepoRoot "target\release\laya-siro.exe"

if (-not (Test-Path $Binary)) {
    [PSCustomObject]@{
        action         = "error"
        confidence     = 0.0
        escalate       = $true
        mutating       = $false
        latency_ms     = 0.0
        routing_source = "WRAPPER"
        status         = "BINARY_NOT_FOUND"
        error          = "Binary not found at $Binary. Please run 'cargo build --release -j 2' first."
        should_ask_user= $true
    } | ConvertTo-Json
    exit 1
}

$Arguments = @("--json")
if ($Execute) {
    $Arguments += "--execute"
}
$Arguments += $Prompt

$Process = Start-Process -FilePath $Binary -ArgumentList $Arguments -NoNewWindow -Wait -PassThru -RedirectStandardOutput "$env:TEMP\siro_out.json"

if (Test-Path "$env:TEMP\siro_out.json") {
    $RawJson = Get-Content "$env:TEMP\siro_out.json" -Raw
    Remove-Item "$env:TEMP\siro_out.json" -Force -ErrorAction SilentlyContinue

    try {
        $Obj = $RawJson | ConvertFrom-Json
        # Add safety flag for AI agents: always require user confirmation for mutating actions
        $ShouldAsk = ($Obj.confidence -lt 0.75) -or ($Obj.escalate -eq $true) -or ($Obj.mutating -eq $true)
        $Obj | Add-Member -NotePropertyName "should_ask_user" -NotePropertyValue $ShouldAsk -Force
        return ($Obj | ConvertTo-Json -Depth 5)
    } catch {
        return $RawJson
    }
}
