$ProfilePath = $PROFILE
$ProfileDir = Split-Path $ProfilePath

if (-not (Test-Path $ProfileDir)) {
    New-Item -ItemType Directory -Path $ProfileDir -Force | Out-Null
}

$ExePath = (Resolve-Path "$PSScriptRoot\..\target\release\laya-siro.exe").Path

$Content = @"
# =======================================================
# Laya-Siro Windows Low-Memory Decision Engine
# =======================================================
function siro {
    & "$ExePath" `$args
}
"@

[System.IO.File]::WriteAllText($ProfilePath, $Content)
Write-Host "Installed 'siro' command to $ProfilePath" -ForegroundColor Green
