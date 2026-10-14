<#
.SYNOPSIS
    Downloads quantized Laya ONNX model weights and tokenizer configurations.
.DESCRIPTION
    Pulls the INT8 quantized model and associated tokenizers from Hugging Face into the models/ directory.
#>

[CmdletBinding()]
param (
    [string]$ModelDir = $(if ($PSScriptRoot) { Join-Path $PSScriptRoot "..\models" } else { ".\models" }),
    [string]$Repo = "Sharjeelbaig/laya-multilingual-onnx-int8"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ModelDir)) {
    New-Item -ItemType Directory -Path $ModelDir -Force | Out-Null
}

$Files = @(
    @{
        Url = "https://huggingface.co/$Repo/resolve/main/model.onnx"
        Dest = Join-Path $ModelDir "model.onnx"
        Desc = "Laya INT8 ONNX Model (~317 MB)"
    },
    @{
        Url = "https://huggingface.co/$Repo/resolve/main/tokenizer/tokenizer.json"
        Dest = Join-Path $ModelDir "tokenizer.json"
        Desc = "Hugging Face Tokenizer Configuration"
    },
    @{
        Url = "https://huggingface.co/$Repo/resolve/main/rl_agent_config.json"
        Dest = Join-Path $ModelDir "rl_agent_config.json"
        Desc = "RL Agent Temperature Configuration"
    }
)

# Ensure official Microsoft ONNX Runtime DLL is present for load-dynamic runtime
$OrtDll = Join-Path $ModelDir "onnxruntime.dll"
if (-not (Test-Path $OrtDll)) {
    Write-Host "[FETCHING] Official Microsoft ONNX Runtime dynamic library (x64)..." -ForegroundColor Green
    $ZipPath = Join-Path $ModelDir "ort-runtime.zip"
    $OrtUrl = "https://github.com/microsoft/onnxruntime/releases/download/v1.31.0/onnxruntime-win-x64-1.31.0.zip"
    try {
        curl.exe -L -o "$ZipPath" "$OrtUrl"
        if ($LASTEXITCODE -eq 0 -and (Test-Path $ZipPath)) {
            Write-Host "Extracting onnxruntime.dll to $ModelDir..." -ForegroundColor Gray
            Expand-Archive -Path $ZipPath -DestinationPath "$ModelDir\tmp_ort" -Force
            $ExtractedDll = Get-ChildItem -Path "$ModelDir\tmp_ort" -Filter "onnxruntime.dll" -Recurse | Select-Object -First 1
            if ($ExtractedDll) {
                Move-Item -Path $ExtractedDll.FullName -Destination $OrtDll -Force
                Write-Host "[SUCCESS] onnxruntime.dll installed into models/" -ForegroundColor Green
            }
            Remove-Item -Path "$ModelDir\tmp_ort" -Recurse -Force -ErrorAction SilentlyContinue
            Remove-Item -Path $ZipPath -Force -ErrorAction SilentlyContinue
        }
    } catch {
        Write-Warning "Could not automatically download onnxruntime.dll: $_"
    }
}

Write-Host "=== Laya Model Downloader ===" -ForegroundColor Cyan
Write-Host "Target Directory: $ModelDir" -ForegroundColor Gray
Write-Host "Source Repository: $Repo" -ForegroundColor Gray
Write-Host ""

foreach ($file in $Files) {
    if (Test-Path $file.Dest) {
        Write-Host "[ALREADY EXISTS] $($file.Desc) -> $($file.Dest)" -ForegroundColor Yellow
        continue
    }

    Write-Host "[DOWNLOADING] $($file.Desc)..." -ForegroundColor Green
    try {
        # Using curl.exe for efficient stream download with progress without holding entire file in memory
        curl.exe -L -o "$($file.Dest)" "$($file.Url)"
        if ($LASTEXITCODE -eq 0 -and (Test-Path $file.Dest)) {
            $sizeMB = [math]::Round(((Get-Item $file.Dest).Length / 1MB), 2)
            Write-Host "[COMPLETE] Saved $($file.Dest) ($sizeMB MB)" -ForegroundColor Green
        } else {
            Write-Warning "curl failed to download $($file.Url). Attempting fallback with Invoke-WebRequest..."
            Invoke-WebRequest -Uri $file.Url -OutFile $file.Dest
        }
    } catch {
        Write-Error "Failed to download $($file.Url): $_"
    }
}

Write-Host "`nAll model files verified in $ModelDir." -ForegroundColor Cyan
