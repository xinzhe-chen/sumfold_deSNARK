# Distributed Benchmark Runner — 4 parties on localhost (PowerShell)
#
# Usage:
#   .\scripts\run_bench.ps1 small          # nv = 10..14
#   .\scripts\run_bench.ps1 large          # nv = 22..26
#   .\scripts\run_bench.ps1 small 3        # nv = 10..14, 3 repetitions
#
# Output: CSV on stdout (master), per-party logs in target\bench_logs\

param(
    [Parameter(Position=0)]
    [string]$Profile,

    [Parameter(Position=1)]
    [int]$Reps = 1
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$BIN  = Join-Path $Root "target\release\examples\dist_bench.exe"
$LogDir = Join-Path $Root "target\bench_logs"

# ─── Build if needed ──────────────────────────────────────────────
if (-not (Test-Path $BIN)) {
    Write-Host "Building dist_bench (release)..." -ForegroundColor Yellow
    Push-Location $Root
    cargo build --example dist_bench -p deSnark --release
    Pop-Location
    if ($LASTEXITCODE -ne 0) { Write-Error "Build failed"; exit 1 }
    Write-Host "Build complete." -ForegroundColor Green
}

# ─── Parse profile ────────────────────────────────────────────────
switch ($Profile) {
    "small" {
        $Config = Join-Path $Root "deSnark\examples\bench_small.toml"
        $NvMin = 10; $NvMax = 14
    }
    "large" {
        $Config = Join-Path $Root "deSnark\examples\bench_large.toml"
        $NvMin = 22; $NvMax = 26
    }
    default {
        Write-Host "Usage:"
        Write-Host "  .\scripts\run_bench.ps1 small [reps]    # nv=10..14, M=8, K=4"
        Write-Host "  .\scripts\run_bench.ps1 large [reps]    # nv=22..26, M=8, K=4"
        exit 0
    }
}

if (-not (Test-Path $Config)) { Write-Error "Config not found: $Config"; exit 1 }

# ─── Kill leftover processes ──────────────────────────────────────
Get-Process -Name dist_bench -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

# ─── Setup logs ───────────────────────────────────────────────────
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$CsvFile = Join-Path $LogDir "bench_nv${NvMin}_${NvMax}_${Timestamp}.csv"

Write-Host "Benchmark: nv=${NvMin}..${NvMax}, reps=${Reps}" -ForegroundColor Yellow
Write-Host "Config: $Config" -ForegroundColor Yellow

# ─── Start workers (parties 1-3) in background ───────────────────
$WorkerProcs = @()
$env:RUST_LOG = "error"
for ($i = 1; $i -le 3; $i++) {
    $logOut = Join-Path $LogDir "p${i}.log"
    $proc = Start-Process -FilePath $BIN `
        -ArgumentList "--party",$i,"--nv-min",$NvMin,"--nv-max",$NvMax,"--reps",$Reps,$Config `
        -RedirectStandardOutput $logOut `
        -RedirectStandardError (Join-Path $LogDir "p${i}_err.log") `
        -NoNewWindow -PassThru
    $WorkerProcs += $proc
}
Start-Sleep -Seconds 2

# ─── Run master (party 0) — CSV to stdout ─────────────────────────
Write-Host "Running master..." -ForegroundColor Green
$masterLog = Join-Path $LogDir "p0.log"
$rawFile   = Join-Path $LogDir "p0_raw.log"

# Run master, capture all stdout (CSV + print-trace) to raw file, stderr to log
& $BIN --party 0 --nv-min $NvMin --nv-max $NvMax --reps $Reps $Config 2> $masterLog > $rawFile

# Extract only CSV lines (header + data rows) from the raw output
$csvHeader = "nv,M,K,setup_ms,prover_ms,verifier_ms,proof_bytes,comm_sent,comm_recv"
$csvHeader | Out-File -FilePath $CsvFile -Encoding utf8
$csvHeader
Get-Content $rawFile | Where-Object { $_ -match '^\d+,' } | ForEach-Object {
    $_ | Out-File -FilePath $CsvFile -Append -Encoding utf8
    $_
}

# ─── Wait for workers ─────────────────────────────────────────────
foreach ($proc in $WorkerProcs) {
    $proc.WaitForExit(30000) | Out-Null
}

Write-Host ""
Write-Host "Done. Results saved to: $CsvFile" -ForegroundColor Green
