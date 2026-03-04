# ═══════════════════════════════════════════════════════════════════════
# Interactive Distributed Benchmark Runner — sumfold_deSNARK (PowerShell)
#
# Prompts the user for benchmark parameters, generates the required
# config files, builds the binary, and runs the distributed benchmark.
#
# Parameters:
#   nMIN / nMAX  — range of n where nv = 2^n (log_num_constraints)
#   k            — Number of Sub_Provers (must be a power of 2)
#   M            — Number of instances  (must be a power of 2)
#
# Output: CSV file saved to target\bench_logs\
# ═══════════════════════════════════════════════════════════════════════

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$BIN  = Join-Path $Root "target\release\examples\dist_bench.exe"
$LogDir = Join-Path $Root "target\bench_logs"
$TmpDir = Join-Path $Root "target\bench_tmp"

# ─── Helper functions ────────────────────────────────────────────────

function Test-PowerOf2([int]$n) {
    return ($n -gt 0) -and (($n -band ($n - 1)) -eq 0)
}

function Get-Log2([int]$n) {
    $log = 0
    while ($n -gt 1) { $n = $n -shr 1; $log++ }
    return $log
}

# ─── Prompt for parameters ──────────────────────────────────────────

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  sumfold_deSNARK — Interactive Benchmark Runner" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

$NvMin = [int](Read-Host "  nMIN (min n, where nv = 2^n)")
$NvMax = [int](Read-Host "  nMAX (max n, where nv = 2^n)")
$K     = [int](Read-Host "  k    (Number of Sub_Provers, power of 2)")
$M     = [int](Read-Host "  M    (Number of instances, power of 2)")
Write-Host ""

# ─── Validate inputs ────────────────────────────────────────────────

if ($NvMin -gt $NvMax) { Write-Error "nMIN ($NvMin) must be <= nMAX ($NvMax)"; exit 1 }
if (-not (Test-PowerOf2 $K)) { Write-Error "k ($K) must be a power of 2"; exit 1 }
if (-not (Test-PowerOf2 $M)) { Write-Error "M ($M) must be a power of 2"; exit 1 }

$LogK = Get-Log2 $K
$LogM = Get-Log2 $M

if ($NvMin -lt $LogK) {
    Write-Error "nMIN ($NvMin) must be >= log2(k) = $LogK (constraints per party must be >= 1)"
    exit 1
}

Write-Host "Parameters:" -ForegroundColor Green
Write-Host "  nv range    : $NvMin .. $NvMax"
Write-Host "  k (Sub_Provers): $K  (log2 = $LogK)"
Write-Host "  M (instances)  : $M  (log2 = $LogM)"
Write-Host ""

# ─── Generate hosts file ────────────────────────────────────────────

New-Item -ItemType Directory -Path $TmpDir -Force | Out-Null
$HostsFile = Join-Path $TmpDir "hosts_${K}.txt"
$BasePort = 12350

$hostLines = @()
for ($i = 0; $i -lt $K; $i++) {
    $hostLines += "127.0.0.1:$($BasePort + $i)"
}
$hostLines | Out-File -FilePath $HostsFile -Encoding utf8
Write-Host "Generated hosts file: $HostsFile  ($K parties on ports $BasePort..$($BasePort + $K - 1))" -ForegroundColor Green

# ─── Generate TOML config ───────────────────────────────────────────

$ConfigFile = Join-Path $TmpDir "bench_config.toml"
@"
# Auto-generated benchmark config
# nv range: ${NvMin}..${NvMax}, M=${M}, K=${K}

[config]
log_num_instances = ${LogM}
log_num_constraints = ${NvMin}
gate_type = "vanilla"
log_num_parties = ${LogK}
srs_path = "srs_interactive_nv${NvMax}.params"

[network]
hosts_file = "${HostsFile}"
"@ | Out-File -FilePath $ConfigFile -Encoding utf8
# Fix path separators for TOML (use forward slashes)
(Get-Content $ConfigFile) -replace '\\','/' | Set-Content $ConfigFile
Write-Host "Generated TOML config: $ConfigFile" -ForegroundColor Green
Write-Host ""

# ─── Resolve nightly toolchain ────────────────────────────────────────
# A standalone stable rustc may shadow the rustup shim on Windows.
# Prepend the nightly bin directory so cargo picks up the correct compiler.

$NightlyRustc = $null
try { $NightlyRustc = (& rustup which rustc --toolchain nightly-2026-02-22 2>$null) } catch { }
if ($NightlyRustc -and (Test-Path $NightlyRustc)) {
    $NightlyBin = Split-Path -Parent $NightlyRustc
    $env:PATH = "$NightlyBin;$($env:PATH)"
    $env:RUSTC = "$NightlyRustc"
    Write-Host "Using nightly rustc: $(& $NightlyRustc --version)" -ForegroundColor Green
} else {
    Write-Host "Warning: could not resolve nightly-2026-02-22 toolchain, using default" -ForegroundColor Yellow
}

# ─── Build ───────────────────────────────────────────────────────────

Write-Host "Building dist_bench (release)..." -ForegroundColor Yellow
Push-Location $Root
cargo build --example dist_bench -p deSnark --release
if ($LASTEXITCODE -ne 0) { Write-Error "Build failed"; exit 1 }
Pop-Location
Write-Host "Build complete." -ForegroundColor Green
Write-Host ""

# ─── Kill leftover processes ─────────────────────────────────────────

Get-Process -Name dist_bench -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

# ─── Setup logs ──────────────────────────────────────────────────────

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$CsvFile = Join-Path $LogDir "bench_nv${NvMin}_${NvMax}_k${K}_M${M}_${Timestamp}.csv"
$RawFile = Join-Path $LogDir "p0_raw.log"

Write-Host "Starting benchmark: nv=${NvMin}..${NvMax}, K=${K}, M=${M}" -ForegroundColor Yellow
Write-Host ""

# ─── CWD into target\bench_tmp so SRS cache files stay inside target\ ─
Push-Location $TmpDir

# ─── Start workers (parties 1..K-1) in background ────────────────────

$WorkerProcs = @()
$env:RUST_LOG = "error"
for ($i = 1; $i -lt $K; $i++) {
    $logOut = Join-Path $LogDir "p${i}.log"
    $logErr = Join-Path $LogDir "p${i}_err.log"
    $proc = Start-Process -FilePath $BIN `
        -ArgumentList "--party",$i,"--nv-min",$NvMin,"--nv-max",$NvMax,$ConfigFile `
        -WorkingDirectory $TmpDir `
        -RedirectStandardOutput $logOut `
        -RedirectStandardError $logErr `
        -NoNewWindow -PassThru
    $WorkerProcs += $proc
}
Start-Sleep -Seconds 2

# ─── Run master (party 0) ────────────────────────────────────────────

Write-Host "Running master (party 0)..." -ForegroundColor Green
$masterLog = Join-Path $LogDir "p0.log"

& $BIN --party 0 --nv-min $NvMin --nv-max $NvMax $ConfigFile 2> $masterLog > $RawFile

# ─── Extract CSV ─────────────────────────────────────────────────────

$csvHeader = "nv,M,K,setup_ms,prover_ms,verifier_ms,proof_bytes,comm_sent,comm_recv,avg_cpu_pct,peak_rss_mb"
$csvHeader | Out-File -FilePath $CsvFile -Encoding utf8
$csvHeader

Get-Content $RawFile | Where-Object { $_ -match '^\d+,' } | ForEach-Object {
    $_ | Out-File -FilePath $CsvFile -Append -Encoding utf8
    $_
}

# ─── Wait for workers ────────────────────────────────────────────────

foreach ($proc in $WorkerProcs) {
    $proc.WaitForExit(60000) | Out-Null
}

Pop-Location

Write-Host ""
Write-Host "========================================================" -ForegroundColor Green
Write-Host "  Benchmark complete!" -ForegroundColor Green
Write-Host "  Results: $CsvFile" -ForegroundColor Green
Write-Host "  Logs:    $LogDir\p*.log" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
