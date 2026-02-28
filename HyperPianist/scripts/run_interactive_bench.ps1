# ═══════════════════════════════════════════════════════════════════════
# Interactive Distributed Benchmark Runner — HyperPianist (PowerShell)
#
# Prompts the user for benchmark parameters, generates the required
# config files, builds the binary, and runs the distributed benchmark.
#
# Parameters:
#   nMIN / nMAX  — range of n where nv = 2^n (num_vars)
#   k            — Number of Sub_Provers (must be a power of 2)
#   M            — Number of instances (used as repetitions per nv)
#
# Output: CSV file saved to HyperPianist\target\bench_logs\
#         Format matches sumfold_deSNARK for direct comparison:
#         nv,M,K,setup_ms,prover_ms,verifier_ms,proof_bytes,comm_sent,comm_recv
# ═══════════════════════════════════════════════════════════════════════

$ErrorActionPreference = "Stop"
$HpRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$BIN    = Join-Path $HpRoot "target\release\examples\hyperpianist-bench.exe"
$LogDir = Join-Path $HpRoot "target\bench_logs"
$TmpDir = Join-Path $HpRoot "target\bench_tmp"

# ─── Helper functions ────────────────────────────────────────────────

function Test-PowerOf2([int]$n) {
    return ($n -gt 0) -and (($n -band ($n - 1)) -eq 0)
}

function Get-Log2([int]$n) {
    $log = 0
    while ($n -gt 1) { $n = $n -shr 1; $log++ }
    return $log
}

# Parse HyperPianist master output into metric values.
# Returns: setup_ms, prover_ms, verifier_ms, proof_bytes, comm_sent, comm_recv
function Parse-HpOutput($RawFile) {
    $setupUs = 0; $proverUs = 0; $verifierUs = 0; $proofBytes = 0
    $commSent = 0; $commRecv = 0

    foreach ($line in (Get-Content $RawFile)) {
        if ($line -match 'key extraction.*?(\d+)\s+us') { $setupUs = [long]$Matches[1] }
        if ($line -match '^proving for.*?(\d+)\s+us') { $proverUs = [long]$Matches[1] }
        if ($line -match '^verifying for.*?(\d+)\s+us') { $verifierUs = [long]$Matches[1] }
        if ($line -match 'compressed:\s*(\d+)\s+bytes' -and $line -notmatch 'uncompressed') {
            $proofBytes = [long]$Matches[1]
        }
        if ($line -match 'bytes_sent:\s*(\d+)') { $commSent += [long]$Matches[1] }
        if ($line -match 'bytes_recv:\s*(\d+)') { $commRecv += [long]$Matches[1] }
    }

    return @{
        SetupMs    = [math]::Round($setupUs / 1000.0, 3)
        ProverMs   = [math]::Round($proverUs / 1000.0, 3)
        VerifierMs = [math]::Round($verifierUs / 1000.0, 3)
        ProofBytes = $proofBytes
        CommSent   = $commSent
        CommRecv   = $commRecv
    }
}

# ─── Prompt for parameters ──────────────────────────────────────────

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  HyperPianist — Interactive Benchmark Runner" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

$NvMin = [int](Read-Host "  nMIN (min n, where nv = 2^n)")
$NvMax = [int](Read-Host "  nMAX (max n, where nv = 2^n)")
$K     = [int](Read-Host "  k    (Number of Sub_Provers, power of 2)")
$M     = [int](Read-Host "  M    (Number of instances / repetitions)")
Write-Host ""

# ─── Validate inputs ────────────────────────────────────────────────

if ($NvMin -gt $NvMax) { Write-Error "nMIN ($NvMin) must be <= nMAX ($NvMax)"; exit 1 }
if (-not (Test-PowerOf2 $K)) { Write-Error "k ($K) must be a power of 2"; exit 1 }
if ($M -lt 1) { Write-Error "M must be >= 1"; exit 1 }

$LogK = Get-Log2 $K

if ($NvMin -le $LogK) {
    Write-Error "nMIN ($NvMin) must be > log2(k) = $LogK (each party needs at least 2 constraints)"
    exit 1
}

Write-Host "Parameters:" -ForegroundColor Green
Write-Host "  nv range      : $NvMin .. $NvMax"
Write-Host "  k (Sub_Provers): $K  (log2 = $LogK)"
Write-Host "  M (instances)  : $M"
Write-Host ""

# ─── Generate hosts file ────────────────────────────────────────────

New-Item -ItemType Directory -Path $TmpDir -Force | Out-Null
$HostsFile = Join-Path $TmpDir "hosts_${K}.txt"
$BasePort = 8000

$hostLines = @()
for ($i = 0; $i -lt $K; $i++) {
    $hostLines += "127.0.0.1:$($BasePort + $i)"
}
$hostLines | Out-File -FilePath $HostsFile -Encoding utf8
Write-Host "Generated hosts file: $HostsFile  ($K parties on ports $BasePort..$($BasePort + $K - 1))" -ForegroundColor Green
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

Write-Host "Building hyperpianist-bench (release)..." -ForegroundColor Yellow
Push-Location $HpRoot
cargo build --example hyperpianist-bench --release
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    Write-Error "Build failed"
    exit 1
}
Pop-Location
Write-Host "Build complete." -ForegroundColor Green
Write-Host ""

# ─── Setup logs ──────────────────────────────────────────────────────

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$CsvFile = Join-Path $LogDir "bench_nv${NvMin}_${NvMax}_k${K}_M${M}_${Timestamp}.csv"

Write-Host "Starting benchmark: nv=${NvMin}..${NvMax}, K=${K}, M=${M} reps" -ForegroundColor Yellow
Write-Host ""

# Write CSV header (same columns as sumfold_deSNARK for direct comparison)
$csvHeader = "nv,M,K,setup_ms,prover_ms,verifier_ms,proof_bytes,comm_sent,comm_recv"
$csvHeader | Out-File -FilePath $CsvFile -Encoding utf8
$csvHeader

# ─── CWD into target\bench_tmp so SRS cache files stay inside target\ ─
Push-Location $TmpDir

# ─── Run benchmarks ──────────────────────────────────────────────────

for ($nv = $NvMin; $nv -le $NvMax; $nv++) {
    $constraints = [math]::Pow(2, $nv)
    Write-Host "──────────────────────────────────────────────────" -ForegroundColor Cyan
    Write-Host "  nv = $nv  (constraints = 2^$nv = $constraints)" -ForegroundColor Cyan
    Write-Host "──────────────────────────────────────────────────" -ForegroundColor Cyan

    # Accumulators — sum across M runs (not averaged) for direct comparison
    # with sumfold_deSNARK which reports total cost for M instances in one batch.
    #
    # Setup is counted ONCE (first rep only): SRS generation and key extraction
    # are one-time costs for both systems.
    # Prover/verifier/comm/proof are summed: M sequential proofs vs 1 batched.
    $totSetup = 0.0; $totProver = 0.0; $totVerifier = 0.0
    $totProofBytes = 0; $totCommSent = 0; $totCommRecv = 0

    for ($rep = 1; $rep -le $M; $rep++) {
        Write-Host "  Repetition ${rep}/${M}..." -ForegroundColor Yellow

        # Clean up leftover processes
        Get-Process -Name "hyperpianist-bench" -ErrorAction SilentlyContinue | Stop-Process -Force
        Start-Sleep -Seconds 1

        # Start workers (parties 1..K-1) in background
        $WorkerProcs = @()
        $env:RUST_LOG = "error"
        for ($i = 1; $i -lt $K; $i++) {
            $logOut = Join-Path $LogDir "hp_p${i}_nv${nv}_rep${rep}.log"
            $logErr = Join-Path $LogDir "hp_p${i}_nv${nv}_rep${rep}_err.log"
            $proc = Start-Process -FilePath $BIN `
                -ArgumentList $i,$HostsFile,$nv `
                -WorkingDirectory $TmpDir `
                -RedirectStandardOutput $logOut `
                -RedirectStandardError $logErr `
                -NoNewWindow -PassThru
            $WorkerProcs += $proc
        }
        Start-Sleep -Seconds 2

        # Run master (party 0)
        $masterRaw = Join-Path $LogDir "hp_p0_nv${nv}_rep${rep}_raw.log"
        $masterLog = Join-Path $LogDir "hp_p0_nv${nv}_rep${rep}.log"

        & $BIN 0 $HostsFile $nv 2> $masterLog > $masterRaw

        # Parse this repetition's output
        if ((Test-Path $masterRaw) -and ((Get-Item $masterRaw).Length -gt 0)) {
            $metrics = Parse-HpOutput $masterRaw

            Write-Host ("  setup={0}ms  prove={1}ms  verify={2}ms  proof={3}B  sent={4}B  recv={5}B" -f `
                $metrics.SetupMs, $metrics.ProverMs, $metrics.VerifierMs, `
                $metrics.ProofBytes, $metrics.CommSent, $metrics.CommRecv) -ForegroundColor Green

            # Setup is one-time; only count from first repetition
            if ($rep -eq 1) { $totSetup = $metrics.SetupMs }
            $totProver   += $metrics.ProverMs
            $totVerifier += $metrics.VerifierMs
            $totProofBytes += $metrics.ProofBytes
            $totCommSent   += $metrics.CommSent
            $totCommRecv   += $metrics.CommRecv
        } else {
            Write-Host "  No output from master" -ForegroundColor Red
        }

        # Wait for workers
        foreach ($proc in $WorkerProcs) {
            $proc.WaitForExit(60000) | Out-Null
        }
    }

    # Output summed CSV line: total cost for M instances (sequential),
    # directly comparable with sumfold_deSNARK's M-instance batch cost
    $csvLine = "{0},{1},{2},{3},{4},{5},{6},{7},{8}" -f $nv, $M, $K, $totSetup, $totProver, $totVerifier, $totProofBytes, $totCommSent, $totCommRecv
    $csvLine | Out-File -FilePath $CsvFile -Append -Encoding utf8
    $csvLine
}

Pop-Location

Write-Host ""
Write-Host "========================================================" -ForegroundColor Green
Write-Host "  Benchmark complete!" -ForegroundColor Green
Write-Host "  Results: $CsvFile" -ForegroundColor Green
Write-Host "  Logs:    $LogDir\hp_p*.log" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
