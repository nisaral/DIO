# Reproducible benchmark results pipeline: CSV -> JSON -> figures
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "Step 1: Analyze Locust CSVs..."
python benchmarks/real_world/analyze_results.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Step 2: Generate figures from JSON..."
python benchmarks/generate_figures_from_json.py --out ../figs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Done. See benchmarks/results_summary.json and figs/"