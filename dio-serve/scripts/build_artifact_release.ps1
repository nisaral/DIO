param(
  [string]$Version = "0.4.0",
  [string]$Out = "release"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$target = Join-Path $root "$Out\dio-$Version"
if (Test-Path $target) { Remove-Item -Recurse -Force $target }
New-Item -ItemType Directory -Force $target | Out-Null
# Missing entries are tolerated - the copy loop below skips them (community
# files such as LICENSE / CONTRIBUTING.md / SECURITY.md may not exist yet).
$paths = @(
  "dio-serve\src", "dio-serve\tests", "dio-serve\scripts",
  "dio-serve\docs", "dio-serve\pyproject.toml", "dio-serve\README.md",
  "dio-serve\LICENSE", "dio-serve\dio.example.yaml", "dio-serve\Dockerfile",
  "dio-serve\.dockerignore", "dio-serve\docker-compose.yml", "dio-serve\docker",
  "dio-serve\results_reanalysis",
  "dio-serve\results_regime_d\summary.json",
  "dio-serve\results_gpu_abc_n10\summary.json",
  "dio-serve\results_workshop_final\summary.json",
  "LICENSE", "CONTRIBUTING.md", "SECURITY.md",
  ".zenodo.json", "CITATION.cff"
)
foreach ($p in $paths) {
  $src = Join-Path $root $p
  if (Test-Path $src) {
    $dest = Join-Path $target $p
    if ((Get-Item $src).PSIsContainer) { Copy-Item $src $dest -Recurse -Force }
    else { New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null; Copy-Item $src $dest -Force }
  }
}
# Never archive local virtual environments or vendored checkouts from smoke tests.
foreach ($junk in @(
  (Join-Path $target "dio-serve\scripts\.venv"),
  (Join-Path $target "dio-serve\scripts\cloud\prodstack_smoke\.venv"),
  (Join-Path $target "dio-serve\scripts\cloud\prodstack_smoke\production-stack"),
  (Join-Path $target "dio-serve\.venv_mock")
)) {
  if (Test-Path $junk) { Remove-Item -Recurse -Force $junk }
}
$manifest = [ordered]@{ version=$Version; generated_utc=(Get-Date).ToUniversalTime().ToString("o"); source_repo="https://github.com/nisaral/DIO"; contains=$paths }
$manifest | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $target "RELEASE_MANIFEST.json") -Encoding utf8
$zip = Join-Path $root "$Out\dio-$Version.zip"
Compress-Archive -Path $target -DestinationPath $zip -Force
Write-Output $zip
