# Regime D — drive setup + campaign from your laptop (SG allows YOUR IP only).
# T4 = gateway + slow worker | A30 = fast worker
#
# Usage (PowerShell):
#   cd $env:USERPROFILE\OneDrive\Desktop\Go-serve
#   powershell -ExecutionPolicy Bypass -File dio-serve\scripts\cloud\launch_regime_d_from_laptop.ps1
#
# Override if needed:
#   $env:T4_IP = "216.48.183.177"; $env:A30_IP = "216.48.190.185"
#   $env:T4_PRIV = "10.18.24.7";  $env:A30_PRIV = "10.18.24.6"
#   $env:SSH_USER = "ubuntu"   # or root
#   $env:SSH_KEY  = "$env:USERPROFILE\.ssh\dio_cloud"

$ErrorActionPreference = "Stop"

$T4_IP    = if ($env:T4_IP)    { $env:T4_IP }    else { "216.48.178.114" }
$A30_IP   = if ($env:A30_IP)   { $env:A30_IP }   else { "" }  # set after A30 create
$T4_PRIV  = if ($env:T4_PRIV)  { $env:T4_PRIV }  else { "10.18.24.6" }
$A30_PRIV = if ($env:A30_PRIV) { $env:A30_PRIV } else { "" }
# E2E Ubuntu GPU images accept the attached key as root (not ubuntu).
$SSH_USER = if ($env:SSH_USER) { $env:SSH_USER } else { "root" }
$SSH_KEY  = if ($env:SSH_KEY)  { $env:SSH_KEY }  else { "$env:USERPROFILE\.ssh\dio_cloud" }
$REPO     = if ($env:REPO)     { $env:REPO }     else { "$env:USERPROFILE\OneDrive\Desktop\Go-serve" }

$sshOpts = @(
  "-i", $SSH_KEY,
  "-o", "IdentitiesOnly=yes",
  "-o", "PreferredAuthentications=publickey",
  "-o", "PasswordAuthentication=no",
  "-o", "StrictHostKeyChecking=accept-new",
  "-o", "ConnectTimeout=20",
  "-o", "ServerAliveInterval=30"
)

function Ssh-Node([string]$ip, [string]$cmd) {
  & ssh @sshOpts "${SSH_USER}@${ip}" $cmd
  if ($LASTEXITCODE -ne 0) { throw "ssh failed on $ip : $cmd" }
}

function Scp-To([string]$ip, [string]$local, [string]$remote) {
  & scp @sshOpts -r $local "${SSH_USER}@${ip}:${remote}"
  if ($LASTEXITCODE -ne 0) { throw "scp failed to $ip" }
}

Write-Host "=== Regime D laptop driver ===" -ForegroundColor Cyan
Write-Host "T4  $T4_IP  (priv $T4_PRIV)  <- gateway"
Write-Host "A30 $A30_IP (priv $A30_PRIV)"
Write-Host "user=$SSH_USER key=$SSH_KEY"
Write-Host ""

if (-not (Test-Path $SSH_KEY)) { throw "SSH key missing: $SSH_KEY" }
if (-not $A30_IP -or -not $A30_PRIV) {
  throw "Set A30 IPs first, e.g.`n  `$env:A30_IP='x.x.x.x'; `$env:A30_PRIV='10.x.x.x'"
}
if (-not (Test-Path "$REPO\dio-serve\scripts\run_regime_d_hetero.py")) {
  throw "Repo missing run_regime_d_hetero.py under $REPO\dio-serve"
}

# --- 1. SSH smoke ---
Write-Host "[1/6] SSH smoke..." -ForegroundColor Yellow
foreach ($ip in @($T4_IP, $A30_IP)) {
  try {
    Ssh-Node $ip "echo OK; whoami; nvidia-smi -L"
  } catch {
    Write-Host "SSH as $SSH_USER failed on $ip. Retrying with root..." -ForegroundColor DarkYellow
    $script:SSH_USER = "root"
    Ssh-Node $ip "echo OK; whoami; nvidia-smi -L"
  }
}

# --- 2. Copy code ---
Write-Host "[2/6] scp dio-serve to both nodes..." -ForegroundColor Yellow
foreach ($ip in @($T4_IP, $A30_IP)) {
  Ssh-Node $ip "mkdir -p ~/Go-serve"
  Scp-To $ip "$REPO\dio-serve" "~/Go-serve/"
  Ssh-Node $ip "test -f ~/Go-serve/dio-serve/scripts/run_regime_d_hetero.py && echo code_ok"
}

# --- 3. setup_worker both (parallel via two ssh jobs is hard in ps1; sequential is safer) ---
Write-Host "[3/6] setup_worker on T4 (10-20 min)..." -ForegroundColor Yellow
Ssh-Node $T4_IP "cd ~/Go-serve/dio-serve/scripts/cloud && bash setup_worker.sh"
Write-Host "[3/6] setup_worker on A30 (10-20 min)..." -ForegroundColor Yellow
Ssh-Node $A30_IP "cd ~/Go-serve/dio-serve/scripts/cloud && bash setup_worker.sh"

# --- 4. start vLLM both ---
Write-Host "[4/6] start_vllm both..." -ForegroundColor Yellow
Ssh-Node $T4_IP  "cd ~/Go-serve/dio-serve/scripts/cloud && bash start_vllm.sh 8000"
Ssh-Node $A30_IP "cd ~/Go-serve/dio-serve/scripts/cloud && bash start_vllm.sh 8000"

# Prefer private IP for A30 if reachable from T4; else public
Write-Host "[5/6] pick A30 URL (private vs public)..." -ForegroundColor Yellow
$a30Url = "http://${A30_PRIV}:8000"
$probe = & ssh @sshOpts "${SSH_USER}@${T4_IP}" "curl -sf --connect-timeout 5 ${a30Url}/v1/models >/dev/null && echo priv_ok || echo priv_fail"
if ($probe -match "priv_fail") {
  Write-Host "Private IP not reachable from T4; using public $A30_IP" -ForegroundColor DarkYellow
  Write-Host ">>> FIX SG: allow TCP 8000 from T4 public $T4_IP/32 (or private $T4_PRIV/32) to A30" -ForegroundColor Red
  $a30Url = "http://${A30_IP}:8000"
  $probe2 = & ssh @sshOpts "${SSH_USER}@${T4_IP}" "curl -sf --connect-timeout 5 ${a30Url}/v1/models >/dev/null && echo pub_ok || echo pub_fail"
  if ($probe2 -match "pub_fail") {
    throw "T4 cannot reach A30:8000 on private or public. Open security group port 8000 from T4 to A30, then re-run."
  }
}
Write-Host "A30 backend URL: $a30Url"

$backends = "t4=http://127.0.0.1:8000,a30=$a30Url"
$vram     = "t4=16000,a30=24000"

# --- 6. campaign in tmux on T4 ---
Write-Host "[6/6] launch campaign in tmux on T4 (seeds=5, ~1.5-2h)..." -ForegroundColor Yellow
$remote = @"
set -e
sudo apt-get install -y -qq tmux jq >/dev/null 2>&1 || true
# kill old session if any
tmux kill-session -t dio 2>/dev/null || true
tmux new-session -d -s dio "bash -lc 'cd ~/Go-serve/dio-serve/scripts/cloud && SEEDS=5 D3_SEEDS=3 bash run_all.sh \"$backends\" \"$vram\" 2>&1 | tee ~/regime_d_run.log; echo DONE >> ~/regime_d_run.log; exec bash'"
sleep 2
tmux ls
echo '--- preflight/campaign running inside: tmux attach -t dio ---'
"@
# write remote script via ssh
Ssh-Node $T4_IP $remote

Write-Host ""
Write-Host "=== LAUNCHED ===" -ForegroundColor Green
Write-Host "Watch:  ssh -i $SSH_KEY ${SSH_USER}@${T4_IP}  then  tmux attach -t dio"
Write-Host "Log:    ssh ... 'tail -f ~/regime_d_run.log'"
Write-Host "When DONE on T4, from laptop:"
Write-Host "  scp -i $SSH_KEY -r ${SSH_USER}@${T4_IP}:~/results_regime_d $REPO\dio-serve\results_regime_d"
Write-Host "THEN DELETE BOTH NODES IN CONSOLE."
Write-Host "backends=$backends"
Write-Host "vram=$vram"
