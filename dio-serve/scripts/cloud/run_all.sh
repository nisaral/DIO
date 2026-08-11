#!/usr/bin/env bash
# Gateway node: install DIO, preflight, run the full Regime D campaign.
#
#   bash run_all.sh "l4=http://10.0.0.11:8000,a30=http://10.0.0.12:8000" \
#                   "l4=24000,a30=24000"
#
# The ids (l4, a30) are labels that flow through to the paper's tables, and the
# VRAM figures must match the real cards -- see README_RUNBOOK.md section 8.4.
#
# Run this under tmux. If your laptop's SSH session drops mid-campaign the run
# dies with it otherwise, and you pay for the GPUs either way.
set -euo pipefail

BACKENDS="${1:-}"
VRAM="${2:-}"
MODEL="${3:-Qwen/Qwen2.5-3B-Instruct}"
# Defaults tuned for a cheaper real-SKU campaign (~1.5–2 h wall-clock vs ~3 h at n=10).
# Override: SEEDS=10 REQS=40 D3_SEEDS=5 bash run_all.sh ...
SEEDS="${SEEDS:-5}"
REQS="${REQS:-40}"
D3_SEEDS="${D3_SEEDS:-3}"
REPO="${REPO:-$HOME/Go-serve/dio-serve}"
VENV="${VENV:-$HOME/dio-gw-venv}"
OUT="${OUT:-$HOME/results_regime_d}"

if [ -z "$BACKENDS" ] || [ -z "$VRAM" ]; then
  echo "usage: bash run_all.sh 'l4=URL,a30=URL' 'l4=24000,a30=24000' [MODEL]" >&2
  exit 2
fi

if [ -z "${TMUX:-}" ]; then
  echo "WARNING: not inside tmux. If SSH drops, this run dies and you still pay."
  echo "         Ctrl-C now and run:  tmux new -s dio"
  sleep 5
fi

# --- gateway deps (CPU only: no torch, no CUDA) -------------------------------
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -e "$REPO"
pip install --quiet httpx transformers

# --- preflight ----------------------------------------------------------------
bash "$REPO/scripts/cloud/preflight.sh" "$BACKENDS"

# --- campaign -----------------------------------------------------------------
START=$(date +%s)
echo
echo "=== Regime D campaign: seeds=$SEEDS d3_seeds=$D3_SEEDS reqs/seed=$REQS ==="
echo "started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

python "$REPO/scripts/run_regime_d_hetero.py" \
  --backends "$BACKENDS" \
  --vram "$VRAM" \
  --model "$MODEL" \
  --tokenizer "$MODEL" \
  --seeds "$SEEDS" \
  --requests-per-seed "$REQS" \
  --d3-seeds "$D3_SEEDS" \
  --d2 --d3 \
  --provider "${PROVIDER:-e2e}" \
  --region "${REGION:-}" \
  --out "$OUT"

ELAPSED=$(( $(date +%s) - START ))
echo
echo "=== campaign finished in $((ELAPSED / 60)) min ==="
echo "results: $OUT"
echo
echo "NEXT: from your laptop, pull the results down:"
echo "  bash scripts/cloud/collect.sh <gateway-ip>"
echo "THEN: destroy the GPU nodes — they bill until deleted, not until idle."
