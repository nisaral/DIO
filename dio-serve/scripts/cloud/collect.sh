#!/usr/bin/env bash
# Pull Regime D results down from the gateway node to this machine.
#
#   bash collect.sh <gateway-ip> [remote-results-dir] [local-dest]
#
# Run from your LAPTOP (Git Bash on Windows works), not from the cloud node.
set -euo pipefail

GW="${1:-}"
REMOTE="${2:-~/results_regime_d}"
DEST="${3:-./results_regime_d}"
KEY="${KEY:-$HOME/.ssh/dio_cloud}"
USER="${SSH_USER:-ubuntu}"

if [ -z "$GW" ]; then
  echo "usage: bash collect.sh <gateway-ip> [remote-dir] [local-dir]" >&2
  exit 2
fi

mkdir -p "$DEST"
echo "pulling $USER@$GW:$REMOTE -> $DEST"
scp -i "$KEY" -o StrictHostKeyChecking=accept-new -r \
  "$USER@$GW:$REMOTE/"* "$DEST/"

echo
echo "--- collected ---"
ls -la "$DEST"
if [ -f "$DEST/paper_snippets.md" ]; then
  echo
  cat "$DEST/paper_snippets.md"
fi
echo
echo "Now DESTROY the GPU nodes in the provider console — stopping is not enough"
echo "on most Indian providers; a stopped node can still bill for its disk."
