#!/usr/bin/env bash
# Paper-critical tests only: T2 heterogeneity + core matrix (skips T7/T1).
# Usage: SKIP_PREFLIGHT=1 bash benchmarks/run_lightning_core.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SKIP_T7=1
exec bash "$SCRIPT_DIR/run_lightning_full.sh" "$@"