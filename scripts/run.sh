#!/usr/bin/env bash
# Start Raxit. Safe to symlink into ~/.termux/boot/ for start-on-boot.
set -euo pipefail

cd "$(dirname "$0")/.."

# Without a wake lock Android suspends the process and scheduled routines
# silently stop firing. Absent off-device, and non-fatal if it refuses.
if command -v termux-wake-lock >/dev/null 2>&1; then
  termux-wake-lock || echo "warning: could not take a wake lock" >&2
fi

# shellcheck disable=SC1091
[ -d .venv ] && source .venv/bin/activate

exec python -m raxit.server
