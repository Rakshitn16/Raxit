#!/data/data/com.termux/files/usr/bin/bash
# One-time setup for Raxit on a Termux-equipped Android device.
#
# Before running: install Termux AND Termux:API from F-Droid (not the Play
# Store builds — they are outdated and the API bridge will not connect).
set -euo pipefail

echo "==> Installing packages"
pkg update -y
pkg install -y python termux-api git libexpat openssl

echo "==> Granting storage access (accept the Android prompt)"
termux-setup-storage || echo "    skipped — grant it later with termux-setup-storage"

echo "==> Creating virtualenv"
cd "$(dirname "$0")/.."
python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "==> Wrote .env — add your ANTHROPIC_API_KEY before starting."
fi

echo "==> Verifying the Termux:API bridge"
if termux-battery-status >/dev/null 2>&1; then
  echo "    OK"
else
  echo "    FAILED. Install the Termux:API *app* from F-Droid, then re-check."
fi

cat <<'EOF'

Setup complete.

  1. Put your key in .env:      ANTHROPIC_API_KEY=sk-ant-...
  2. Keep Termux alive:         termux-wake-lock
  3. Start Raxit:               ./scripts/run.sh
  4. Open the tablet browser:   http://127.0.0.1:8788

To survive reboots, install Termux:Boot from F-Droid and symlink run.sh into
~/.termux/boot/.
EOF
