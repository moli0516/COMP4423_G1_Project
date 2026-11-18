#!/usr/bin/env bash
# macOS/Linux synthetic DeskCheck launcher. No installs or model download.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

python=".venv/bin/python"
port="${DESKCHECK_PORT:-8503}"
if [[ ! -x "$python" ]]; then
  printf '%s\n' 'Missing .venv/bin/python. Create a fresh venv and install requirements.txt; see docs/C_SOURCE_RELEASE.md.' >&2
  exit 1
fi
if [[ ! "$port" =~ ^[0-9]{1,5}$ ]] || (( 10#$port < 1 || 10#$port > 65535 )); then
  printf '%s\n' 'DESKCHECK_PORT must be an integer from 1 to 65535.' >&2
  exit 2
fi

"$python" -B scripts/check_c_setup.py --profile ui --port "$port"
exec "$python" -m streamlit run app/streamlit_app.py \
  --server.address=127.0.0.1 \
  --server.port="$port" \
  --server.maxUploadSize=32 \
  --browser.gatherUsageStats=false
