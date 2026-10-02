#!/usr/bin/env bash
# Record one prospective BoC forecast. Run this on an origin date.
#
# Origins are announcement dates minus 28 days. The next one is 2026-09-30,
# for the 2026-10-28 announcement; the script refuses to write anything on a
# date that is not an origin, and tells you when the next one is.
#
#   bash implementations/boc_rate_decisions/turn_detection/record_live_forecast.sh
#   bash .../record_live_forecast.sh --dry-run      # look without recording
#
# There is no scheduler on this machine, so this has to be run by hand.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT/implementations"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is not set; the agent cannot run." >&2
    exit 1
fi

PYTHON="$REPO_ROOT/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="python"

exec "$PYTHON" -u -m boc_rate_decisions.turn_detection.live_forecast "$@"
