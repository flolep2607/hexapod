#!/usr/bin/env bash
# Dev loop: fast build, static server, rebuild + browser reload on save.
#
#   ./dev.sh              serve on :8000 (next free port if taken)
#   PORT=9000 ./dev.sh    pin the port -- fails if busy
#   ./dev.sh --help
#
# No tests and no LTO here (see build.sh --fast) -- run ./build.sh before
# pushing, that is the one that tests and builds the release wasm.
set -uo pipefail
cd "$(dirname "$0")"

case ${1:-} in -h|--help) sed -n '2,9p' "$0" | cut -c3-; exit 0 ;; esac

# On WSL2 with mirrored networking the Windows port space is shared, so a port
# can be busy even though `ss -ltn` shows nothing listening inside Linux.
PINNED=${PORT:+1}
PORT=${PORT:-8000}

./build.sh --fast || echo "!! initial build failed, serving whatever is in dist/"

serve() {   # $1 = port; echoes pid on success
    local log; log=$(mktemp)
    python3 -m http.server "$1" --bind 0.0.0.0 --directory dist >"$log" 2>&1 &
    local pid=$!
    sleep 0.5
    if kill -0 $pid 2>/dev/null; then rm -f "$log"; echo $pid; return 0; fi
    wait $pid 2>/dev/null
    tail -1 "$log" >&2; rm -f "$log"; return 1
}

SERVER=$(serve "$PORT") || {
    if [ -n "$PINNED" ]; then
        echo "!! port $PORT is busy (check the Windows side too -- mirrored networking)" >&2
        exit 1
    fi
    for try in $(seq $((PORT + 1)) $((PORT + 20))); do
        if SERVER=$(serve "$try"); then PORT=$try; break; fi
    done
    [ -n "${SERVER:-}" ] || { echo "!! no free port in $PORT..$((PORT + 20))" >&2; exit 1; }
    echo "==> :${PORT} (8000 was busy)"
}
trap 'kill $SERVER 2>/dev/null' EXIT

URL="http://localhost:$PORT/hexapod-simulator.html"

echo
echo "==> $URL"
echo "==> watching web/ crates/ build.sh -- edit, save, the page reloads itself"
echo

STAMP=$(mktemp)
while sleep 1; do
    changed=$(find web crates build.sh -type f -newer "$STAMP" -not -name '*.gen.json' -print -quit)
    [ -z "$changed" ] && continue
    echo "==> $changed"
    ./build.sh --fast || echo "!! build failed, page left as-is"
    touch "$STAMP"   # after the build so the files it writes don't retrigger
done
