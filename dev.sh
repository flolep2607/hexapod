#!/usr/bin/env bash
# Dev loop: fast build, static server, rebuild + browser reload on save.
#
#   ./dev.sh              serve on :8000
#   PORT=9000 ./dev.sh
#
# No tests and no LTO here (see build.sh --fast) -- run ./build.sh before
# pushing, that is the one that tests and builds the release wasm.
set -uo pipefail
cd "$(dirname "$0")"

PORT=${PORT:-8000}
URL="http://localhost:$PORT/hexapod-simulator.html"

./build.sh --fast || echo "!! initial build failed, serving whatever is in dist/"

python3 -m http.server "$PORT" --bind 0.0.0.0 --directory dist >/dev/null 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT

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
