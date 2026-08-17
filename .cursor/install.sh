#!/usr/bin/env bash
# Idempotent bootstrap for the Hexapod Gait Lab dev environment.
#
# Prepares everything two documented workflows need:
#   ./build.sh          -> test, compile wasm, emit dist/hexapod-simulator.html
#   node test/smoke.mjs -> drive the built page in a real browser
#
# Deliberately does NOT run the test suite or a fragile build; it only refreshes
# toolchains, dependencies and warm compile caches so those commands run fast.
set -euo pipefail
cd "$(dirname "$0")/.."

# 1. wasm target for the browser module (no-op once present).
rustup target add wasm32-unknown-unknown

# 2. Node dev tooling. Only Playwright, used by test/smoke.mjs; the app ships no
#    runtime JS dependencies.
npm install

# 3. Chromium at the exact revision test/smoke.mjs hardcodes
#    (/opt/pw-browsers/chromium-1194). /opt is root-owned, so create the tree
#    once and hand it to the current user; every later run just reuses it.
BROWSERS_DIR=/opt/pw-browsers
if [ ! -d "$BROWSERS_DIR" ]; then
  sudo mkdir -p "$BROWSERS_DIR"
fi
if [ ! -w "$BROWSERS_DIR" ]; then
  sudo chown -R "$(id -u):$(id -g)" "$BROWSERS_DIR"
fi
PLAYWRIGHT_BROWSERS_PATH="$BROWSERS_DIR" npx playwright install chromium

# 4. Warm the Rust build caches (native + wasm) so ./build.sh is quick. Offline
#    is safe: the whole workspace is path-only crates with no registry deps.
cargo build --release --offline
cargo build --release --offline --target wasm32-unknown-unknown -p hexapod-wasm
