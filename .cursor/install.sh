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
# Rapier 0.32 needs Rust 1.87 (see rust-toolchain.toml).
rustup toolchain install 1.87.0
rustup target add wasm32-unknown-unknown --toolchain 1.87.0

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

# 4. Warm the Rust build caches (native + wasm) so ./build.sh is quick.
#    Rapier and its registry deps are fetched once, then later builds can go
#    offline from this cache.
cargo fetch
cargo build --release
cargo build --release --target wasm32-unknown-unknown -p hexapod-wasm
cargo test --release -p hexapod-core --features rapier --offline --no-run || \
  cargo test --release -p hexapod-core --features rapier --no-run
