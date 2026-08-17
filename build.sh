#!/usr/bin/env bash
# Build the single-file dashboard.
#
#   ./build.sh            release build -> dist/hexapod-simulator.html
#
# The wasm module is inlined as base64 and the telemetry offsets are generated
# from the Rust source, so the output is self-contained and the two sides of
# the buffer layout cannot drift apart.
set -euo pipefail
cd "$(dirname "$0")"

CARGO_FLAGS="--release --offline"
WASM=target/wasm32-unknown-unknown/release/hexapod_wasm.wasm

echo "==> cargo test"
cargo test $CARGO_FLAGS --offline -q 2>&1 | tail -5 || cargo test --offline -q

echo "==> building wasm"
cargo build $CARGO_FLAGS --target wasm32-unknown-unknown -p hexapod-wasm

echo "==> generating servo catalogue"
cargo run $CARGO_FLAGS -q -p hexapod-cli -- servos > web/servos.gen.json
cargo run $CARGO_FLAGS -q -p hexapod-cli -- parts  > web/parts.gen.json
cargo run $CARGO_FLAGS -q -p hexapod-cli -- courses > web/courses.gen.json

echo "==> assembling dist/hexapod-simulator.html"
mkdir -p dist
python3 - "$WASM" <<'PY'
import base64, json, pathlib, re, sys

root = pathlib.Path(".")
web = root / "web"
wasm = pathlib.Path(sys.argv[1]).read_bytes()

# Telemetry offsets straight out of the Rust that defines them.
layout_src = (root / "crates/hexapod-wasm/src/layout.rs").read_text()
layout = {
    m.group(1): int(m.group(2))
    for m in re.finditer(r"pub const ([TS]_\w+): usize = (\d+);", layout_src)
}
for req in ("T_LEN", "S_LEN"):
    if req not in layout:
        raise SystemExit(f"layout.rs: {req} missing")

servos = json.loads((web / "servos.gen.json").read_text())
parts = json.loads((web / "parts.gen.json").read_text())
courses = json.loads((web / "courses.gen.json").read_text())["courses"]

html = (web / "index.html").read_text()
title, body = html.split("\n", 1)
if not title.startswith("<title>"):
    raise SystemExit("index.html must start with its <title>")

parts = [
    title,
    "<style>\n" + (web / "style.css").read_text() + "\n</style>",
    body,
    "<script>\n"
    f"window.HX_LAYOUT={json.dumps(layout, separators=(',', ':'))};\n"
    f"window.HX_SERVOS={json.dumps(servos, separators=(',', ':'))};\n"
    f"window.HX_PARTS={json.dumps(parts['parts'], separators=(',', ':'))};\n"
    f"window.HX_COURSES={json.dumps(courses, separators=(',', ':'))};\n"
    f'window.HX_WASM_B64="{base64.b64encode(wasm).decode()}";\n'
    "</script>",
    # Each module is wrapped so the two inlined scripts cannot collide in the
    # shared global scope; they talk to each other only through `window`.
    "<script>\n(function(){\n" + (web / "render.js").read_text() + "\n})();\n</script>",
    "<script>\n(function(){\n" + (web / "app.js").read_text() + "\n})();\n</script>",
]

out = root / "dist/hexapod-simulator.html"
out.write_text("\n".join(parts))

kb = out.stat().st_size / 1024
print(f"    wasm    {len(wasm)/1024:8.1f} KB")
print(f"    layout  {len(layout)} offsets, T_LEN={layout['T_LEN']}, S_LEN={layout['S_LEN']}")
print(f"    servos  {len(servos['servos'])} parts, checked {servos['checked']}")
print(f"    output  {kb:8.1f} KB  ->  {out}")
PY
