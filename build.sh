#!/usr/bin/env bash
# Build the single-file dashboard.
#
#   ./build.sh            release build -> dist/hexapod-simulator.html
#   ./build.sh --fast     dev build: no tests, no LTO, live-reload snippet
#
# The wasm module is inlined as base64 and the telemetry offsets are generated
# from the Rust source, so the output is self-contained and the two sides of
# the buffer layout cannot drift apart.
set -euo pipefail
cd "$(dirname "$0")"

if [ "${1:-}" = "--fast" ]; then
    PROFILE=dev-fast
    CARGO_FLAGS="--profile dev-fast --offline"
    export HX_LIVE=1          # inject the live-reload poller
else
    PROFILE=release
    CARGO_FLAGS="--release --offline"
fi
WASM=target/wasm32-unknown-unknown/$PROFILE/hexapod_wasm.wasm

if [ -z "${HX_LIVE:-}" ]; then
    # Rapier aborts the process when two of its worlds are stepped at once, so
    # anything that links it is excluded here and run single-threaded below.
    # hexapod-wasm is what drags the feature in: excluding it also keeps the
    # workspace pass from unifying `rapier` onto hexapod-core.
    echo "==> cargo test"
    cargo test $CARGO_FLAGS -q --workspace --exclude hexapod-wasm 2>&1 | tail -5 \
      || cargo test --offline -q --workspace --exclude hexapod-wasm
    echo "==> cargo test (rapier plant)"
    cargo test $CARGO_FLAGS -q -p hexapod-core --features rapier -- --test-threads=1 2>&1 | tail -8 \
      || cargo test --offline -q -p hexapod-core --features rapier -- --test-threads=1
    echo "==> cargo test (wasm bridge, rapier)"
    cargo test $CARGO_FLAGS -q -p hexapod-wasm -- --test-threads=1 2>&1 | tail -8 \
      || cargo test --offline -q -p hexapod-wasm -- --test-threads=1
fi

echo "==> building wasm"
cargo build $CARGO_FLAGS --target wasm32-unknown-unknown -p hexapod-wasm

echo "==> generating servo catalogue"
cargo run $CARGO_FLAGS -q -p hexapod-cli -- servos > web/servos.gen.json
cargo run $CARGO_FLAGS -q -p hexapod-cli -- parts  > web/parts.gen.json
cargo run $CARGO_FLAGS -q -p hexapod-cli -- courses > web/courses.gen.json

echo "==> assembling dist/hexapod-simulator.html"
mkdir -p dist
python3 - "$WASM" <<'PY'
import base64, json, os, pathlib, re, sys

root = pathlib.Path(".")
web = root / "web"
wasm = pathlib.Path(sys.argv[1]).read_bytes()

# Telemetry offsets straight out of the Rust that defines them.
layout_src = (root / "crates/hexapod-wasm/src/layout.rs").read_text(encoding="utf-8")
layout = {
    m.group(1): int(m.group(2))
    for m in re.finditer(r"pub const ([TS]_\w+): usize = (\d+);", layout_src)
}
for req in ("T_LEN", "S_LEN"):
    if req not in layout:
        raise SystemExit(f"layout.rs: {req} missing")

servos = json.loads((web / "servos.gen.json").read_text(encoding="utf-8"))
parts = json.loads((web / "parts.gen.json").read_text(encoding="utf-8"))
courses = json.loads((web / "courses.gen.json").read_text(encoding="utf-8"))["courses"]

html = (web / "index.html").read_text(encoding="utf-8")
title, body = html.split("\n", 1)
if not title.startswith("<title>"):
    raise SystemExit("index.html must start with its <title>")

# Charset has to be in the first 1024 bytes or a Windows/Latin-1 default
# turns em dashes and middle dots into â€” / Â· in the gait table.
parts = [
    "<!DOCTYPE html>",
    '<html lang="en">',
    "<head>",
    '<meta charset="utf-8">',
    '<meta name="viewport" content="width=device-width, initial-scale=1">',
    title.strip(),
    "<style>\n" + (web / "style.css").read_text(encoding="utf-8") + "\n</style>",
    "</head>",
    "<body>",
    body.rstrip("\n"),
    "<script>\n"
    f"window.HX_LAYOUT={json.dumps(layout, separators=(',', ':'))};\n"
    f"window.HX_SERVOS={json.dumps(servos, separators=(',', ':'))};\n"
    f"window.HX_PARTS={json.dumps(parts['parts'], separators=(',', ':'))};\n"
    f"window.HX_COURSES={json.dumps(courses, separators=(',', ':'))};\n"
    f'window.HX_WASM_B64="{base64.b64encode(wasm).decode()}";\n'
    "</script>",
    # Each module is wrapped so the two inlined scripts cannot collide in the
    # shared global scope; they talk to each other only through `window`.
    "<script>\n(function(){\n" + (web / "render.js").read_text(encoding="utf-8") + "\n})();\n</script>",
    "<script>\n(function(){\n" + (web / "app.js").read_text(encoding="utf-8") + "\n})();\n</script>",
]

# Dev builds reload themselves when dev.sh rewrites this file.
if os.environ.get("HX_LIVE"):
    parts.append(
        "<script>\n(function(){let seen=null;setInterval(async()=>{try{"
        "const r=await fetch(location.pathname,{method:'HEAD',cache:'no-store'});"
        "const m=r.headers.get('last-modified');"
        "if(seen&&m!==seen)location.reload();seen=m;}catch(e){}},1000);})();\n</script>"
    )

parts += [
    "</body>",
    "</html>",
]

out = root / "dist/hexapod-simulator.html"
out.write_text("\n".join(parts) + "\n", encoding="utf-8")

kb = out.stat().st_size / 1024
print(f"    wasm    {len(wasm)/1024:8.1f} KB")
print(f"    layout  {len(layout)} offsets, T_LEN={layout['T_LEN']}, S_LEN={layout['S_LEN']}")
print(f"    servos  {len(servos['servos'])} parts, checked {servos['checked']}")
print(f"    output  {kb:8.1f} KB  ->  {out}")
PY
