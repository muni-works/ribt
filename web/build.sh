#!/usr/bin/env bash
# Build everything web/ serves: lilt wasm (web/pkg), the WebGPU graphs and
# weights (web/dist), and the parity references (web/parity/{native,fp16}).
#   web/build.sh [RELEASE_DIR]      (or set RIBT_WEIGHTS; the weights from huggingface.co/municipal-robotics/ribt)
# Then: python3 -m http.server 8737 -d web  →  /index.html, /parity.html
set -euo pipefail
cd "$(dirname "$0")/.."
REL="${1:-${RIBT_WEIGHTS:?pass the weights dir or set RIBT_WEIGHTS}}"
TARGET="$(cargo metadata --format-version 1 --no-deps | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')"

echo "== native (metal): ribt-tts + ribt-web-export"
cargo build --release --bin ribt-tts --bin ribt-web-export

echo "== parity references: ribt-tts synth on the 5 lines (release.toml [synth])"
RUN="$(mktemp -d)"; NAT="$(mktemp -d)"
ln -s "$REL/acoustic_norm.txt" "$RUN/norm.txt"
ln -s "$REL/acoustic_af_bon.safetensors" "$RUN/"
"$TARGET/release/ribt-tts" synth --run "$RUN" --ckpt acoustic_af_bon.safetensors \
  --flow "$REL/flow_student_af.safetensors" --nfe 1 --shortcut --temp 0.3 \
  --voc "$REL/vocoder_vocos_import.safetensors" --ref "$REL/ref_afheart.wav" \
  --seed 0 --pitch 1 --accent minnesota --lines web/parity/lines.tsv --out-dir "$NAT"
mkdir -p web/parity/native
for f in "$NAT"/*_ref.wav; do
  n="$(basename "$f" _ref.wav)"
  cp "$NAT/${n}_ref.wav" "web/parity/native/$n.wav"
  cp "$NAT/${n}_ref.mel.npy" "web/parity/native/$n.mel.npy"
  cp "$NAT/${n}_ref.durs.txt" "web/parity/native/$n.durs.txt"
done

echo "== export graphs + weights, native check of the browser pipeline"
rm -rf web/dist
"$TARGET/release/ribt-web-export" --rel "$REL" --out web/dist --check web/parity/lines.tsv --native "$NAT"

echo "== lilt → wasm"
want="$(sed -n 's/^wasm-bindgen = "=\(.*\)"/\1/p' web/lilt-wasm/Cargo.toml)"
have="$(wasm-bindgen --version | awk '{print $2}')"
[ "$want" = "$have" ] || { echo "wasm-bindgen CLI $have, crate pins $want: cargo install wasm-bindgen-cli --version $want (or re-pin)"; exit 1; }
(cd web/lilt-wasm && cargo build --release --target wasm32-unknown-unknown)
rm -rf web/pkg
wasm-bindgen --target web --no-typescript --out-dir web/pkg "$TARGET/wasm32-unknown-unknown/release/lilt_wasm.wasm"
if command -v wasm-opt >/dev/null; then wasm-opt -Oz --all-features -o web/pkg/lilt_wasm_bg.wasm web/pkg/lilt_wasm_bg.wasm; fi

du -sh web/dist web/pkg
echo "serve: python3 -m http.server 8737 -d web"
