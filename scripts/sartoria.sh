#!/usr/bin/env bash
# Sartoria (experimental): re-cut the model against the server's OWN
# traffic. The generation archive (palimpsest corpus) becomes an imatrix
# calibration set; the model is re-quantized so that the weights your
# domain actually exercises keep precision and the rest slims down.
#
#   ./scripts/sartoria.sh <f16-master.gguf> <calibration.txt> <out.gguf> [QTYPE]
#
# STATUS — read before trusting this (Qwen2.5-1.5B, synthetic domain):
#   * Mechanism: proven end-to-end (runs in minutes on a laptop).
#   * At Q3_K_M: statistical TIE with the generic cut, replicated with
#     both a 1.4k-word and a 20k-word calibration corpus (tailored 3.18
#     vs generic 3.14, Q4 reference 3.02). Comfortable quants don't need
#     tailoring.
#   * At Q2_K — where blind cutting hurts most — the tailored cut WINS:
#     PPL 3.91±0.28 vs 4.09±0.29 generic (-4.4%), at 718MB vs the 1.0GB
#     Q4. Sartoria's use case is making the impossible quant usable
#     (e.g. a 7B in ~2.4GB tailored to your traffic), not improving the
#     comfortable one.
#   * Caveat: synthetic self-generated corpora; validate on real
#     production traffic from the palimpsest archive before deploying.
set -euo pipefail

MASTER="${1:?usage: sartoria.sh <f16-master.gguf> <calibration.txt> <out.gguf> [QTYPE]}"
CALIB="${2:?missing calibration text}"
OUT="${3:?missing output path}"
QTYPE="${4:-Q3_K_M}"

TOOLS="${SOVRANX_LLAMA_TOOLS:-}"
if [[ -z "$TOOLS" ]]; then
    echo "Building llama.cpp tools (one-off)..."
    cmake -S "$(dirname "$0")/../third_party/llama.cpp" -B /tmp/sovranx-llama-tools \
        -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF \
        -DGGML_METAL=OFF -DLLAMA_CURL=OFF > /dev/null
    cmake --build /tmp/sovranx-llama-tools --parallel \
        --target llama-imatrix llama-quantize > /dev/null
    TOOLS=/tmp/sovranx-llama-tools/bin
fi

IMATRIX="$(mktemp -t sartoria).imatrix"
echo "== 1/2 measuring weight importance on YOUR corpus =="
"$TOOLS/llama-imatrix" -m "$MASTER" -f "$CALIB" -o "$IMATRIX"

echo "== 2/2 cutting the tailored $QTYPE =="
"$TOOLS/llama-quantize" --imatrix "$IMATRIX" "$MASTER" "$OUT" "$QTYPE"

echo "Done: $OUT"
echo "Validate before deploying: llama-perplexity on HELD-OUT domain text,"
echo "against a generic $QTYPE cut of the same master."
