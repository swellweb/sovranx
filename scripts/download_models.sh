#!/usr/bin/env bash
# Download test models (GGUF) from Hugging Face into models/.
#   ./scripts/download_models.sh              only TinyLlama 1.1B (test model)
#   ./scripts/download_models.sh --spec-pair  Qwen2.5 1.5B + 0.5B (speculative perf tests)
#   ./scripts/download_models.sh --7b         also Qwen2.5 7B (intermediate tests)
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p models

download() {
    local url="$1" out="models/$2"
    if [[ -f "$out" ]]; then
        echo "== Already present: $out"
        return 0
    fi
    echo "== Downloading $2 ..."
    # -L follows HF redirects to the CDN; -C - resumes partial downloads.
    curl -L -C - --fail --progress-bar -o "$out.part" "$url"
    mv "$out.part" "$out"
    echo "== Done: $out ($(du -h "$out" | cut -f1))"
}

# TinyLlama 1.1B Q4_K_M (~670 MB) - used by the integration tests.
download \
    "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
    "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

if [[ "${1:-}" == "--spec-pair" ]]; then
    # Same tokenizer family: speculative decoding needs a shared vocab.
    # Qwen2.5 1.5B (~1 GB) as target, 0.5B (~500 MB) as draft.
    download \
        "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
        "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    download \
        "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
        "qwen2.5-0.5b-instruct-q4_k_m.gguf"
fi

if [[ "${1:-}" == "--7b" ]]; then
    # Qwen2.5 7B Instruct Q4_K_M (~4.7 GB) - intermediate testing.
    # The HF repo ships it split in two parts; llama.cpp loads the split
    # set when pointed at the first file.
    download \
        "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf" \
        "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
    download \
        "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf" \
        "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"
fi

echo
echo "Integration tests pick up models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
echo "automatically, or set SOVRANO_TEST_MODEL=/path/to/model.gguf"
