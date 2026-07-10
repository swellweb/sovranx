# Sovrano

**A lean, fully-tested LLM inference server built on [llama.cpp](https://github.com/ggml-org/llama.cpp) — designed for the hardware you already have: shared vCPUs, free tiers, 2-core ARM boxes.**

Sovrano is not the first inference server. It's the first one that treats cheap CPU
hardware as a first-class citizen instead of a fallback. Its thesis is simple:

> **On a CPU, never compute the same thing twice.**

- 🗂️ **Persistent shared-prefix KV cache** — prompt prefixes are snapshotted to disk
  (zstd, checksummed, LRU-budgeted) and reused **across different prompts, restarts
  and processes**. A system prompt is paid for once, by the first user.
- 🔮 **Self-regulating speculative decoding** — a small draft model *or* zero-cost
  n-gram lookup proposes tokens; the target verifies them in one batched pass.
  Sovrano *measures* whether speculation pays on your hardware and switches it
  off by itself when it doesn't.
- 👥 **Interleaved multi-user serving** — N concurrent generations advance together
  inside single multi-sequence batches, sharing every read of the model weights
  (the cost that dominates memory-bound CPU decoding).
- 🌐 **OpenAI-compatible REST API** — `/v1/completions`, `/v1/chat/completions`,
  SSE streaming, sessions, bearer auth, metrics. Point any OpenAI client at it.
- 🧪 **160+ isolated tests** — every layer is mockable and tested without a model;
  correctness of the multi-sequence and speculative paths is pinned against real
  models in integration tests.

![Architecture](docs/figures/architecture.svg)

## Measured, not promised

Every number below was produced by the shipped binary on the hardware named —
including the negative results that shaped the design.

| Hardware | Model | Configuration | Result |
|---|---|---|---|
| Oracle Cloud **free tier** (2× ARM, 12 GB, €0/mo) | Qwen2.5-7B Q4_K_M | plain, KV q8_0 | **3.3 tok/s** |
| Oracle Cloud free tier | TriLM 3.9B ternary TQ2_0 | 1.1 GB total RAM | **~10 tok/s** |
| Apple M3 Pro (6 threads) | Qwen2.5-1.5B Q4_K_M | plain | **52 tok/s** |
| Shared Contabo VPS (18 oversubscribed vCPUs) | 1.5B + 0.5B draft | speculative, 87% acceptance | **3.2× speedup** |
| Shared Contabo VPS | TinyLlama 1.1B | warm disk cache vs cold | **4.8× end-to-end** |
| Apple M3 Pro | Qwen2.5-1.5B | prompt-lookup on a rewrite task | **1.44×** |
| Apple M3 Pro | TinyLlama, 3 concurrent users | interleaved vs serialized | **1.6×** |

The negative result that matters: on heavily oversubscribed shared vCPUs a draft
model runs as slowly as its target, so speculation is counter-productive there —
Sovrano detects this and disables it at runtime. Benchmarks that only show wins
are advertising; these are engineering.

## How it works

**Shared-prefix disk cache.** Prompts are split into fixed token blocks; a chain
hash keys a KV snapshot at every block boundary. A *different* prompt that shares
a prefix restores the longest cached boundary and decodes only its own tail.
Unlike GPU-resident prefix caches, snapshots live on NVMe: they survive restarts.

![Shared-prefix cache](docs/figures/prefix-cache.svg)

**Self-regulating speculation.** Classic Leviathan/Chen acceptance (the rejected
token is resampled from the residual distribution, so the output distribution is
exactly the target's), with two CPU-first twists: the draft source can be free
n-gram lookup mined from the prompt itself — ideal for extraction and rewrite
workloads — and a feedback controller adapts the draft length and turns
speculation off when measured acceptance or draft economics go negative.

![Speculative decoding](docs/figures/speculative.svg)

## Quick start

```bash
git clone https://github.com/swellweb/sovrano
cd sovrano
git submodule update --init --depth 1 third_party/llama.cpp
./build.sh                       # Release build + 160+ tests

./scripts/download_models.sh     # TinyLlama (test model, ~670 MB)

./build/src/sovrano --config config/sovrano.conf --prompt "Hello" --max-tokens 32
./build/src/sovrano --config config/sovrano.conf --serve   # OpenAI-compatible API
```

Dependencies: CMake ≥ 3.16, a C++17 compiler, and for the server Boost (headers),
nlohmann-json and zstd:

```bash
# Debian/Ubuntu
sudo apt install build-essential cmake libboost-dev nlohmann-json3-dev libzstd-dev pkg-config
# macOS
brew install cmake boost nlohmann-json zstd pkg-config
```

## Configuration highlights

```ini
[model]
path = models/qwen2.5-7b-instruct-q4_k_m.gguf
context_length = 4096      # total KV budget (shared across users when parallel > 1)
threads = 4                # fewer is often faster on shared vCPUs — measure!

[memory]
kv_cache_type = q8_0       # f16 | q8_0 | q4_0 — halve/quarter context RAM

[speculative]
enabled = true
mode = lookup              # model (needs draft_model_path) | lookup (no 2nd model)

[cache]
directory = /opt/sovrano/cache
max_size_mb = 4096         # LRU byte budget on disk

[server]
port = 8080
api_key =                  # bearer auth when set
parallel = 1               # >1 = interleaved multi-user serving
```

## API

| Endpoint | Description |
|---|---|
| `POST /v1/completions` | text completion (SSE with `"stream": true`) |
| `POST /v1/chat/completions` | chat completion |
| `POST /v1/sessions` · `.../save` · `.../load` · `DELETE .../{id}` | KV session snapshots |
| `GET /metrics` | request counters + speculative/cache metrics |
| `GET /health` | liveness (auth-exempt) |

## A note on energy

Sovrano's footprint is watt-scale, not kilowatt-scale: it targets machines that
already exist and are already powered on — no new silicon is racked to serve your
model. We don't claim better joules-per-token than a saturated datacenter GPU —
we claim you don't need one.

## Status & scope

Sovrano is young and deliberately **opinionated and focused**: CPU-only serving,
one model per process, correctness pinned by tests at every layer. Not goals:
GPU offload, training, model management UX. The llama.cpp submodule is pinned to
a known-good commit and bumped deliberately.

Documentation in Italian: [docs/README.it.md](docs/README.it.md).

## License

[MIT](LICENSE). Built on the shoulders of [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).
