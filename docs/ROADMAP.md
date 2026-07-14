# Roadmap

The plan, in phases. Each item is either shipped, next, or later — and a few
things are explicitly *not* goals, decided by measurement, not taste.

The through-line never changes: **on a CPU, never compute the same thing
twice.** Every phase either avoids work or shares work.

## Shipped

The engine and its memory are done and public (MIT, CI on Linux + macOS,
254 test cases):

- **Core** — llama.cpp CPU engine, OpenAI-compatible HTTP API, sessions,
  zero-config CLI (`reame run` / `list`), per-model chat templates.
- **Speculation** — self-regulating: prompt-lookup, draft-model, and it turns
  itself off when it stops paying.
- **Palimpsest** — the generation archive drafts future tokens from past ones.
- **Il Suggeritore** — grammar as a free draft source.
- **The Conclave** — `--best-of N`, consensus as a quality knob.
- **Interleaved multi-user** — N generations share each weight read.
- **MoE serving** — OLMoE 7B-A1B is the measured fast pick (26.7 tok/s on a
  free ARM box).
- **ARCA** — the Redis-compatible shared memory daemon: L1 exact responses,
  L4 shared generation corpus, and **transparent caching** (one config line,
  `[arca] remote = host:6420`, deterministic requests cached automatically).
- **Launch** — public repo, Homebrew, release binaries, honest benchmarks
  page, logo, sponsors, site.

## Phase 1 — ARCA to its full value (capability)

The memory layer is the moat. Finish it.

1. **Warm-ahead prefill** — ✅ *shipped*. `POST /v1/warm {"prompt": ...}`
   prefills a document into the disk cache before anyone asks; the first real
   request that shares that prefix skips the prefill. Measured: time-to-first-
   token 20.6s → 3.4s (6.1×) on the Oracle free tier, 8.7s → 1.6s on an M3 Pro.
   Next: store/share the warm KV across the fleet via ARCA L3.
2. **ARCA L2 — semantic cache** — match *similar* questions, not just
   identical, via a small embedding model. Turns the exact-match cache into a
   fuzzy one; big for RAG and support workloads.
3. **ARCA L3 — shared KV blocks** — serve prefill KV across nodes, so a system
   prompt prefilled by node A is a warm start for B and C. Pairs with
   warm-ahead: the bot fills L3, the fleet drinks from it.
4. **Auto-sync L4 → local decoder** — each node periodically pulls the shared
   corpus into its own speculative decoder, so one node's output speeds up the
   others' *generation*, not just their cache hits.

## Phase 2 — adoption (turn the launch into users)

Features nobody uses don't matter. This is where an hour pays most right now.

1. **"Reame for business" page** on swellweb — the funnel: *your LLM on your
   servers, data never leaves*. Legal/medical/PA angle, the €0 free-tier proof.
2. **Continue.dev autocomplete demo** — private code completion (Qwen2.5-Coder
   + the OpenAI-compatible API), a GIF in the README. Concrete, shareable.
3. **npm package** (`npx reame`) — binaries are already built per platform;
   wrap them.
4. **Nurture the launch** — answer HN/GitHub, fix reported friction same-day.
   The "a human answers in an hour" reputation is worth more than any feature.

## Phase 3 — polish & maintenance

1. **Bump the llama.cpp submodule** — a year old; a recent one brings better
   ARM low-bit kernels (KleidiAI) for free. Re-measure after.
2. **systemd unit for ARCA** on the fleet boxes (productionize the daemon).
3. **Sync the Italian README** to the English one (currently banner-flagged as
   possibly stale).

## Not goals (decided by measurement)

- **A custom ternary matmul kernel.** Measured: ARCA + warm-ahead skip prefill;
  a faster prefill kernel only accelerates it ~few %, and our ternary prefill
  is already healthy (136 tok/s on an M3). The white-space (NEON-only ternary
  prefill) is a real research target but a bad product bet. Revisit only if we
  ever run on SME/SVE hardware or commit to the BitNet path.
- **GPU offload.** Reame's moat is being CPU-first and focused. GPU inference
  is a crowded field where Reame has no edge; adding it dilutes the one thing
  that makes Reame special.

## How we work

TDD, isolated test first. Each feature on its own branch → PR → CI green →
merge. Benchmarks measured on real hardware, negative results published.
