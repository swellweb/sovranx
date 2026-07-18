# Benchmarks

Every number here came out of a terminal on the hardware named — a free
Oracle Cloud ARM box, a shared Contabo VPS, or an Apple M3 Pro laptop. The
failed experiments are kept next to the wins, because a benchmark table that
only shows wins is advertising. If a number does not reproduce on your
hardware, [open an issue](https://github.com/swellweb/reame/issues) — the
scripts are in the repo.

## Hardware

| Name | Spec | Cost |
|---|---|---|
| Oracle free tier | Ampere A1, 2 or 4 ARM cores, 12–24 GB (Always Free) | €0/mo |
| Contabo VPS | 18 oversubscribed shared vCPUs | ~€6/mo |
| Apple M3 Pro | 6 performance threads used | laptop |

Method: decode speed is measured with a two-run difference (time for N tokens
minus time for a short run) to isolate generation from model load and prefill,
or with `llama-cli`'s native `llama_perf` report where noted. Accuracy uses a
fixed long-context document with eight fact-retrieval ("needle") questions,
graded on the final answer only.

## Decode speed by model

The single most important CPU lesson: **architecture beats size**. A
mixture-of-experts model reads only its active parameters per token; a dense
model reads all of them. On memory-bandwidth-bound CPU decode, that is the
whole ballgame.

| Machine | Model | Type | Decode | Verdict |
|---|---|---|---|---|
| Oracle free (4 core) | Marco-Nano 8B-A0.6B | MoE, 0.6B active | **46.2 tok/s** | fastest here; multilingual |
| Oracle free (4 core) | OLMoE 7B-A1B | MoE, 1B active | **26.7 tok/s** | the live-serving pick |
| Oracle free (4 core) | Qwen2.5-3B | dense | 14.3 tok/s | dense reference point |
| Oracle free (2 core) | Qwen2.5-7B | dense | 3.3 tok/s | superseded by OLMoE |
| Oracle free | TriLM 3.9B TQ2_0 | dense ternary | ~10 tok/s | 1.1 GB RAM total |
| Oracle free (4 core) | Qwen3-30B-A3B | MoE, 3B active | ~335 s/question | batch only |
| Oracle free (4 core) | Qwen3.6-27B | dense | **~0.1 tok/s** | unusable here |
| Oracle free (4 core) | Ornith-1.0-9B (Qwen3.5-9B finetune) | dense | 5.4 tok/s | usable for batch judgment |
| Oracle free (4 core) | Gemma 4 E2B | dense, 2B effective | 18.2 tok/s | 8/8 needle, 3.3 GB — best small |
| Oracle free (4 core) | Gemma 4 E4B | dense, 4B effective | 10.1 tok/s | verbose, no gain over E2B |
| M3 Pro | Qwen2.5-1.5B | dense | 52 tok/s | laptop default |
| M3 Pro | Qwen3.5-9B | dense | 16.6 tok/s | judgment tasks |

Community recommendations get tested too: Ornith-1.0-9B (an HN-suggested
finetune of Qwen3.5-9B) runs at 5.4 tok/s on the free tier — the same base as
the judgment pick, and a genuinely usable 9B for reasoning batches on a €0 box.

**Fewer active parameters, faster decode — measured in one sitting.** The three
rows below were taken back-to-back on the same 4-core Oracle box, same prompt,
same two-run method, so the ratios are apples-to-apples:

| Model | Active params | Decode (same session) |
|---|---|---|
| Marco-Nano 8B-A0.6B | 0.6B | 46.2 tok/s |
| OLMoE 7B-A1B | 1B | 33.5 tok/s |
| Qwen2.5-3B | 3B (dense) | 14.3 tok/s |

Active parameter count predicts decode speed better than total size: an 8B MoE
touching 0.6B/token outruns a 3B dense model by 3.2×, despite being larger on
disk. Note the honest wrinkle: OLMoE measured **33.5 tok/s** in this session
versus the **26.7 tok/s** published above from an earlier one. Same box, same
method, different day — that spread is the run-to-run variance you should
expect on a shared cloud instance. Trust the ratios within a session more than
absolute numbers across sessions.

**Language coverage is a real axis, not a footnote.** OLMoE is English-centric:
asked in Italian it code-switches mid-sentence and drops English words into the
answer. Marco-Nano (a multilingual MoE) answers the same prompts in fluent
Italian. For a public demo that non-English speakers will poke at, that matters
as much as tok/s. Neither model is a knowledge oracle: asked for a carbonara
recipe, Marco-Nano stays plausible (guanciale, no onion) but still suggests
optional cream, which a dense Qwen3.5-9B correctly refuses. Small models on CPU
are for narrow work over supplied context, not for recall.

**Not tested: Marco-Mini 17.3B-A0.86B.** The larger sibling has no public GGUF
quantization at the time of writing, so it cannot be loaded by llama.cpp
without converting the weights first. Listed here so the gap is visible rather
than silently omitted.

Read the two extremes together: OLMoE (7B total, 1B active) serves at 26.7
tok/s, while Qwen3.6-27B (dense) crawls at ~0.1 tok/s on the same box — a
250× gap driven entirely by how many parameters each reads per token, not by
how many they contain. The 27B is also a reasoning model, so it burns hundreds
of `<think>` tokens before answering: minutes of wait for a single reply.

## Accuracy (long-context extraction)

| Machine | Model | Needle score | Note |
|---|---|---|---|
| Oracle free | Qwen2.5-7B dense | 8/8 | baseline |
| Oracle free | OLMoE 7B-A1B | 8/8 | same accuracy, 5.4× the speed |
| Oracle free | Qwen3-30B-A3B | 8/8 | same accuracy, 10× the time |

When the answer lives in the context you provide, extra parameters buy
nothing. A 7B-active model retrieves facts as accurately as a 30B one — it
just reads the document faster. This is the empirical core of Reame's thesis.

## Judgment (reasoning over your data)

Extraction is not judgment. On an SEO audit of a live page — where the model
must *reason*, not just retrieve — smaller models invented findings that
weren't there. Qwen3.5-9B was the only model tested with **zero invented
findings**, completing the full audit in **73s on the M3 Pro laptop**. For
tasks that need real reasoning in batches, a 9B is the floor.

## Feature speedups

Each of these is a way to avoid recomputing something, measured against the
same model doing the naive thing.

| Feature | Workload | Speedup |
|---|---|---|
| Warm disk cache vs cold | TinyLlama, repeated prefix | **4.8×** end-to-end |
| Palimpsest (generation archive) | Qwen2.5-1.5B, repeated request | **2.3×** (22→51 tok/s) |
| Il Suggeritore (form drafting) | Qwen2.5-1.5B, fresh list | **2.1×** (4.4s→2.1s) |
| Interleaved multi-user | TinyLlama, 3 concurrent | **1.6×** vs serialized |
| Prompt-lookup speculation | Qwen2.5-1.5B, rewrite task | **1.44×** |
| Draft-model speculation | 1.5B + 0.5B draft, Contabo | **3.2×** (87% acceptance) |
| Conclave (shared prefill + early consensus) | 1.5B ×5, arithmetic quiz | wall **97s → ~50s** |
| Warm-ahead (POST /v1/warm) | OLMoE, 1116-token doc pre-digested, Oracle free tier | TTFT **20.6s → 3.4s (6.1×)** |
| Warm-ahead | same, M3 Pro | TTFT **8.7s → 1.6s (5.3×)** |

## Reame vs llama.cpp

Reame is built on llama.cpp and calls its kernels directly, so raw decode
speed is identical — Reame adds no per-token overhead. What Reame adds is the
memory layer llama.cpp does not have: the disk KV cache, the generation
archive, self-regulating speculation, the Conclave. On a single cold request
Reame ≈ llama.cpp; on a repeated or cached workload Reame pulls ahead by
exactly the feature-speedup factors above. The point of Reame is not to be a
faster llama.cpp — it is to be a llama.cpp that remembers.

## Negative results that shaped the design

- **A 30B-class MoE does not beat a 7B on extraction.** Same 8/8 accuracy, 10×
  the time. MoE prefill touches nearly every expert, so the active-parameter
  discount vanishes on document reading. Reserve 30B-class models for
  hard-reasoning batch jobs.
- **Speculation is counter-productive on oversubscribed shared vCPUs.** A 0.5B
  draft runs as slowly as its 7B target when the cores are contended, so the
  draft cost is pure loss. Reame measures this at runtime and disables it.
- **The Conclave does not create capability.** Majority voting over N attempts
  corrects random slips, not systematic misunderstanding: a 1.5B ×5 lands
  between a 1.5B and a 3B, never above the 3B. It is a variance knob, not a
  size upgrade.
- **A dense 27B is unusable on a free-tier CPU.** ~0.1 tok/s. Bigger is not
  better when every parameter is read every token.
