# Sovrano

Motore di inferenza CPU-only per LLM in formato GGUF (target: modelli 30B),
progettato per VPS Contabo (18 vCPU, 96 GB RAM, 350 GB NVMe).

Combina due tecniche:

1. **Speculative decoding** (ispirato a DSpark/DeepSeek): un modello draft
   piccolo propone token, il modello target 30B li verifica in un singolo
   passaggio batched.
2. **Ottimizzazioni memoria** (ispirato a DwarfStar4/antirez): mmap del
   modello, quantizzazione della KV-cache, cap sulla RAM.

## Stack

- C++17, base [llama.cpp](https://github.com/ggerganov/llama.cpp) (integrazione nei prossimi step)
- Boost.Asio (server HTTP), nlohmann/json (API), Zstandard (compressione)
- CMake ≥ 3.16, flag AVX2/AVX-512 opzionali (solo x86_64)
- Test: Catch2 v3 (via FetchContent) + CTest

## Dipendenze

Il core (utils + test) compila senza dipendenze esterne. Il server richiede:

```bash
# Debian/Ubuntu (VPS)
sudo apt install build-essential cmake libboost-system-dev nlohmann-json3-dev libzstd-dev pkg-config

# macOS (sviluppo)
brew install cmake boost nlohmann-json zstd pkg-config
```

Se le dipendenze server mancano, il build le segnala e disabilita il target
server (equivalente a `--no-server`).

## Setup iniziale

```bash
git clone <repo-url> && cd Sovrano
git submodule update --init --depth 1 third_party/llama.cpp
```

Senza il submodule il progetto compila comunque (warning a configure-time),
ma il caricamento modelli fallisce a runtime con messaggio esplicativo.

Modelli di test (GGUF, in `models/`, gitignorata):

```bash
./scripts/download_models.sh        # TinyLlama 1.1B (~670 MB, per i test)
./scripts/download_models.sh --7b   # + Qwen2.5 7B (~4.7 GB)
```

## Compilazione

```bash
./build.sh                 # Release + test
./build.sh --debug         # build Debug
./build.sh --clean         # ricompila da zero
./build.sh --no-tests      # senza test
./build.sh --avx512        # abilita AVX-512 (CPU che lo supporta)
./build.sh --no-server     # solo core, senza dipendenze server
```

Oppure manualmente:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Esecuzione

```bash
./build/src/sovrano --config config/sovrano.conf
```

Configurazione: vedi [config/sovrano.conf](config/sovrano.conf) (formato INI,
chiavi lette come `sezione.chiave`).

## Struttura

```
Sovrano/
├── CMakeLists.txt          # build principale
├── build.sh                # script di build
├── cmake/                  # moduli CMake (flag compilatore, SIMD)
├── config/sovrano.conf     # configurazione di esempio
├── include/sovrano/        # header pubblici
│   └── utils/              # Config, Logger
├── src/
│   ├── main.cpp            # entry point
│   ├── core/               # LlamaModel + backend llama.cpp (reale/stub)
│   ├── speculative/        # speculative decoding (prossimi step)
│   ├── memory/             # ottimizzazioni memoria (prossimi step)
│   ├── server/             # server HTTP (prossimi step)
│   └── utils/              # config parser, logger
├── tests/
│   ├── unit/               # test isolati (Catch2)
│   └── mock/               # MockBackend (llama.cpp mockato)
├── scripts/                # download_models.sh
└── third_party/llama.cpp   # submodule (pinnato)
```

## Sviluppo (TDD)

Ogni unità ha il suo test isolato in `tests/unit/` (dipendenze mockate,
nessun filesystem/rete nei test). Ciclo: test RED → implementazione GREEN →
lint → commit.
