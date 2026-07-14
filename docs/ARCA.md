# ARCA — the shared memory daemon

ARCA (**A**rchivio di **R**isposte e **C**ache per l'**A**I) is Reame's memory,
turned into a service your whole fleet can share. It speaks the **Redis wire
protocol**, so any language's existing Redis client talks to it with zero SDK.

Two layers today:

- **L1 — exact responses.** Store a computed answer once; every later identical
  request reads it back in under a millisecond instead of re-running the model.
- **L4 — shared generation corpus.** One node feeds its finished generations to
  a fleet-wide archive; other nodes draft their speculation from it. One node's
  work speeds up all the others.

Measured on a free Oracle ARM box: a question that costs ~1 s of inference is
served from ARCA in ~0.02 s the second time — **~50× faster**.

## Do you even need it?

- **One machine, casual use** → probably not yet. Run `reame run <model>` and
  ignore ARCA. (Transparent per-process use is on the roadmap; see the bottom.)
- **A fleet, or an app that repeats work** → yes. Documents re-analysed,
  batch pipelines, the same prompts across users — that is exactly where
  "never compute the same thing twice" pays, across the whole fleet.

## 1. Start the daemon

Same `reame` binary, a subcommand — nothing new to install:

```bash
reame arca --dir ~/.reame/arca
# ARCA listening on 127.0.0.1:6420 (dir ~/.reame/arca) …
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--port N` | `6420` | TCP port |
| `--dir PATH` | `~/.reame/arca` | where entries live (must be writable) |
| `--max-mb N` | unlimited | LRU byte budget on disk |

It binds to `127.0.0.1` — private by default. To share across machines, put it
behind your own firewall/VPN or an SSH tunnel; do not expose it raw to the
internet.

## 2. Use it as a response cache (L1)

Any Redis client works. From the shell with `redis-cli`:

```bash
redis-cli -p 6420 PING              # PONG
redis-cli -p 6420 SET answer:42 "Paris"
redis-cli -p 6420 GET answer:42     # "Paris"
redis-cli -p 6420 DBSIZE            # 1
```

From your app — cache a model answer keyed by the prompt (Python, no ARCA SDK,
just `redis`):

```python
import redis, hashlib
r = redis.Redis(port=6420)

def answer(prompt):
    key = "resp:" + hashlib.sha256(prompt.encode()).hexdigest()
    hit = r.get(key)
    if hit:
        return hit.decode()            # ~0.02 s
    text = call_reame(prompt)           # the slow path, once
    r.set(key, text)
    return text
```

The entries persist on disk and survive a restart — the archive is fleet
memory, not a per-process cache.

Commands: `PING`, `SET`, `GET`, `DEL`, `EXISTS`, `DBSIZE`.

## 3. Use it as shared generation memory (L4)

`ARCA.OBSERVE` feeds a finished generation to the shared corpus; `ARCA.DRAFT`
returns the best historical continuation after an n-gram. Tokens travel as
little-endian `u32` in the bulk body. A minimal client:

```python
import socket, struct

def _pack(toks):  return b"".join(struct.pack("<I", t) for t in toks)
def _unpack(b):   return [struct.unpack("<I", b[i:i+4])[0]
                          for i in range(0, len(b) - len(b) % 4, 4)]

def _cmd(sock, *parts):
    w = b"*%d\r\n" % len(parts)
    for p in parts:
        p = p if isinstance(p, bytes) else p.encode()
        w += b"$%d\r\n%s\r\n" % (len(p), p)
    sock.sendall(w)
    return sock.recv(65536)

s = socket.create_connection(("127.0.0.1", 6420))
_cmd(s, "ARCA.OBSERVE", _pack([10, 11, 12, 13, 14]))        # node A generated this
reply = _cmd(s, "ARCA.DRAFT", _pack([10, 11, 12]), "4")     # node B asks "what's next?"
body = reply.split(b"\r\n", 1)[1].rstrip(b"\r\n")
print(_unpack(body))                                         # [13, 14]
```

## 4. The fleet scenario

```
   node A (reame --serve)  ┐
   node B (reame --serve)  ├──►  reame arca  (one box, big disk)
   node C (reame --serve)  ┘
```

Point one small box at a big disk, run `reame arca` on it, and every Reame node
shares one archive: a system prompt prefilled by node A is a cache hit for B
and C; a generation observed by A drafts for the others. The archive gets more
valuable the longer the fleet runs.

## 5. What's automatic today, and what's next

**Today**: ARCA is a daemon you run and talk to from your app (or from a thin
wrapper around Reame's OpenAI-compatible API). The response-cache pattern above
is a dozen lines.

**Roadmap — transparent integration**: Reame's engine will check a configured
ARCA before generating and populate it after, so a single `[arca] remote =
host:6420` line in the config gets you fleet-shared caching with no glue code.
That is the next build after this guide.

Until then: the daemon works, it's Redis-compatible, and it's proven on real
hardware — the glue is a few lines in whatever already calls Reame.
