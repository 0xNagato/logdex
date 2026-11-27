# logdex

Turn a file you cannot open into references you can.

`logdex` indexes huge logs, JSONL tapes, transcripts and datasets so they can be
**searched without being read**. The rule it enforces: retrieval returns
**pointers, never payload** — a search hands back `path:start-end` plus a
summary, and you read only that slice. That is why context stays small and why
the index is still useful ten sessions later. Built for AI agents with limited
context windows, but useful to anyone drowning in multi-gigabyte logs.

## Two layers — use the cheap one first

**1. Manifest (always free, no network, stdlib only).** One record per chunk:
line range, byte range, time range, event histogram, field names, sample lines.

```bash
python3 logdex.py index big.jsonl            # writes big.jsonl.logdex.jsonl
```

The manifest is ~0.9% of the source and is plain JSONL — **grep it**:

```bash
grep '"ts_min":"2026-07-16T10' big.jsonl.logdex.jsonl   # chunks in a time range
grep 'window_closed'           big.jsonl.logdex.jsonl   # chunks that saw an event
python3 logdex.py show big.jsonl.logdex.jsonl big.jsonl#000042   # print one chunk
```

*Measured: a 127 MB JSONL became a 1.1 MB manifest in 4.4 s.*

**2. Vectors (optional, for conceptual queries).** Embeds each chunk's real text
into [Qdrant](https://qdrant.tech/) twice: a dense MiniLM vector for meaning and a
BM25 sparse vector for exact terms, fused at query time with Reciprocal Rank
Fusion. The payload is still the manifest record, so a hit is a reference.

```bash
python3 logdex.py embed  big.jsonl.logdex.jsonl
python3 logdex.py search "ran out of balance mid-trade" -k 8
```

Why hybrid: a dense vector alone scored near-exact keyword matches barely above
generic text. BM25 makes an exact term always count, so recall no longer depends
on guessing the phrasing.

## When to use which

- **Structural / exact** (a timestamp, event name, id, error string) → **grep the
  manifest**. Do not embed.
- **Conceptual** ("where did it start behaving oddly") → **`search`**.

Vectors earn their keep across a *heterogeneous* corpus. Search scores are fused
rank scores, not cosine similarity — judge by **ranking** and the gap to the next
hit, never the absolute number, and confirm by reading the slice.

Full docs, conventions and gotchas: [`SKILL.md`](SKILL.md).

## Install

```bash
# index + show: nothing to install (Python 3 stdlib)
pip install fastembed        # only for embed + search
# and run a Qdrant instance (default http://localhost:6333)
```

Config via env: `QDRANT_URL`, `QDRANT_API_KEY` (falls back to
`/etc/qdrant/config.yaml`).

Upgrading from the summary-only version: a collection it created has no BM25
field, and logdex never recreates an existing collection. Searching it still
works (dense only), but `embed` into it fails with `Not existing vector name
error: bm25`. Delete it (`curl -X DELETE $QDRANT_URL/collections/logdex`) and
re-embed your manifests.

## Using it as an agent skill

`SKILL.md` is written so an LLM agent (Claude, OpenClaw, etc.) knows when to reach
for logdex instead of trying to read a file too big for its context. Drop this
folder into your agent's skills directory.

## License

MIT — see [LICENSE](LICENSE).
