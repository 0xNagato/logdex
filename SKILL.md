---
name: logdex
description: Break huge logs and files into context-friendly indexes and vectors so they can be searched without being read. Use when a log, JSONL, tape, transcript, or dataset is too big to open (anything over ~5 MB), when you need to find where in a large file something happened, when a grep would return thousands of lines, or when you want durable searchable memory over a corpus. Keyword triggers "index this log", "too big to read", "search the logs", "find where", "chunk this file", "vector index", "semantic search over logs", "logdex".
---

# logdex

Turns a file you cannot open into references you can.

**The rule this enforces: retrieval returns POINTERS, never payload.** A search
gives back `path:start-end` plus a summary. You then read only that slice. This
is why context stays small and why the index is still useful ten sessions later.

## Two layers — use the cheap one first

**1. Manifest (always, free, no network).** One record per chunk with line range,
byte range, time range, event histogram, field names, and sample lines.

```bash
python3 logdex.py index <file> [--chunk-lines 400] [--out PATH]
```

Writes `<file>.logdex.jsonl` (~0.9% of the source). Measured: a 127 MB JSONL
became a 1.1 MB manifest in 4.4 s.

The manifest is a plain JSONL — **grep it**. This answers most questions with no
embeddings at all:

```bash
# which chunks cover a time range?
grep '"ts_min":"2026-07-16T10' <file>.logdex.jsonl
# which chunks saw a given event?
grep 'window_closed' <file>.logdex.jsonl
```

Then read only the slice, via your editor/agent's read-with-offset, or:
```bash
python3 logdex.py show <manifest> <chunk_id>
```

**2. Vectors (optional, for conceptual queries).** Embeds each chunk's *summary*
— never its bulk text — into Qdrant.

```bash
python3 logdex.py embed  <manifest> [--collection logdex]
python3 logdex.py search "<question>" [--collection logdex] [-k 8]
```

Measured: 1381 chunk-summaries embedded in 15 s; a query for "ran out of balance,
could not complete the second leg" ranked first a chunk containing 225
`"reason":"balance"` lines, against 0 in a control chunk.

## When to use which

- **Structural / exact** — a timestamp, an event name, an order id, an error
  string → **grep the manifest**, or grep the source directly. Do not embed.
- **Conceptual** — "where did it start behaving oddly", "find the balance
  starvation" → **`search`**.

Vectors earn their keep across a *heterogeneous* corpus (many files, mixed
content). Within one homogeneous log every chunk looks alike, so absolute cosine
scores run low (~0.25) — judge by **ranking**, not by score, and confirm the hit
by reading the slice.

## Where manifests live

A manifest defaults to `<file>.logdex.jsonl`, right next to the source. To build
a searchable library over many files, point `--out` at a directory of your
choosing and keep every manifest there; its filenames make a plain-text index
you can `ls` and `grep`:

```bash
python3 logdex.py index big.jsonl --out ./manifests/big.jsonl.logdex.jsonl
ls ./manifests/                                  # what is already indexed
grep -l 'window_closed' ./manifests/*.logdex.jsonl   # which file saw it
```

Re-run `index` + `embed` for a file that has grown; point ids are a stable
sha256 of the chunk id, so re-embedding **updates in place** rather than
duplicating. Line numbers of existing chunks stay valid for append-only logs.

Nothing to embed: binary columnar stores (e.g. ClickHouse MergeTree parts,
Parquet column data) carry no free text — index the query output, not the parts.

## Conventions it keeps

Vector name `fast-all-minilm-l6-v2`, 384-dim, cosine, model
`sentence-transformers/all-MiniLM-L6-v2` via **fastembed** — deliberately the
same shape [`mcp-server-qdrant`](https://github.com/qdrant/mcp-server-qdrant)
writes, so anything indexed here is format-compatible with a Qdrant MCP server.
Do not change the vector name without re-indexing every collection.

Qdrant URL/key come from `QDRANT_URL` (default `http://localhost:6333`) and
`QDRANT_API_KEY`, falling back to `/etc/qdrant/config.yaml`.

Note on MCP interop: an `mcp-server-qdrant` is pinned to a single
`COLLECTION_NAME`, so it will only see the logdex collection if you point it at
the same name. Use `logdex.py search` for logdex data, or pass
`--collection <name>` to write chunk-references into a collection your MCP
server already reads.

## Gotchas

- The manifest is small but not tiny (~1.1 MB for 127 MB). **Grep it; do not
  read it whole.**
- `--chunk-lines` is the precision/recall dial. Smaller chunks = sharper
  references, bigger manifest, more vectors.
- `--max-chunk-bytes` (default 32 KB) caps a chunk's size so a "reference" never
  turns into a context bomb — 400 lines of a wide JSONL can be ~900 KB.
- Timestamps: seconds vs milliseconds are told apart by magnitude, not field
  name, because both spellings carry both units in the wild.

## Requirements

- Python 3, no dependencies for `index`/`show` (stdlib only).
- `pip install fastembed` for `embed`/`search`.
- A running [Qdrant](https://qdrant.tech/) for `embed`/`search`.
