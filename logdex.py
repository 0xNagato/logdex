#!/usr/bin/env python3
"""
logdex - turn huge logs/files into context-friendly indexes and vectors.

Two layers, deliberately separate:

  1. MANIFEST (always)  - a small .logdex.jsonl. One record per chunk: line range,
     byte range, time range, a deterministic summary. Greppable. No embeddings,
     no network, no model. This alone answers most questions.

  2. VECTORS (hybrid, default) - embed each chunk's REAL text into Qdrant twice:
     a dense MiniLM vector for meaning + a BM25 sparse vector for exact terms,
     fused with Reciprocal Rank Fusion at query time. Payload = the manifest
     record, so search still returns a REFERENCE (path + line range + summary)
     and the agent reads only the slice it needs.

The point: context stays small because retrieval hands back pointers, not payload;
hybrid means an exact keyword match always counts even when dense meaning-match is weak.

Usage:
    logdex.py index  <file> [--chunk-lines 400] [--out PATH]
    logdex.py embed  <manifest.logdex.jsonl> [--collection logdex]
    logdex.py search "<query>" [--collection logdex] [-k 8]
    logdex.py show   <manifest.logdex.jsonl> <chunk_id>

Env:
    QDRANT_URL      (default http://localhost:6333)
    QDRANT_API_KEY  (default: read from /etc/qdrant/config.yaml)
    FASTEMBED_CACHE_PATH (default ~/.cache/fastembed)
"""

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

# Must match what mcp-server-qdrant/fastembed already writes, so anything indexed
# here is searchable by the qdrant MCP the agents already have.
VECTOR_NAME = "fast-all-minilm-l6-v2"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384
# Hybrid default: a BM25 sparse vector sits alongside the dense one so exact terms
# always count (dense MiniLM alone scored near-exact matches barely above random).
# IDF is applied server-side via the sparse `modifier: idf`.
SPARSE_NAME = "bm25"
SPARSE_MODEL = "Qdrant/bm25"
# fastembed's default cache is the OS temp folder, which macOS clears; a cleared model made embeds fail.
MODEL_CACHE = os.environ.get("FASTEMBED_CACHE_PATH", str(Path.home() / ".cache" / "fastembed"))
# Dense embed is capped (MiniLM sees ~256 tokens anyway); BM25 is bag-of-words with
# no token-window limit, so it reads much more of the chunk for keyword coverage.
DENSE_MAX_CHARS = int(os.environ.get("LOGDEX_DENSE_CHARS", "1600"))
SPARSE_MAX_CHARS = int(os.environ.get("LOGDEX_SPARSE_CHARS", "8000"))
# Qdrant's MCP server (mcp-server-qdrant) reads payload "document" and "metadata"; logdex writes both so
# its qdrant-find works on logdex collections, and logdex search accepts points that server stores.
DOCUMENT_MAX_CHARS = int(os.environ.get("LOGDEX_DOCUMENT_CHARS", "2000"))


def mcp_fields(rec, text):
    """The two payload fields mcp-server-qdrant reads: the chunk's text and where it came from."""
    return {"document": (text or rec.get("summary", "") or "")[:DOCUMENT_MAX_CHARS],
            "metadata": {"path": rec.get("path"), "lines": "%s-%s" % (rec.get("start_line"), rec.get("end_line")),
                         "source": "logdex"}}

DEFAULT_CHUNK_LINES = 400
# A chunk is a thing an agent will READ. Cap its bytes so a "reference" never
# turns into a context bomb: 400 lines of a wide JSONL can be ~900 KB.
DEFAULT_MAX_CHUNK_BYTES = 32 * 1024
UPSERT_BATCH = 64

TS_FIELDS = ("ts_ms", "ts", "timestamp", "time", "at", "created_at")
ISO_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})")
# Epoch-ms event times under keys the TS_FIELDS list misses, at any nesting depth.
MS_RE = re.compile(r'"(?:timestamp_ms|received_at_ms|observed_at_ms)":\s*(\d{13})\b')
LEVEL_RE = re.compile(r"\b(ERROR|WARN|WARNING|INFO|DEBUG|TRACE|PANIC|FATAL)\b")


# ---------------------------------------------------------------- qdrant io

def qdrant_url():
    return os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")


def qdrant_key():
    key = os.environ.get("QDRANT_API_KEY")
    if key:
        return key
    cfg = Path("/etc/qdrant/config.yaml")
    if cfg.exists():
        for line in cfg.read_text().splitlines():
            if "api_key" in line and ":" in line:
                return line.split(":", 1)[1].strip().strip('"').strip("'")
    return ""


def qreq(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(qdrant_url() + path, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    key = qdrant_key()
    if key:
        req.add_header("api-key", key)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise SystemExit("qdrant %s %s -> HTTP %s: %s" % (method, path, e.code, e.read().decode()[:400]))
    except urllib.error.URLError as e:
        raise SystemExit("qdrant unreachable at %s: %s" % (qdrant_url(), e.reason))


def collection_config(name):
    """Return the collection's config dict, or None if it does not exist."""
    try:
        return qreq("GET", "/collections/" + name).get("result")
    except SystemExit:
        return None


def has_sparse(name):
    cfg = collection_config(name)
    if not cfg:
        return False
    sv = (cfg.get("config", {}).get("params", {}) or {}).get("sparse_vectors") or {}
    return SPARSE_NAME in sv


def ensure_collection(name):
    """Create the collection with a dense + BM25 sparse vector if it does not exist.
    Never recreates an existing collection (that would drop data): to add the sparse
    field to a dense-only collection, recreate it explicitly, e.g.
        curl -X DELETE $QDRANT_URL/collections/<name>   then re-embed."""
    if collection_config(name):
        return
    qreq("PUT", "/collections/" + name, {
        "vectors": {VECTOR_NAME: {"size": VECTOR_SIZE, "distance": "Cosine"}},
        "sparse_vectors": {SPARSE_NAME: {"modifier": "idf"}},
    })
    print("created hybrid collection '%s' (dense %s + sparse %s)" % (name, VECTOR_NAME, SPARSE_NAME))


# ---------------------------------------------------------------- chunking

def iso_from_ms(ms):
    import datetime as _dt
    return _dt.datetime.fromtimestamp(ms / 1000.0, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def extract_ts(line, obj):
    """Best-effort ISO timestamp for a line, or None."""
    if isinstance(obj, dict):
        for f in TS_FIELDS:
            if f in obj:
                v = obj[f]
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)):
                    # Disambiguate seconds vs millis by magnitude, not by field
                    # name - both spellings appear with both units in the wild.
                    ms = float(v) * (1000.0 if float(v) < 1e11 else 1.0)
                    try:
                        return iso_from_ms(ms)
                    except (OverflowError, OSError, ValueError):
                        continue
                if isinstance(v, str):
                    m = ISO_RE.search(v)
                    if m:
                        return m.group(1).replace(" ", "T")
    m = ISO_RE.search(line)
    if m:
        return m.group(1).replace(" ", "T")
    m = MS_RE.search(line)
    if m:
        return iso_from_ms(int(m.group(1)))
    return None


def summarize(lines, keys, evts, lvls, ts_lo, ts_hi):
    """Deterministic summary. No LLM: this must run over gigabytes for free."""
    bits = []
    if ts_lo and ts_hi:
        bits.append(ts_lo if ts_lo == ts_hi else "%s .. %s" % (ts_lo, ts_hi))
    if evts:
        bits.append("events: " + ", ".join("%sx%d" % (k, v) for k, v in evts.most_common(6)))
    if lvls:
        bits.append("levels: " + ", ".join("%s:%d" % (k, v) for k, v in lvls.most_common(4)))
    if keys:
        bits.append("fields: " + ", ".join(k for k, _ in keys.most_common(10)))
    sample = [l.strip()[:180] for l in lines[:2] if l.strip()]
    if sample:
        bits.append("sample: " + " | ".join(sample))
    return " || ".join(bits)


def index_file(path, chunk_lines, out, max_chunk_bytes=DEFAULT_MAX_CHUNK_BYTES):
    """Stream the file once. Never load it whole - these run over 10GB+ files."""
    records = []
    state = {"idx": 0, "start_line": 1, "start_byte": 0, "n_json": 0}
    buf = []
    keys, evts, lvls = Counter(), Counter(), Counter()
    span = {"lo": None, "hi": None}
    byte_pos = 0

    def flush(end_line, end_byte):
        if not buf:
            return
        records.append({
            "chunk_id": "%s#%06d" % (path.name, state["idx"]),
            "path": str(path.resolve()),
            "start_line": state["start_line"],
            "end_line": end_line,
            "start_byte": state["start_byte"],
            "end_byte": end_byte,
            "lines": len(buf),
            "bytes": end_byte - state["start_byte"],
            "ts_min": span["lo"],
            "ts_max": span["hi"],
            "json_lines": state["n_json"],
            "summary": summarize(buf, keys, evts, lvls, span["lo"], span["hi"]),
        })
        state["idx"] += 1
        del buf[:]
        keys.clear(); evts.clear(); lvls.clear()
        span["lo"] = span["hi"] = None
        state["n_json"] = 0

    with open(path, "rb") as fh:
        for raw in fh:
            byte_pos += len(raw)
            line = raw.decode("utf-8", errors="replace")
            s = line.strip()
            obj = None
            if s.startswith("{"):
                try:
                    obj = json.loads(s)
                    state["n_json"] += 1
                except json.JSONDecodeError:
                    obj = None
            if isinstance(obj, dict):
                keys.update(obj.keys())
                # "e" first: some structured logs name the event in a one-letter field.
                # Bound the length so a free-text field can't pollute the histogram.
                for f in ("e", "event", "evt", "kind", "type", "msg_type", "action", "record_type"):
                    v = obj.get(f)
                    if isinstance(v, str) and 0 < len(v) <= 40:
                        evts[v] += 1
                        break
                for f in ("level", "lvl", "severity"):
                    if isinstance(obj.get(f), str):
                        lvls[obj[f].upper()] += 1
                        break
            else:
                m = LEVEL_RE.search(line)
                if m:
                    lvls[m.group(1)] += 1

            ts = extract_ts(line, obj)
            if ts:
                if span["lo"] is None or ts < span["lo"]:
                    span["lo"] = ts
                if span["hi"] is None or ts > span["hi"]:
                    span["hi"] = ts

            buf.append(line)
            cur_line = state["start_line"] + len(buf) - 1
            if len(buf) >= chunk_lines or (byte_pos - state["start_byte"]) >= max_chunk_bytes:
                flush(cur_line, byte_pos)
                state["start_line"] = cur_line + 1
                state["start_byte"] = byte_pos

        if buf:
            flush(state["start_line"] + len(buf) - 1, byte_pos)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    total = sum(r["bytes"] for r in records)
    msize = out.stat().st_size
    print("indexed %s" % path)
    print("  %d chunks over %s bytes (%.1f MB)" % (len(records), format(total, ","), total / 1e6))
    print("  manifest: %s (%s bytes)" % (out, format(msize, ",")))
    if total:
        print("  manifest is %.2f%% the size of the source" % (msize / total * 100))
    return records


# ---------------------------------------------------------------- embedding

_DENSE = None
_SPARSE = None


def load_embedder():
    global _DENSE
    if _DENSE is None:
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise SystemExit("fastembed not installed. pip install fastembed")
        _DENSE = TextEmbedding(model_name=EMBED_MODEL, cache_dir=MODEL_CACHE)
    return _DENSE


def load_sparse():
    global _SPARSE
    if _SPARSE is None:
        try:
            from fastembed import SparseTextEmbedding
        except ImportError:
            raise SystemExit("fastembed not installed. pip install fastembed")
        _SPARSE = SparseTextEmbedding(SPARSE_MODEL, cache_dir=MODEL_CACHE)
    return _SPARSE


def point_id(chunk_id):
    h = hashlib.sha256(chunk_id.encode()).hexdigest()[:32]
    return "%s-%s-%s-%s-%s" % (h[:8], h[8:12], h[12:16], h[16:20], h[20:32])


def slice_text(rec):
    """Read the chunk's real bytes from its source; fall back to the summary."""
    try:
        with open(rec["path"], "rb") as fh:
            fh.seek(rec["start_byte"])
            raw = fh.read(rec["end_byte"] - rec["start_byte"])
        return raw.decode("utf-8", "replace")
    except (OSError, KeyError):
        return rec.get("summary", "") or ""


def embed_manifest(manifest, collection):
    records = [json.loads(l) for l in open(manifest) if l.strip()]
    if not records:
        print("  (empty manifest, nothing to embed: %s)" % manifest)
        return 0
    ensure_collection(collection)
    dense = load_embedder()
    sparse = load_sparse()
    done = 0
    for i in range(0, len(records), UPSERT_BATCH):
        batch = records[i:i + UPSERT_BATCH]
        raws = [slice_text(r) for r in batch]
        summ = [(r.get("summary") or "") for r in batch]
        # Dense: file tag + summary + real text, capped to MiniLM's useful window.
        dense_texts = [("%s :: %s\n%s" % (os.path.basename(batch[j]["path"]), summ[j][:160], raws[j]))[:DENSE_MAX_CHARS]
                       for j in range(len(batch))]
        # Sparse BM25: summary (event/field names) + real text, over a wider window.
        sparse_texts = [("%s\n%s" % (summ[j], raws[j]))[:SPARSE_MAX_CHARS] for j in range(len(batch))]
        dvecs = list(dense.embed(dense_texts))
        svecs = list(sparse.embed(sparse_texts))
        points = []
        for r, raw, dv, sv in zip(batch, raws, dvecs, svecs):
            payload = {**r, **mcp_fields(r, raw)}
            payload["source"] = "logdex-hybrid"
            points.append({
                "id": point_id(r["chunk_id"]),
                "vector": {
                    VECTOR_NAME: [float(x) for x in dv],
                    SPARSE_NAME: {"indices": [int(x) for x in sv.indices],
                                  "values": [float(x) for x in sv.values]},
                },
                "payload": payload,
            })
        qreq("PUT", "/collections/%s/points?wait=true" % collection, {"points": points})
        done += len(points)
        print("  upserted %d/%d" % (done, len(records)))
    print("done. %d chunks (dense+bm25) in '%s'" % (done, collection))


def fill_sparse(collection):
    """Give points stored without a BM25 vector (mcp-server-qdrant writes dense only) one from their document
    text, so hybrid search ranks them on keywords too. Returns how many were filled."""
    if not has_sparse(collection):
        return 0
    sparse, offset, filled = None, None, 0
    while True:
        body = {"limit": 256, "with_payload": ["document"], "with_vector": [SPARSE_NAME],
                "filter": {"must_not": [{"key": "source", "match": {"value": "logdex-hybrid"}}]}}
        if offset is not None:
            body["offset"] = offset
        res = qreq("POST", "/collections/%s/points/scroll" % collection, body).get("result", {})
        todo = [pt for pt in res.get("points", [])
                if not (pt.get("vector") or {}).get(SPARSE_NAME) and (pt.get("payload") or {}).get("document")]
        if todo:
            sparse = sparse or load_sparse()
            vecs = list(sparse.embed([pt["payload"]["document"][:SPARSE_MAX_CHARS] for pt in todo]))
            qreq("PUT", "/collections/%s/points/vectors?wait=true" % collection, {"points": [
                {"id": pt["id"], "vector": {SPARSE_NAME: {"indices": [int(x) for x in sv.indices],
                                                          "values": [float(x) for x in sv.values]}}}
                for pt, sv in zip(todo, vecs)]})
            filled += len(todo)
        offset = res.get("next_page_offset")
        if offset is None:
            return filled


def search(query, collection, k):
    dvec = [float(x) for x in list(load_embedder().embed([query]))[0]]
    if has_sparse(collection):
        sv = list(load_sparse().query_embed(query))[0]
        pool = max(k * 4, 20)
        res = qreq("POST", "/collections/%s/points/query" % collection, {
            "prefetch": [
                {"query": dvec, "using": VECTOR_NAME, "limit": pool},
                {"query": {"indices": [int(x) for x in sv.indices],
                           "values": [float(x) for x in sv.values]},
                 "using": SPARSE_NAME, "limit": pool},
            ],
            "query": {"fusion": "rrf"},
            "limit": k,
            "with_payload": True,
        })
        hits = res.get("result", {}).get("points", [])
    else:
        # Dense-only fallback for a collection that predates the hybrid schema.
        res = qreq("POST", "/collections/%s/points/search" % collection,
                   {"vector": {"name": VECTOR_NAME, "vector": dvec}, "limit": k, "with_payload": True})
        hits = res.get("result", [])
    if not hits:
        print("no hits")
        return
    for h in hits:
        p = h.get("payload", {})
        when = ""
        if p.get("ts_min"):
            when = " [%s .. %s]" % (p.get("ts_min"), p.get("ts_max"))
        if p.get("path"):
            print("\nscore %.4f  %s:%s-%s%s" % (h.get("score", 0.0), p.get("path"),
                                                p.get("start_line"), p.get("end_line"), when))
        else:
            print("\nscore %.4f  (stored memory, no source file)" % h.get("score", 0.0))
        print("  " + (p.get("summary") or p.get("document") or "")[:400])
    print("\n-- read a hit with the Read tool (offset/limit) or: sed -n 'START,ENDp' PATH")


def show(manifest, chunk_id):
    for line in open(manifest):
        r = json.loads(line)
        if r["chunk_id"] == chunk_id or r["chunk_id"].endswith("#" + str(chunk_id)):
            with open(r["path"], "rb") as fh:
                fh.seek(r["start_byte"])
                sys.stdout.write(fh.read(r["end_byte"] - r["start_byte"]).decode("utf-8", "replace"))
            return
    raise SystemExit("chunk %s not found in %s" % (chunk_id, manifest))


# ---------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description="logdex - index huge files into references + vectors")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="build a chunk manifest for a large file")
    p.add_argument("file")
    p.add_argument("--chunk-lines", type=int, default=DEFAULT_CHUNK_LINES)
    p.add_argument("--max-chunk-bytes", type=int, default=DEFAULT_MAX_CHUNK_BYTES)
    p.add_argument("--out")

    p = sub.add_parser("embed", help="embed a manifest's summaries into qdrant")
    p.add_argument("manifest")
    p.add_argument("--collection", default="logdex")

    p = sub.add_parser("search", help="semantic search -> file:line references")
    p.add_argument("query")
    p.add_argument("--collection", default="logdex")
    p.add_argument("-k", type=int, default=8)

    p = sub.add_parser("show", help="print one chunk's raw text")
    p.add_argument("manifest")
    p.add_argument("chunk_id")

    p = sub.add_parser("fill-sparse", help="add BM25 vectors to points stored without one (e.g. by mcp-server-qdrant)")
    p.add_argument("--collection", default="logdex")

    a = ap.parse_args()
    if a.cmd == "index":
        src = Path(a.file)
        if not src.exists():
            raise SystemExit("no such file: %s" % src)
        out = Path(a.out) if a.out else Path(str(src) + ".logdex.jsonl")
        index_file(src, a.chunk_lines, out, a.max_chunk_bytes)
    elif a.cmd == "embed":
        embed_manifest(Path(a.manifest), a.collection)
    elif a.cmd == "search":
        search(a.query, a.collection, a.k)
    elif a.cmd == "show":
        show(Path(a.manifest), a.chunk_id)
    elif a.cmd == "fill-sparse":
        print("filled BM25 vectors on %d point(s) in '%s'" % (fill_sparse(a.collection), a.collection))


if __name__ == "__main__":
    main()
