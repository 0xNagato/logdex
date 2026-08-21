#!/usr/bin/env python3
"""
logdex - turn huge logs/files into context-friendly indexes and vectors.

Two layers, deliberately separate:

  1. MANIFEST (always)  - a small .logdex.jsonl. One record per chunk: line range,
     byte range, time range, a deterministic summary. Greppable. No embeddings,
     no network, no model. This alone answers most questions.

  2. VECTORS (optional) - embed each chunk's SUMMARY (never its bulk text) into
     Qdrant, payload = the manifest record. Search returns a REFERENCE
     (path + line range + summary), so the agent reads only the slice it needs.

The point: context stays small because retrieval hands back pointers, not payload.

Usage:
    logdex.py index  <file> [--chunk-lines 400] [--out PATH]
    logdex.py embed  <manifest.logdex.jsonl> [--collection logdex]
    logdex.py search "<query>" [--collection logdex] [-k 8]
    logdex.py show   <manifest.logdex.jsonl> <chunk_id>

Env:
    QDRANT_URL      (default http://localhost:6333)
    QDRANT_API_KEY  (default: read from /etc/qdrant/config.yaml)
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

DEFAULT_CHUNK_LINES = 400
# A chunk is a thing an agent will READ. Cap its bytes so a "reference" never
# turns into a context bomb: 400 lines of a wide JSONL can be ~900 KB.
DEFAULT_MAX_CHUNK_BYTES = 32 * 1024
UPSERT_BATCH = 64

TS_FIELDS = ("ts_ms", "ts", "timestamp", "time", "at", "created_at")
ISO_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})")
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


def ensure_collection(name):
    existing = set(c["name"] for c in qreq("GET", "/collections")["result"]["collections"])
    if name in existing:
        return
    qreq("PUT", "/collections/" + name,
         {"vectors": {VECTOR_NAME: {"size": VECTOR_SIZE, "distance": "Cosine"}}})
    print("created collection '%s'" % name)


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
                # "e" first: some structured logs name the event in a one-letter
                # field. Bound the length so a free-text field can't pollute the histogram.
                for f in ("e", "event", "evt", "kind", "type", "msg_type", "action"):
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

def load_embedder():
    try:
        from fastembed import TextEmbedding
    except ImportError:
        raise SystemExit("fastembed not installed. pip install fastembed")
    return TextEmbedding(model_name=EMBED_MODEL)


def point_id(chunk_id):
    h = hashlib.sha256(chunk_id.encode()).hexdigest()[:32]
    return "%s-%s-%s-%s-%s" % (h[:8], h[8:12], h[12:16], h[16:20], h[20:32])


def embed_manifest(manifest, collection):
    records = [json.loads(l) for l in open(manifest) if l.strip()]
    if not records:
        raise SystemExit("%s is empty" % manifest)
    ensure_collection(collection)
    model = load_embedder()
    done = 0
    for i in range(0, len(records), UPSERT_BATCH):
        batch = records[i:i + UPSERT_BATCH]
        texts = ["%s lines %d-%d :: %s" % (r["path"], r["start_line"], r["end_line"], r["summary"])
                 for r in batch]
        vecs = list(model.embed(texts))
        points = []
        for r, v in zip(batch, vecs):
            payload = dict(r)
            payload["source"] = "logdex"
            points.append({"id": point_id(r["chunk_id"]),
                           "vector": {VECTOR_NAME: [float(x) for x in v]},
                           "payload": payload})
        qreq("PUT", "/collections/%s/points?wait=true" % collection, {"points": points})
        done += len(points)
        print("  upserted %d/%d" % (done, len(records)))
    print("done. %d chunk-summaries in '%s'" % (done, collection))


def search(query, collection, k):
    model = load_embedder()
    vec = [float(x) for x in list(model.embed([query]))[0]]
    res = qreq("POST", "/collections/%s/points/search" % collection,
               {"vector": {"name": VECTOR_NAME, "vector": vec}, "limit": k, "with_payload": True})
    hits = res.get("result", [])
    if not hits:
        print("no hits")
        return
    for h in hits:
        p = h.get("payload", {})
        when = ""
        if p.get("ts_min"):
            when = " [%s .. %s]" % (p.get("ts_min"), p.get("ts_max"))
        print("\nscore %.4f  %s:%s-%s%s" % (h.get("score", 0.0), p.get("path"),
                                            p.get("start_line"), p.get("end_line"), when))
        print("  " + (p.get("summary", "") or "")[:400])
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


if __name__ == "__main__":
    main()
