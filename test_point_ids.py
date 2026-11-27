"""End-to-end check of logdex point ids against a throwaway Qdrant collection.

Run: python3 test_point_ids.py    Log: logs/test-point-ids.log (next to this file)
"""
import json, os, shutil, subprocess, sys, time, urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGDEX = HERE / "logdex.py"
LOG = HERE / "logs" / "test-point-ids.log"
DATA = HERE / "test-data-tmp"
COLL = "logdex_test_%d" % int(time.time())
QURL = os.environ.get("QDRANT_URL", "http://localhost:6333")
LAGOS = timezone(timedelta(hours=1))
passed = failed = 0


def log(msg):
    LOG.parent.mkdir(exist_ok=True)
    line = "%s %s" % (datetime.now(LAGOS).strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def q(method, url, body=None):
    req = urllib.request.Request(QURL + url, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["result"]


def count(path):
    flt = {"must": [{"key": "path", "match": {"value": str(Path(path).resolve())}}]}
    return q("POST", "/collections/%s/points/count" % COLL, {"filter": flt, "exact": True})["count"]


def logdex(*args):
    r = subprocess.run([sys.executable, str(LOGDEX)] + list(args), capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("logdex %s failed: %s" % (args[0], (r.stderr or r.stdout)[-300:]))
    return r.stdout


def add(path):
    m = DATA / ("%s.manifest.jsonl" % str(path.relative_to(DATA)).replace("/", "__"))
    logdex("index", str(path), "--chunk-lines", "10", "--out", str(m))
    logdex("embed", str(m), "--collection", COLL)


def write(path, n, tag):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join("%s line %d: a sentence about %s number %d\n" % (tag, i, tag, i) for i in range(1, n + 1)))


def expect(name, got, want):
    global passed, failed
    if got == want:
        passed += 1
        log("PASS: %s (%d chunks)" % (name, got))
    else:
        failed += 1
        log("FAIL: %s: expected %d chunks, found %d" % (name, want, got))


shutil.rmtree(DATA, ignore_errors=True)
a, b = DATA / "alpha" / "SKILL.md", DATA / "beta" / "SKILL.md"
log("start: logdex point-id test in throwaway collection %s" % COLL)
try:
    write(a, 30, "alpha"); write(b, 30, "beta")
    add(a); add(b)
    expect("two files with the same name in different folders: the first keeps all its chunks", count(a), 3)
    expect("two files with the same name in different folders: the second keeps all its chunks", count(b), 3)
    add(b)
    expect("re-embedding an unchanged file keeps the same chunks, no duplicates", count(b), 3)
    write(a, 10, "alpha"); add(a)
    expect("a file that shrank from 30 to 10 lines keeps only its current chunk", count(a), 1)
    b.unlink()
    logdex("prune-missing", "--collection", COLL)
    expect("prune-missing removes the chunks of a deleted file", count(b), 0)
    expect("prune-missing leaves the chunks of a file that still exists", count(a), 1)
finally:
    try:
        q("DELETE", "/collections/%s" % COLL)
        log("ok: deleted the throwaway collection %s" % COLL)
    except Exception as e:
        log("error: could not delete collection %s: %s" % (COLL, e))
    shutil.rmtree(DATA, ignore_errors=True)
log("verdict: %d passed, %d failed" % (passed, failed))
sys.exit(1 if failed else 0)
