"""Concurrent-student load test for the chat endpoint (spec: 5 concurrent sessions, Tier-1 p95 < 5 s,
Tier-2 p95 < 10 s). Talks plain HTTP to a running backend, so it works against a laptop, the demo
machine or the staging server. Standard library only.

    # tokens you already have (e.g. copied from the browser), or log students in:
    python eval/load_test.py --base-url http://localhost:8000 --course-id 1 \\
        --login student1@x.io:secret --login student2@x.io:secret

    python eval/load_test.py --base-url http://localhost:8000 --course-id 1 --tokens TOK1,TOK2 \\
        --students 5 --rounds 3 --out eval/reports/load.json --strict

Each simulated student is one thread with its own account (accounts are reused round-robin when you
give fewer than --students). A student asks the question list in a different order from the others,
so the run exercises retrieval, the semantic cache (--repeat re-asks a question) and the LLM at once.
Use --distinct for the worst case (every question is new, so every answer needs the LLM).

Measured per request, from the SSE stream: time to first byte, time to the first answer token, total
time, and the tier / cache / reply_mode reported in the `meta` event. The summary groups by tier.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERIES = ROOT / "eval" / "grounding_test.jsonl"


def _json_request(url: str, payload: dict, headers: dict | None = None, timeout: float = 30) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", **(headers or {})}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def login(base: str, email: str, password: str) -> str:
    data = _json_request(f"{base}/api/auth/signin", {"email": email, "password": password})
    token = data.get("access_token") or data.get("accessToken") or data.get("token")
    if not token:
        raise RuntimeError(f"login for {email} returned no token: {list(data)}")
    return token


def load_queries(path: Path, limit: int | None) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            q = json.loads(line).get("query")
            if q:
                out.append(q)
    return out[:limit] if limit else out


def ask(base: str, token: str, course_id: int | None, question: str, timeout: float) -> dict:
    """One streamed chat turn. Returns timings and what the stream said about itself."""
    body: dict = {"message": question, "conversationHistory": []}
    if course_id is not None:
        body["courseId"] = course_id
    req = urllib.request.Request(
        f"{base}/api/chat/stream",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}", "Accept": "text/event-stream"},
    )
    res = {"question": question, "ok": False, "status": None, "ttfb": None, "first_token": None, "total": None,
           "tier": None, "requested_tier": None, "cached": None, "reply_mode": None, "confidence": None, "chars": 0, "error": None}
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            res["status"] = r.status
            event = None
            done = False
            for raw in r:
                now = time.perf_counter() - t0
                if res["ttfb"] is None:
                    res["ttfb"] = now
                line = raw.decode("utf-8", "replace").rstrip("\n")
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data = line[6:] if line.startswith("data: ") else line[5:]  # SSE strips exactly one space; tokens keep theirs
                    if data == "[DONE]":
                        if not done:
                            res["total"] = now  # the answer is complete here; keep reading to the end like a browser
                        done = True
                        continue
                    if event == "meta":
                        try:
                            meta = json.loads(data)
                            res["tier"] = meta.get("tier")
                            res["requested_tier"] = meta.get("requested_tier")
                            res["cached"] = meta.get("cached")
                            res["reply_mode"] = meta.get("reply_mode")
                            res["confidence"] = meta.get("confidence")
                        except ValueError:
                            pass
                    elif event in (None, "message", "token"):
                        if res["first_token"] is None and data:
                            res["first_token"] = now
                        res["chars"] += len(data)
                    event = None
                elif line == "":
                    event = None
            res["ok"] = done and res["chars"] > 0
            if not res["ok"]:
                res["error"] = "stream ended without [DONE]" if not done else "empty answer"
    except urllib.error.HTTPError as e:
        res["status"] = e.code
        res["error"] = f"HTTP {e.code}"
    except Exception as e:  # timeouts, resets
        res["error"] = f"{type(e).__name__}: {e}"
    if res["total"] is None:
        res["total"] = time.perf_counter() - t0
    return res


def student(idx: int, args, token: str, questions: list[str], results: list, lock: threading.Lock, start: threading.Event):
    if args.distinct:  # no two students ask the same thing: nothing can come from the shared cache
        order = questions[idx::args.students] or questions[idx % len(questions):][:1]
    else:
        order = questions[idx % len(questions):] + questions[: idx % len(questions)]
    seq = []
    for _ in range(args.rounds):
        seq.extend(order)
    if args.repeat:
        seq.extend(order[: args.repeat])  # re-ask: should come back from the semantic cache
    start.wait()
    for q in seq:
        r = ask(args.base_url, token, args.course_id, q, args.timeout)
        r["student"] = idx
        with lock:
            results.append(r)
        if args.think_time:
            time.sleep(args.think_time)


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = max(0, min(len(values) - 1, int(round(p / 100 * (len(values) - 1)))))
    return values[k]


def summarize(results: list[dict], wall: float) -> dict:
    def stats(rows: list[dict], key: str) -> dict:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return {"p50": pct(vals, 50), "p95": pct(vals, 95), "max": max(vals) if vals else None,
                "mean": statistics.fmean(vals) if vals else None}

    groups: dict[str, list[dict]] = {}
    for r in results:
        label = "error" if not r["ok"] else f"tier{r['tier']}" if r["tier"] is not None else "tier?"
        groups.setdefault(label, []).append(r)
    by_tier = {k: {"n": len(v), "first_token": stats(v, "first_token"), "total": stats(v, "total")}
               for k, v in sorted(groups.items()) if k != "error"}
    ok = [r for r in results if r["ok"]]
    # The spec's latency targets are per query class: simple (Tier 1) < 5 s, complex (Tier 2) < 10 s.
    # A complex question served by the local model because no Tier 2 is configured is still complex.
    by_class: dict[str, list[dict]] = {}
    for r in ok:
        if r.get("cached"):
            by_class.setdefault("cache", []).append(r)
        elif r.get("requested_tier") in (1, 2):
            by_class.setdefault("simple" if r["requested_tier"] == 1 else "complex", []).append(r)
    by_class_stats = {k: {"n": len(v), "first_token": stats(v, "first_token"), "total": stats(v, "total")}
                      for k, v in sorted(by_class.items())}
    return {
        "requests": len(results),
        "ok": len(ok),
        "errors": [{"student": r["student"], "question": r["question"], "error": r["error"]} for r in results if not r["ok"]],
        "wall_seconds": round(wall, 1),
        "throughput_rps": round(len(results) / wall, 3) if wall else None,
        "all": {"first_token": stats(ok, "first_token"), "total": stats(ok, "total")},
        "by_tier": by_tier,
        "by_class": by_class_stats,
        "cache_hits": sum(1 for r in ok if r.get("cached")),
        "reply_modes": {m: sum(1 for r in ok if r["reply_mode"] == m) for m in {r["reply_mode"] for r in ok}},
    }


def fmt(v: float | None) -> str:
    return "  -  " if v is None else f"{v:5.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--course-id", type=int, default=None, help="scope every question to this course")
    ap.add_argument("--students", type=int, default=5, help="concurrent simulated students (spec: 5)")
    ap.add_argument("--login", action="append", default=[], metavar="EMAIL:PASSWORD", help="repeat per account")
    ap.add_argument("--tokens", default="", help="comma-separated access tokens instead of --login")
    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES, help="jsonl with a `query` field")
    ap.add_argument("--limit", type=int, default=6, help="questions taken from the file")
    ap.add_argument("--rounds", type=int, default=1, help="passes over the question list per student")
    ap.add_argument("--repeat", type=int, default=2, help="re-ask this many questions at the end (cache check)")
    ap.add_argument("--distinct", action="store_true",
                    help="give every student different questions (worst case: no shared-cache hits)")
    ap.add_argument("--think-time", type=float, default=0.0, help="seconds a student waits between questions")
    ap.add_argument("--timeout", type=float, default=120.0, help="per-request timeout")
    ap.add_argument("--tier1-p95", type=float, default=5.0, help="gate: Tier-1 p95 total seconds")
    ap.add_argument("--tier2-p95", type=float, default=10.0, help="gate: Tier-2 p95 total seconds")
    ap.add_argument("--out", type=Path, default=None, help="write the full report as JSON")
    ap.add_argument("--strict", action="store_true", help="exit 1 when a gate fails")
    args = ap.parse_args()
    args.base_url = args.base_url.rstrip("/")

    tokens = [t.strip() for t in args.tokens.split(",") if t.strip()]
    for cred in args.login:
        email, _, password = cred.partition(":")
        tokens.append(login(args.base_url, email, password))
    if not tokens:
        ap.error("give --login EMAIL:PASSWORD (repeatable) or --tokens")
    questions = load_queries(args.queries, args.limit)
    if not questions:
        ap.error(f"no queries in {args.queries}")

    print(f"{args.students} students -> {args.base_url}  course={args.course_id}  "
          f"{len(questions)} questions x {args.rounds} round(s) + {args.repeat} repeat(s) each")
    results: list[dict] = []
    lock = threading.Lock()
    start = threading.Event()
    threads = [
        threading.Thread(target=student, args=(i, args, tokens[i % len(tokens)], questions, results, lock, start), daemon=True)
        for i in range(args.students)
    ]
    for t in threads:
        t.start()
    t0 = time.perf_counter()
    start.set()  # all students begin together
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    rep = summarize(results, wall)
    print(f"\n{rep['ok']}/{rep['requests']} ok in {rep['wall_seconds']} s  ({rep['throughput_rps']} req/s)  "
          f"cache hits: {rep['cache_hits']}  reply modes: {rep['reply_modes']}")
    print("\n tier   n   first-token p50/p95    total p50/p95/max   (seconds)")
    for tier, s in rep["by_tier"].items():
        ft, tt = s["first_token"], s["total"]
        print(f" {tier:<5} {s['n']:>3}   {fmt(ft['p50'])} / {fmt(ft['p95'])}      {fmt(tt['p50'])} / {fmt(tt['p95'])} / {fmt(tt['max'])}")
    if rep["by_class"]:
        print("\n class    n   first-token p50/p95    total p50/p95/max   (by the classifier's requested tier)")
        for cls, s in rep["by_class"].items():
            ft, tt = s["first_token"], s["total"]
            print(f" {cls:<7} {s['n']:>3}   {fmt(ft['p50'])} / {fmt(ft['p95'])}      {fmt(tt['p50'])} / {fmt(tt['p95'])} / {fmt(tt['max'])}")
    for e in rep["errors"][:10]:
        print(f" ERROR student {e['student']}: {e['error']}  <- {e['question'][:60]}")

    gates = {"no_errors": not rep["errors"]}
    if rep["by_class"].get("simple") or rep["by_class"].get("complex"):
        t1, t2 = rep["by_class"].get("simple"), rep["by_class"].get("complex")
    else:  # older servers do not report the requested tier
        t1, t2 = rep["by_tier"].get("tier1"), rep["by_tier"].get("tier2")
    if t1:
        gates[f"tier1_p95<={args.tier1_p95}s"] = t1["total"]["p95"] is not None and t1["total"]["p95"] <= args.tier1_p95
    if t2:
        gates[f"tier2_p95<={args.tier2_p95}s"] = t2["total"]["p95"] is not None and t2["total"]["p95"] <= args.tier2_p95
    rep["gates"] = gates
    print("\n" + "  ".join(f"{'PASS' if ok else 'FAIL'} {name}" for name, ok in gates.items()))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items() if k not in ("login", "tokens")},
                                        "summary": rep, "requests": results}, indent=2), encoding="utf-8")
        print(f"report: {args.out}")
    return 1 if (args.strict and not all(gates.values())) else 0


if __name__ == "__main__":
    sys.exit(main())
