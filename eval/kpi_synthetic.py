"""Score the three PoC KPIs (and the retrieval KPIs) on the synthetic EE210 course.

Runs the real app in-process (routers, ingestion, retrieval, model calls) against a temporary
SQLite database and a temporary knowledge base, so nothing real is touched. Needs the model
server (Ollama) with the chat model and the embedding model, plus aiosqlite.

    python eval/kpi_synthetic.py --out eval/reports/kpi_synthetic            # K1, K2, K3 + retrieval A/B
    python eval/kpi_synthetic.py --only k1 --out ...                          # one KPI

What is real and what is not:
  * The course, the 50 questions, the 20 timing queries and the 30 coding exercises were written
    by the AI team (eval/synthetic/). They are not the instructors' held-out set.
  * K1 is rated automatically (rater A: every key fact present, cited from the right module).
    A second, human reading (rater B) is added afterwards with --rater-b <json>.
  * K2 measures Saarthi's side for real and MODELS the manual-search side from the actual PDF
    text with stated assumptions (see manual_seconds). It is an estimate, not a stopwatch study.
  * K3 runs the code Saarthi writes against the test cases in the app's own code runner.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval" / "synthetic"))

import kpi_sets as KS  # noqa: E402
from ee210_course import COURSE_TITLE, MODULES, write_pdfs  # noqa: E402

READ_WPM = 238      # average silent reading speed for non-fiction (Brysbaert, 2019)
TYPE_CPS = 3.3      # about 40 words per minute of typing
CHECK_SOURCE_S = 10  # open the cited page and glance at it


# ── app set-up ────────────────────────────────────────────────────────────────

def build_app(tmp: Path):
    os.environ.setdefault("JWT_SECRET", "kpi-synthetic-secret-0123456789abcdef")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from saarthi_backend.main import saarthi_backend_exception_handler  # noqa: F401  (chdirs to repo root)
    from saarthi_backend.model import Base, User
    from saarthi_backend.routers import api_router
    from saarthi_backend.service import indexing_service as ix
    from saarthi_backend.utils.exceptions import SaarthiBackendError
    import src.cache.semantic_cache as sc
    import src.telemetry.traces as traces
    import src.utils.budget as budget

    os.chdir(tmp)  # after importing main: relative knowledge_base/ now means tmp/knowledge_base
    ix._KB_ROOT = tmp / "knowledge_base" / "courses"
    ix._VIDEO_KB_ROOT = tmp / "knowledge_base" / "videos"
    traces.TRACE_DIR = tmp / "traces"
    sc.DB_PATH = tmp / "cache" / "semantic_cache.sqlite"
    sc._singleton = None
    budget._USAGE_FILE = tmp / "cache" / "llm_usage.json"
    uploads = tmp / "uploads"
    uploads.mkdir()
    for name in ("course_router", "teacher_router"):
        sys.modules[f"saarthi_backend.routers.{name}"]._UPLOAD_DIR = uploads

    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp / 'kpi.db').as_posix()}")

    @event.listens_for(engine.sync_engine, "connect")
    def _wal(conn, _rec):
        conn.cursor().execute("PRAGMA journal_mode=WAL")

    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)

    async def setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with Session() as s:
            t = User(email="teacher@ee210.test", password_hash="x", full_name="EE210 Teacher", role="teacher")
            st = User(email="student@ee210.test", password_hash="x", full_name="EE210 Student", role="student")
            s.add_all([t, st])
            await s.commit()
            return t.id, st.id

    tid, sid = asyncio.run(setup())
    app = FastAPI()
    app.state.db_session_factory = Session
    app.include_router(api_router, prefix="/api")
    app.add_exception_handler(SaarthiBackendError, saarthi_backend_exception_handler)
    return TestClient(app, raise_server_exceptions=False), tid, sid


def headers(uid: int) -> dict:
    from saarthi_backend.utils.jwt_utils import create_access_token
    return {"Authorization": f"Bearer {create_access_token(str(uid))}"}


def create_course(client, T: dict, S: dict, pdfs: list[Path]) -> int:
    cid = client.post("/api/courses", json={"title": COURSE_TITLE, "code": "EE210", "instructor": "EE210 Teacher"},
                      headers=T).json()["id"]
    mids = []
    for title, pdf in zip(MODULES, pdfs):
        up = client.post("/api/courses/upload", files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")},
                         headers=T).json()
        mids.append(client.post(f"/api/courses/{cid}/materials",
                                json={"title": title, "type": "pdf", "url": "http://localhost:8000" + up["url"]},
                                headers=T).json()["id"])
    for mid in mids:
        for _ in range(600):
            st = client.get(f"/api/courses/{cid}/materials/{mid}/index-status", headers=T).json()
            if st["status"] not in ("queued", "running"):
                break
            time.sleep(0.25)
        if st["status"] != "done":
            raise SystemExit(f"material {mid} not indexed: {st}")
    code = client.post(f"/api/courses/{cid}/invite", json={"email": "*"}, headers=T).json()["inviteCode"]
    client.post("/api/courses/join", json={"inviteCode": code}, headers=S)
    return int(cid)


def ask(client, S: dict, course_id: int, question: str, answer_mode: str = "direct") -> dict:
    body = {"message": question, "courseId": course_id, "conversationHistory": [], "answerMode": answer_mode}
    t0 = time.perf_counter()
    r = client.post("/api/chat/stream", json=body, headers=S)
    seconds = time.perf_counter() - t0
    text, meta, event = [], {}, None
    for line in r.text.splitlines():
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data = line[6:] if line.startswith("data: ") else line[5:]  # SSE strips exactly one space; tokens keep theirs
            if data == "[DONE]":
                continue
            if event == "meta":
                try:
                    meta = json.loads(data)
                except ValueError:
                    pass
            elif event is None:
                text.append(data.replace("\\n", "\n"))
            event = None
    return {"status": r.status_code, "seconds": round(seconds, 2), "answer": "".join(text), "meta": meta}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def plain(answer: str) -> str:
    """LaTeX to plain text before the answer-key regexes, so "$20 \\text{ ms}$" or
    "\\frac{1}{T}" count the same as "20 ms" and "1/T"."""
    s = answer or ""
    for _ in range(3):  # nested \frac
        s = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\(text|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}", r"\2", s)
    s = re.sub(r"\\(left|right|displaystyle)", "", s)
    s = re.sub(r"\\[,;:! ]", " ", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\pi", "pi").replace("\\pm", "+/-")
    s = s.replace("\\{", "{").replace("\\}", "}").replace("$", " ")
    s = re.sub(r"\((\w+)\)/\((\w+)\)", r"\1/\2", s)  # (1)/(T) -> 1/T
    return re.sub(r"[ \t]+", " ", s)


# ── K1 ───────────────────────────────────────────────────────────────────────

def run_k1(client, S, cid) -> list[dict]:
    rows = []
    for qid, level, q, model, keys, (module, page) in KS.K1:
        res = ask(client, S, cid, q)
        ans = res["answer"]
        refused = res["meta"].get("reply_mode") == "not_covered"
        key_hits = [bool(re.search(k, plain(ans), re.I)) for k in keys]
        cites = res["meta"].get("citations") or []
        mod_cites = [c for c in cites if _norm(module) in _norm(c.get("source_file", ""))]
        right_module = bool(mod_cites)
        right_page = any(str(c.get("page_number")) == str(page) for c in mod_cites)
        if refused or not any(key_hits):
            rating = "I"
        elif all(key_hits) and right_module:
            rating = "C"
        else:
            rating = "P"
        rows.append({
            "id": qid, "module": module, "page": page, "level": level, "question": q, "model_answer": model,
            "answer": ans, "seconds": res["seconds"], "reply_mode": res["meta"].get("reply_mode"),
            "tier": res["meta"].get("tier"), "requested_tier": res["meta"].get("requested_tier"),
            "confidence": res["meta"].get("confidence"), "refused": refused,
            "keys_all": all(key_hits), "keys_any": any(key_hits), "right_module_cited": right_module,
            "right_page_cited": right_page, "citations": [(c.get("source_file"), c.get("page_number")) for c in cites],
            "rater_a": rating,
        })
        print(f"K1 {qid:6} {rating}  {res['seconds']:5.1f}s  {q[:60]}", flush=True)
    return rows


def retrieval_ab(cid: int) -> dict:
    """Top-3 source hit and precision@3 on the 50 K1 questions, without and with the cross-encoder."""
    from src.retrieval import rerankers, store
    from src.tools.retrieve import retrieve

    out = {}
    for label, flag in (("no_rerank", "0"), ("rerank", "1")):
        os.environ["RETRIEVAL_RERANK"] = flag
        hits, precs, times = [], [], []
        for qid, level, q, model, keys, (module, page) in KS.K1:
            t = time.perf_counter()
            chunks = retrieve(q, course_id=cid, k=3)[:3]
            times.append((time.perf_counter() - t) * 1000)
            rel = [(_norm(module) in _norm(c.source_file or c.title or "")) and c.page == page for c in chunks]
            hits.append(any(rel))
            precs.append(sum(rel) / 3)
        status = rerankers.cross_encoder_order("probe", ["a", "b"])[1] if flag == "1" else "off"
        out[label] = {"hit@3": round(sum(hits) / len(hits), 4), "precision@3": round(sum(precs) / len(precs), 4),
                      "median_ms": round(statistics.median(times), 1), "reranker": status}
    os.environ.pop("RETRIEVAL_RERANK", None)
    a, b = out["no_rerank"]["precision@3"], out["rerank"]["precision@3"]
    out["lift_abs"] = round(b - a, 4)
    out["lift_rel"] = round((b - a) / a, 4) if a else None
    return out


# ── K2 ───────────────────────────────────────────────────────────────────────

def _pdf_pages(pdf: Path) -> list[str]:
    from pypdf import PdfReader
    return [p.extract_text() or "" for p in PdfReader(str(pdf)).pages]


def manual_seconds(module_pages: list[str], page: int, keyword: str, calc: bool, library: str) -> dict:
    """Modelled time for a student to find and confirm the answer by hand. Assumptions:
      open the right file: 15 s (EE210, 4 clearly named PDFs) or 45 s (spec-sized library of
        20-50 PDFs per course: about three files opened before the right one)
      Ctrl+F: 5 s to type, 3 s per match visited before the right page (x3 in the spec-sized
        library, whose notes are longer); if the word is not on the right page, skim the file at
        450 words per minute instead
      read half of the target page at 238 wpm; +30 s when a small calculation is needed."""
    choose = 15 if library == "ee210" else 45
    kw = keyword.lower()
    before = sum(p.lower().count(kw) for p in module_pages[: page - 1])
    on_page = module_pages[page - 1].lower().count(kw)
    if on_page:
        find = 5 + 3 * (before + 1) * (1 if library == "ee210" else 3)
    else:
        words = sum(len(p.split()) for p in module_pages)
        find = 5 + words / 450 * 60 * (1 if library == "ee210" else 3)
    read = len(module_pages[page - 1].split()) * 0.5 / READ_WPM * 60
    total = choose + find + read + (30 if calc else 0)
    return {"choose": choose, "find": round(find, 1), "read": round(read, 1), "calc": 30 if calc else 0,
            "total": round(total, 1)}


def run_k2(client, S, cid, pdfs: dict[str, Path]) -> dict:
    pages = {m: _pdf_pages(p) for m, p in pdfs.items()}
    rows = []
    for q, module, page, kw, calc, key in KS.K2:
        res = ask(client, S, cid, q)
        ok = bool(re.search(key, plain(res["answer"]), re.I)) and res["meta"].get("reply_mode") == "grounded"
        words = len(res["answer"].split())
        saarthi = len(q) / TYPE_CPS + res["seconds"] + words / READ_WPM * 60 + CHECK_SOURCE_S
        m_small = manual_seconds(pages[module], page, kw, calc, "ee210")
        m_spec = manual_seconds(pages[module], page, kw, calc, "spec")
        rows.append({"question": q, "correct": ok, "wait_s": res["seconds"], "answer_words": words,
                     "saarthi_s": round(saarthi, 1), "manual_ee210": m_small, "manual_spec": m_spec,
                     "ratio_ee210": round(m_small["total"] / saarthi, 2), "ratio_spec": round(m_spec["total"] / saarthi, 2)})
        print(f"K2 {'ok ' if ok else 'BAD'} saarthi {saarthi:5.1f}s manual {m_small['total']:5.1f}/{m_spec['total']:5.1f}s  {q[:55]}", flush=True)
    return {
        "rows": rows,
        "correct": sum(r["correct"] for r in rows),
        "median_saarthi_s": statistics.median(r["saarthi_s"] for r in rows),
        "median_wait_s": statistics.median(r["wait_s"] for r in rows),
        "median_ratio_ee210": statistics.median(r["ratio_ee210"] for r in rows),
        "median_ratio_spec": statistics.median(r["ratio_spec"] for r in rows),
    }


# ── K3 ───────────────────────────────────────────────────────────────────────

_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S | re.I)


def extract_code(answer: str, name: str) -> str | None:
    blocks = _BLOCK.findall(answer or "")
    for b in blocks:
        if f"def {name}" in b:
            return b
    if f"def {name}" in (answer or ""):  # code without a fence
        start = answer.index(f"def {name}")
        return answer[start:]
    return None


def run_k3(client, S, cid) -> list[dict]:
    from saarthi_backend.service.sandbox import run_code

    rows = []
    for name, task, tests in KS.K3:
        prompt = (f"Write a Python function {task}. Use NumPy where it helps. "
                  f"Show the complete function `{name}` in one Python code block.")
        res = ask(client, S, cid, prompt)
        code = extract_code(res["answer"], name)
        row = {"name": name, "seconds": res["seconds"], "reply_mode": res["meta"].get("reply_mode"),
               "code_found": code is not None, "ran": False, "passed": False, "error": None}
        if code is None:
            row["error"] = "no code block with the function" if res["meta"].get("reply_mode") != "not_covered" else "refused (not covered)"
        else:
            program = "import numpy as np\n" + code + "\n\n" + tests + "\nprint('ALL_TESTS_PASSED')\n"
            out = run_code("python", program, timeout=10)
            row["ran"] = "SyntaxError" not in (out.get("stderr") or "") and "Execution timed out" not in (out.get("stderr") or "")
            row["passed"] = out.get("exit_code") == 0 and "ALL_TESTS_PASSED" in (out.get("stdout") or "")
            if not row["passed"]:
                err = (out.get("stderr") or out.get("stdout") or "").strip().splitlines()
                row["error"] = err[-1][:200] if err else "failed"
        row["code"] = code
        rows.append(row)
        print(f"K3 {'PASS' if row['passed'] else 'FAIL'} {name:30} {row['error'] or ''}", flush=True)
    return rows


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="folder for the JSON results")
    ap.add_argument("--only", choices=["k1", "k2", "k3", "retrieval"], default=None)
    args = ap.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    pdfs = write_pdfs(Path(tempfile.mkdtemp(prefix="ee210_pdf_")))
    tmp = Path(tempfile.mkdtemp(prefix="saarthi_kpi_"))
    client, tid, sid = build_app(tmp)
    T, S = headers(tid), headers(sid)
    t0 = time.perf_counter()
    cid = create_course(client, T, S, pdfs)
    print(f"course {cid} indexed in {time.perf_counter() - t0:.1f}s", flush=True)
    from src.utils.warmup import warm_up
    print("warm-up:", warm_up(tmp / "knowledge_base"), flush=True)

    from src.utils.llm import chat_model_name, embedding_model_name
    info = {"chat_model": chat_model_name(tier=1), "embedding_model": embedding_model_name(),
            "date": time.strftime("%Y-%m-%d %H:%M"), "course": COURSE_TITLE}
    (args.out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    if args.only in (None, "retrieval"):
        ab = retrieval_ab(cid)
        print("retrieval A/B:", ab, flush=True)
        (args.out / "retrieval_ab.json").write_text(json.dumps(ab, indent=2), encoding="utf-8")
    if args.only in (None, "k1"):
        k1 = run_k1(client, S, cid)
        (args.out / "k1.json").write_text(json.dumps(k1, indent=2, ensure_ascii=False), encoding="utf-8")
        c = sum(r["rater_a"] == "C" for r in k1)
        print(f"K1 rater A: {c}/50 Correct (pass at 45)", flush=True)
    if args.only in (None, "k2"):
        k2 = run_k2(client, S, cid, dict(zip(MODULES, pdfs)))
        (args.out / "k2.json").write_text(json.dumps(k2, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"K2 median ratio: {k2['median_ratio_ee210']}x (EE210 size), {k2['median_ratio_spec']}x (spec-sized library)", flush=True)
    if args.only in (None, "k3"):
        k3 = run_k3(client, S, cid)
        (args.out / "k3.json").write_text(json.dumps(k3, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"K3: {sum(r['passed'] for r in k3)}/30 pass all tests (pass at 27)", flush=True)

    os.chdir(ROOT)
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
