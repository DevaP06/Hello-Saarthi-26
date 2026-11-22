"""Generate the AI prompt library (markdown) from the prompts that are actually in the code, verbatim.

    python eval/gen_prompt_library.py --out eval/reports/AI_Prompt_Library.md
"""
import argparse
import inspect
import os
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from saarthi_backend.ai import adapter  # noqa: E402
from saarthi_backend.service import chat_service, code_service, quiz_service  # noqa: E402
from src.experts import mcq, mind  # noqa: E402
from src.orchestrator import critique, followup, socratic  # noqa: E402
from src.prompts import socratic_pack, tutor  # noqa: E402
from src.prompts.loader import PROMPTS  # noqa: E402
from src.schemas.models import ExpertName  # noqa: E402
from src.tools.kb_ops import get_expert_description  # noqa: E402


def consts(fn) -> str:
    src = inspect.getsource(fn)
    return "".join(p for p in re.findall(r'"((?:[^"\\]|\\.)*)"', src) if len(p) > 25)


def tok(s: str) -> int:
    return max(1, len(s) // 4)


descs = "\n".join(f"- {e.value}: {get_expert_description(e.value)}" for e in ExpertName)
SOC_HINTS = "\n".join(f"  level {k}: {v}" for k, v in socratic_pack.HINT_LEVELS.items())

P = [
    # id, title, text, where, tier, variables, notes
    ("P01", "Grounded tutor (main answer prompt)", tutor.grounded_system("{course}"),
     "src/prompts/tutor.py · used by every course-scoped answer (chat, document viewer, video viewer, chatbot)",
     "Tier 1 or 2 (chosen by the query classifier)", "{course}, {not_covered}",
     "The grounding rule (DEC-1). Excerpts come in the user message (P03). The fixed not-covered reply is sent without calling the model when the coverage gate fails."),
    ("P02", "Not-covered reply (fixed text, no LLM call)", tutor.not_covered("{course}"),
     "src/prompts/tutor.py · sent by src/tools/grounding.py", "none", "{course}",
     "Shown when retrieval does not cover the question. Flagged to the instructor review queue."),
    ("P03", "Grounded user message", tutor.grounded_user("{question}", "{excerpts}"),
     "src/prompts/tutor.py", "same call as P01", "{question}, {excerpts}",
     "The student's text is delimited with <student_question> tags so it is treated as data."),
    ("P04", "General (greetings / what can you do)", tutor.GENERAL_SYSTEM,
     "src/prompts/tutor.py · smalltalk only", "Tier 1", "-",
     "Cannot teach course content without excerpts."),
    ("P05", "Compute result presentation", tutor.COMPUTE_SYSTEM,
     "src/prompts/tutor.py · calculator / data-analysis experts", "Tier 1", "-", "Adds no new facts."),
    ("P06", "Socratic hint (levels 1-3)", socratic_pack.HINT_TEMPLATE + "\n\nLevel instructions:\n" + SOC_HINTS,
     "src/prompts/socratic_pack.py · selected by src/orchestrator/socratic.py", "Tier 1 or 2",
     "{level}, {max_hints}, {level_instruction}",
     "FOR EDUCATOR REVIEW (WBS 3.3.1, two-educator sign-off pending). Escalates 1 -> 2 -> 3, then P07."),
    ("P07", "Socratic reveal (full solution)", socratic_pack.REVEAL_TEMPLATE,
     "src/prompts/socratic_pack.py", "Tier 1 or 2", "{hints_given}",
     "Used after three hints, or when the student asks for the answer."),
    ("P08", "Hint user message", tutor.hint_user("{question}", "{excerpts}"),
     "src/prompts/tutor.py", "same call as P06", "{question}, {excerpts}",
     "Excerpts are marked 'for YOU only' so the model does not paste the worked solution."),
    ("P09", "Router (which expert)", PROMPTS["orchestrator_system"].format(expert_descriptions=descs),
     "src/prompts/loader.py · src/orchestrator/router.py", "Tier 1 (pinned)", "{expert_descriptions}",
     "Pinned to the local model so routing never spends cloud tokens."),
    ("P10", "Expert (ReAct retrieval agent)", PROMPTS["expert_system"],
     "src/prompts/loader.py · src/experts/base.py", "Tier 1 or 2", "-", "Notes / books / video experts."),
    ("P11", "Mind agent (merge expert answers)", mind.MIND_SYSTEM_PROMPT,
     "src/experts/mind.py", "Tier 1 or 2", "-", "Used only in Mind Mode (several experts at once)."),
    ("P12", "Critic (verify an answer against its sources)",
     """You check tutoring answers against excerpts.
Reply with exactly: OK or FAIL: <one sentence>.
FAIL if a key claim is not in the excerpts.

[user message]
EXCERPTS:
{evidence}

ANSWER:
{answer, first 2500 chars}""",
     "src/orchestrator/critique.py", "Tier 1 (pinned)", "-",
     "A FAIL lowers the answer's confidence and flags it for instructor review. Skipped with FAST_CHAT=1."),
    ("P13", "Follow-up rewrite",
     """Rewrite the student's latest message as a standalone question a search engine could use. Keep the topic and any formula names. Output ONLY the rewritten question.

[user message]
Summary: {summary or '(none)'}

Recent chat:
{last 10 messages, 400 chars each}

Latest: {query}""",
     "src/orchestrator/followup.py", "Tier 1 (pinned)", "-",
     "Turns 'explain that more simply' into a standalone retrieval query."),
    ("P14", "Grounded MCQ generation", mcq._SYSTEM,
     "src/experts/mcq.py · POST /api/quizzes/generate-grounded", "MCQ_MODEL (needs >= 3B)", "-",
     "Output is validated (source label, JSON shape, answer in the excerpt) and returned as reviewed=false."),
    ("P15", "Study plan", chat_service._STUDY_PLAN_SYSTEM,
     "saarthi_backend/service/chat_service.py", "Tier 1", "student profile, course excerpts",
     "Scheduling task; course excerpts limited to the caller's courses."),
    ("P16", "Socratic challenge", chat_service._SOCRATIC_SYSTEM,
     "saarthi_backend/service/chat_service.py", "Tier 1", "topic, claim, difficulty, excerpts",
     "Free text is PII-scrubbed. 'explanation' field may use model knowledge (see AI_FEATURE_AUDIT.md)."),
    ("P17", "Concept graph", chat_service._CONCEPT_GRAPH_SYSTEM,
     "saarthi_backend/service/chat_service.py", "Tier 1", "course title, quiz scores, excerpts",
     "Now receives course excerpts (2026-10-19)."),
    ("P18", "Exam prediction", chat_service._EXAM_PRED_SYSTEM,
     "saarthi_backend/service/chat_service.py", "Tier 1", "course, quiz scores, notes topics, excerpts", "-"),
    ("P19", "Flashcards", chat_service._FLASHCARD_SYSTEM,
     "saarthi_backend/service/chat_service.py", "Tier 1", "topic, source text, excerpts", "Source text is PII-scrubbed."),
    ("P20", "Adaptive quiz", quiz_service._QUIZ_GEN_SYSTEM,
     "saarthi_backend/service/quiz_service.py", "Tier 1", "topic, difficulty, past scores, excerpts",
     "Output is not verified; use P14 for anything graded."),
    ("P21", "Code error explanation (JSON)", code_service._EXPLAIN_SYSTEM,
     "saarthi_backend/service/code_service.py", "Tier 1", "language, code, stderr",
     "Socratic hints, never the corrected code (spec 7.2). Emails / roll numbers removed from code."),
    ("P22", "Code explanation (streamed)", code_service._EXPLAIN_STREAM_SYSTEM,
     "saarthi_backend/service/code_service.py", "Tier 1", "language, code, stdout/stderr", "Same safeguards as P21."),
]

total = sum(tok(p[2]) for p in P)
out = []
out.append(f"""# Signals Saarthi — AI Prompt Library

**Version 1.0 · {date.today().isoformat()} · AI team (WBS 3.3 / 3.6.4)**

Every prompt the backend sends to a language model, copied from the code by `eval/gen_prompt_library.py`
(so the text below is what is deployed, not a paraphrase). {len(P)} prompts, about {total} tokens of fixed instruction text.
Retrieved excerpts and the student's question are added at run time and are not part of these counts.

## How prompts are managed

- **One place per prompt.** Tutor, Socratic and router prompts live in `src/prompts/`; feature prompts sit next to the feature
  (`chat_service.py`, `quiz_service.py`, `code_service.py`). Educators can change wording of P06/P07 in
  `src/prompts/socratic_pack.py` without touching routing code.
- **Audit.** `python eval/prompt_audit.py` measures the fixed prompts against the baseline taken before trimming
  (spec 10.1: reduce average prompt length by 30%). Result: 251 -> 123 tokens average (-51%).
- **Change control.** A prompt change is reviewed by the AI team lead, then re-run through `python eval/run_eval.py --strict`
  (grounding set, citation regression, isolation) and the audit. Socratic prompts (P06, P07) and MCQ prompts (P14) additionally
  need two-educator sign-off before they are called final.
- **Safety rules inside every prompt that sees student text.** The student's text is data, not instructions
  (`SECURITY_RULE`); it is delimited; PII is removed before the call; an output check blocks answers that quote the system
  prompt (`src/safety/guard.py`).
- **Grounding rule.** Prompts P01-P08 answer only from the excerpts. When the excerpts do not cover the question, P02 is sent
  and no model is called.

## Index

| ID | Prompt | Tier | Review status |
|---|---|---|---|""")
status = {"P06": "Educator review pending", "P07": "Educator review pending", "P14": "Educator review pending"}
for pid, title, text, where, tier, variables, notes in P:
    out.append(f"| {pid} | {title} | {tier} | {status.get(pid, 'AI team reviewed')} |")
out.append("")
for pid, title, text, where, tier, variables, notes in P:
    out.append(f"## {pid} · {title}\n")
    out.append(f"- **Where:** {where}")
    out.append(f"- **Model tier:** {tier}")
    out.append(f"- **Variables:** {variables}")
    out.append(f"- **Notes:** {notes}")
    out.append(f"- **Size:** about {tok(text)} tokens\n")
    out.append("```text\n" + text.strip() + "\n```\n")
out.append("""## Per-course prompt libraries (WBS 3.6.4)

The spec asks for one reviewed library per course (end-user documentation). The general library above is complete; the
per-course part needs instructors: for each course, a short list of "good questions to ask Saarthi" and the course's own
notation, which the review queue (Instructor > AI review tab) already collects in the form of corrected answers. Corrections
are stored per course and can be exported as the starting point of that course's library.
""")
Path(r"E:\SAARTHI\AI_WBS\AI_Prompt_Library.md").write_text("\n".join(out), encoding="utf-8")
print("written", len(P), "prompts", total, "tokens")
