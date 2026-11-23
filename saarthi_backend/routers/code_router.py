"""Code execution routes — Piston runner + AI error explanation."""

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from saarthi_backend.deps import get_current_user, get_db
from saarthi_backend.dao.notification_dao import NotificationDAO
from saarthi_backend.model import Course, CourseModule, Enrollment, User
from saarthi_backend.model.code_problem_model import CodeProblem, CodeSubmission
from saarthi_backend.schema.code_schemas import (
    CodeExecuteRequest,
    CodeExecuteResponse,
    CodeExplainRequest,
    CodeExplainResponse,
    CodeProblemCreate,
    CodeProblemResponse,
    CodeProblemUpdate,
    CodeSubmissionItem,
    CodeSubmitRequest,
    CodeSubmitResponse,
    TestCasePublic,
    TestCaseResult,
)
from saarthi_backend.service import code_service

router = APIRouter(prefix="/code", tags=["code"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_list(value: str | None) -> list[str]:
    """Parse a JSON-encoded list of strings; return [] on error or None."""
    if not value:
        return []
    try:
        result = json.loads(value)
        if isinstance(result, list):
            return [str(item) for item in result]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def _parse_json_dict(value: str | None) -> dict[str, str]:
    """Parse a JSON-encoded dict of strings; return {} on error or None."""
    if not value:
        return {}
    try:
        result = json.loads(value)
        if isinstance(result, dict):
            return {str(k): str(v) for k, v in result.items()}
    except (json.JSONDecodeError, TypeError):
        pass
    return {}


def _test_cases(p: CodeProblem) -> list[dict]:
    """The problem's test cases; a problem with none uses its expected output as one case."""
    try:
        cases = json.loads(p.test_cases_json) if p.test_cases_json else []
    except (json.JSONDecodeError, TypeError):
        cases = []
    cases = [c for c in cases if isinstance(c, dict) and "expectedOutput" in c]
    if not cases and p.expected_output:
        cases = [{"input": "", "expectedOutput": p.expected_output, "hidden": False}]
    return cases


def _normalize_output(text: str) -> str:
    """Compare outputs line by line, ignoring extra spaces and blank lines at the end."""
    lines = [" ".join(line.split()) for line in (text or "").replace("\r\n", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _problem_to_response(p: CodeProblem, course_title: str | None = None, can_edit: bool = False) -> CodeProblemResponse:
    """Convert a CodeProblem ORM row to CodeProblemResponse. Editors see hidden test cases in full."""
    return CodeProblemResponse(
        id=p.id,
        title=p.title,
        difficulty=p.difficulty,
        points=p.points,
        description=p.description,
        requirements=_parse_json_list(p.requirements_json),
        expectedOutput=p.expected_output,
        hints=_parse_json_list(p.hints_json),
        starterCode=_parse_json_dict(p.starter_code_json),
        topics=p.topics,
        sortOrder=p.sort_order,
        testCases=[
            TestCasePublic(hidden=True) if c.get("hidden") and not can_edit
            else TestCasePublic(input=c.get("input", ""), expectedOutput=c["expectedOutput"], hidden=bool(c.get("hidden")))
            for c in _test_cases(p)
        ],
        courseId=p.course_id,
        courseTitle=course_title,
        topic=p.topic,
        canEdit=can_edit,
    )


# ---------------------------------------------------------------------------
# Access: practice problems (no course) are for everyone; a course's problems are
# for its teacher and enrolled students. Only the course teacher (or an admin) edits.
# ---------------------------------------------------------------------------

_PROBLEM_ROW = select(CodeProblem, Course.title, Course.owner_id).outerjoin(Course, Course.id == CodeProblem.course_id)


def _normalize_topic(topic: str | None) -> str | None:
    t = (topic or "").strip()
    return None if not t or t == "No topic" else t


async def _member_course_ids(db: AsyncSession, user: User) -> list[int] | None:
    """Ids of the courses whose problems the user may see; None means every course (admin)."""
    if user.role == "admin":
        return None
    if user.role == "teacher":
        stmt = select(Course.id).where(Course.owner_id == user.id)
    else:
        stmt = select(Enrollment.course_id).where(Enrollment.user_id == user.id)
    return list((await db.execute(stmt)).scalars().all())


def _can_edit(user: User, p: CodeProblem, owner_id: int | None) -> bool:
    if user.role == "admin":
        return True
    return p.course_id is not None and user.role == "teacher" and owner_id == user.id


async def _visible_problem(
    db: AsyncSession, user: User, problem_id: int, *, active_only: bool = True,
) -> tuple[CodeProblem, str | None, bool]:
    """(problem, course title, can edit) when the user may see the problem; otherwise 404."""
    stmt = _PROBLEM_ROW.where(CodeProblem.id == problem_id)
    if active_only:
        stmt = stmt.where(CodeProblem.is_active == True)  # noqa: E712
    row = (await db.execute(stmt)).first()
    if row:
        p, course_title, owner_id = row
        ids = await _member_course_ids(db, user) if p.course_id is not None else None
        if ids is None or p.course_id in ids:
            return p, course_title, _can_edit(user, p, owner_id)
    raise HTTPException(status_code=404, detail="Problem not found")


async def _editable_problem(db: AsyncSession, user: User, problem_id: int, **kw) -> tuple[CodeProblem, str | None]:
    p, course_title, can_edit = await _visible_problem(db, user, problem_id, **kw)
    if not can_edit:
        raise HTTPException(status_code=403, detail="Only the course teacher can change this problem.")
    return p, course_title


@router.post("/execute", response_model=CodeExecuteResponse)
async def execute_code(
    body: CodeExecuteRequest,
    user: Annotated[User, Depends(get_current_user)],
):
    """
    Run code via Piston API and return stdout/stderr/exit code.
    If explainOnError=True and execution fails, also returns AI explanation.
    """
    return await code_service.execute_code(body)


@router.post("/explain/stream")
async def stream_explain(
    body: CodeExplainRequest,
    user: Annotated[User, Depends(get_current_user)],
):
    """
    Stream an AI explanation of a code error token-by-token via SSE.
    Call this after /execute returns a non-zero exit code for real-time feedback.
    """
    return StreamingResponse(
        code_service.stream_explain_error(body),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/explain", response_model=CodeExplainResponse)
async def explain_error(
    body: CodeExplainRequest,
    user: Annotated[User, Depends(get_current_user)],
):
    """Synchronous AI error explanation (non-streaming fallback)."""
    return await code_service.explain_error(body)


@router.get("/runtimes")
async def list_runtimes():
    """List the Code Lab runner's languages and availability."""
    return await code_service.list_runtimes()


# ---------------------------------------------------------------------------
# Coding Lab problem endpoints
# ---------------------------------------------------------------------------

@router.get("/problems", response_model=list[CodeProblemResponse])
async def list_problems(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Practice problems first, then each of the user's courses, lesson by lesson (Classwork order)."""
    stmt = (
        _PROBLEM_ROW
        .outerjoin(CourseModule, and_(CourseModule.course_id == CodeProblem.course_id, CourseModule.title == CodeProblem.topic))
        .where(CodeProblem.is_active == True)  # noqa: E712
    )
    ids = await _member_course_ids(db, user)
    if ids is not None:
        stmt = stmt.where(or_(CodeProblem.course_id.is_(None), CodeProblem.course_id.in_(ids)))
    stmt = stmt.order_by(
        CodeProblem.course_id.is_not(None), Course.title, CourseModule.sort_order.nulls_last(),
        CodeProblem.topic.nulls_last(), CodeProblem.sort_order, CodeProblem.id,
    )
    rows = (await db.execute(stmt)).all()
    return [_problem_to_response(p, title, _can_edit(user, p, owner_id)) for p, title, owner_id in rows]


@router.get("/problems/{problem_id}", response_model=CodeProblemResponse)
async def get_problem(
    problem_id: int,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Return a single active coding lab problem the user may see."""
    return _problem_to_response(*await _visible_problem(db, user, problem_id))


@router.post("/problems", response_model=CodeProblemResponse, status_code=201)
async def create_problem(
    body: CodeProblemCreate,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Add a problem to a lesson of one of your courses. Only admins may add course-less practice problems."""
    if user.role not in ("admin", "teacher"):
        raise HTTPException(status_code=403, detail="Admin or teacher role required")
    course_title = None
    if body.courseId is None:
        if user.role != "admin":
            raise HTTPException(status_code=400, detail="Choose one of your courses for this problem.")
    else:
        course = await db.get(Course, body.courseId)
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")
        if user.role != "admin" and course.owner_id != user.id:
            raise HTTPException(status_code=403, detail="You can only add problems to your own courses.")
        course_title = course.title
    problem = CodeProblem(
        course_id=body.courseId,
        topic=_normalize_topic(body.topic),
        title=body.title,
        difficulty=body.difficulty,
        points=body.points,
        description=body.description,
        requirements_json=json.dumps(body.requirements),
        expected_output=body.expectedOutput,
        hints_json=json.dumps(body.hints),
        starter_code_json=json.dumps(body.starterCode),
        topics=body.topics,
        sort_order=body.sortOrder,
        test_cases_json=json.dumps([c.model_dump() for c in body.testCases]) if body.testCases else None,
    )
    db.add(problem)
    await db.commit()
    await db.refresh(problem)
    return _problem_to_response(problem, course_title, True)


@router.patch("/problems/{problem_id}", response_model=CodeProblemResponse)
async def update_problem(
    problem_id: int,
    body: CodeProblemUpdate,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Partially update a problem (course teacher or admin). The course cannot change; the lesson can."""
    problem, course_title = await _editable_problem(db, user, problem_id, active_only=False)
    if body.topic is not None:
        problem.topic = _normalize_topic(body.topic)
    if body.title is not None:
        problem.title = body.title
    if body.difficulty is not None:
        problem.difficulty = body.difficulty
    if body.points is not None:
        problem.points = body.points
    if body.description is not None:
        problem.description = body.description
    if body.requirements is not None:
        problem.requirements_json = json.dumps(body.requirements)
    if body.expectedOutput is not None:
        problem.expected_output = body.expectedOutput
    if body.hints is not None:
        problem.hints_json = json.dumps(body.hints)
    if body.starterCode is not None:
        problem.starter_code_json = json.dumps(body.starterCode)
    if body.topics is not None:
        problem.topics = body.topics
    if body.sortOrder is not None:
        problem.sort_order = body.sortOrder
    if body.isActive is not None:
        problem.is_active = body.isActive
    if body.testCases is not None:
        problem.test_cases_json = json.dumps([c.model_dump() for c in body.testCases])
    await db.commit()
    await db.refresh(problem)
    return _problem_to_response(problem, course_title, True)


@router.delete("/problems/{problem_id}", status_code=204)
async def delete_problem(
    problem_id: int,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Hide a problem from the lab (soft delete: past submissions are kept). Course teacher or admin."""
    problem, _ = await _editable_problem(db, user, problem_id)
    problem.is_active = False
    await db.commit()


# ---------------------------------------------------------------------------
# Submissions: run every test case, save the result, tell the student's teachers
# ---------------------------------------------------------------------------

@router.post("/problems/{problem_id}/submit", response_model=CodeSubmitResponse)
async def submit_solution(
    problem_id: int,
    body: CodeSubmitRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Run the code against each test case, save the submission and notify the student's teachers."""
    problem, _, _ = await _visible_problem(db, user, problem_id)
    cases = _test_cases(problem)
    if not cases:
        raise HTTPException(status_code=400, detail="This problem has no test cases yet.")

    results: list[TestCaseResult] = []
    for i, case in enumerate(cases):
        run = await code_service.execute_code(CodeExecuteRequest(
            language=body.language, code=body.code, stdin=case.get("input", ""), explainOnError=False,
        ))
        r = run.result
        ok = r.exitCode == 0 and _normalize_output(r.stdout) == _normalize_output(case["expectedOutput"])
        hidden = bool(case.get("hidden"))
        results.append(TestCaseResult(
            index=i + 1,
            passed=ok,
            hidden=hidden,
            input=None if hidden else case.get("input", ""),
            expectedOutput=None if hidden else case["expectedOutput"],
            actualOutput=None if hidden else r.stdout,
            stderr=None if hidden else (r.stderr or None),
        ))
    passed = sum(1 for r in results if r.passed)

    sub = CodeSubmission(
        problem_id=problem.id, user_id=user.id, language=body.language.lower(),
        code=body.code, passed=passed, total=len(results),
    )
    db.add(sub)
    await db.flush()
    if user.role == "student":
        # A course problem goes to that course's teacher; a practice problem to all of the student's teachers.
        owners_stmt = (
            select(Course.owner_id).join(Enrollment, Enrollment.course_id == Course.id)
            .where(Enrollment.user_id == user.id, Course.owner_id.is_not(None))
        )
        if problem.course_id is not None:
            owners_stmt = owners_stmt.where(Course.id == problem.course_id)
        owners = (await db.execute(owners_stmt.distinct())).scalars().all()
        for owner_id in owners:
            await NotificationDAO.create(
                db, user_id=owner_id, type="code_submission",
                title="New Code Lab submission",
                body=f"{user.full_name} submitted \"{problem.title}\": {passed}/{len(results)} tests passed",
                link=f"/code-lab?problem={problem.id}",
            )
    await db.commit()
    await db.refresh(sub)
    return CodeSubmitResponse(
        id=sub.id, passed=passed, total=len(results), results=results,
        createdAt=sub.created_at.isoformat() if sub.created_at else "",
    )


@router.get("/problems/{problem_id}/submissions", response_model=list[CodeSubmissionItem])
async def list_submissions(
    problem_id: int,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Students see their own submissions; a teacher sees those of students in their courses."""
    await _visible_problem(db, user, problem_id)
    stmt = (
        select(CodeSubmission, User.full_name)
        .join(User, User.id == CodeSubmission.user_id)
        .where(CodeSubmission.problem_id == problem_id)
    )
    if user.role == "student":
        stmt = stmt.where(CodeSubmission.user_id == user.id)
    elif user.role == "teacher":
        my_students = (
            select(Enrollment.user_id).join(Course, Course.id == Enrollment.course_id)
            .where(Course.owner_id == user.id)
        )
        stmt = stmt.where(CodeSubmission.user_id.in_(my_students))
    rows = (await db.execute(stmt.order_by(CodeSubmission.created_at.desc()).limit(200))).all()
    return [
        CodeSubmissionItem(
            id=s.id, problemId=s.problem_id, userId=s.user_id, studentName=name, language=s.language,
            code=s.code, passed=s.passed, total=s.total,
            createdAt=s.created_at.isoformat() if s.created_at else "",
        )
        for s, name in rows
    ]
