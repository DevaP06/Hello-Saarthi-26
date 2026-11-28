"""Quiz feature service: quizzes, questions, attempts, adaptive generation."""

import json
import logging
import os
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import select, func

from saarthi_backend.dao import QuizAttemptDAO, QuizDAO, QuizQuestionDAO
from saarthi_backend.model import Quiz, QuizAttempt, QuizQuestion
from saarthi_backend.schema.quiz_schemas import (
    AdaptiveQuizRequest,
    AdaptiveQuizResponse,
    GeneratedQuizQuestion,
    PeerComparisonResponse,
    QuizAnswer,
    QuizAnswerResult,
    ScoreBucket,
)
from saarthi_backend.utils.exceptions import NotFoundError, ValidationError

logger = logging.getLogger(__name__)


async def list_quizzes(
    db: AsyncSession,
    course_id: int | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[Quiz], int]:
    quizzes = await QuizDAO.list_all(db, course_id=course_id, limit=limit, offset=offset)
    total = await QuizDAO.count_all(db, course_id=course_id)
    return quizzes, total


async def get_quiz_with_questions(
    db: AsyncSession, quiz_id: int
) -> tuple[Quiz | None, list[QuizQuestion]]:
    quiz = await QuizDAO.get_by_id(db, quiz_id)
    if not quiz:
        return None, []
    questions = await QuizQuestionDAO.list_by_quiz(db, quiz_id)
    return quiz, questions


async def start_attempt(db: AsyncSession, user_id: int, quiz_id: int) -> QuizAttempt | None:
    quiz = await QuizDAO.get_by_id(db, quiz_id)
    if not quiz:
        return None
    return await QuizAttemptDAO.create(db, user_id, quiz_id)


async def submit_attempt(
    db: AsyncSession,
    quiz_id: int,
    attempt_id: int,
    user_id: int,
    answers: list[QuizAnswer],
) -> tuple[QuizAttempt, list[QuizAnswerResult]]:
    """Grade the attempt against the stored correct answers and save it.

    The score is the percentage of all quiz questions answered correctly, so
    unanswered questions count as wrong. An attempt can be submitted only once.
    """
    attempt = await QuizAttemptDAO.get_by_id(db, attempt_id)
    if not attempt or attempt.user_id != user_id or attempt.quiz_id != quiz_id:
        raise NotFoundError("Attempt not found.", details=None)
    if attempt.submitted_at is not None:
        raise ValidationError("This attempt has already been submitted.", details=None)

    questions = await QuizQuestionDAO.list_by_quiz(db, quiz_id)
    selected = {a.questionId: a.selectedIndex for a in answers}
    results = [
        QuizAnswerResult(
            questionId=str(q.id),
            selectedIndex=selected.get(str(q.id)),
            correctIndex=q.correct_index,
            isCorrect=selected.get(str(q.id)) == q.correct_index,
            explanation=q.explanation,
        )
        for q in questions
    ]
    correct = sum(1 for r in results if r.isCorrect)
    score = round(correct / len(questions) * 100, 2) if questions else 0.0

    answers_json = json.dumps([{"questionId": r.questionId, "selectedIndex": r.selectedIndex} for r in results])
    submitted = await QuizAttemptDAO.submit(db, attempt_id, user_id, score=score, answers_json=answers_json)
    return submitted, results


async def list_attempts(
    db: AsyncSession,
    user_id: int,
    quiz_id: int,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[QuizAttempt], int]:
    items = await QuizAttemptDAO.list_by_user_quiz(db, user_id, quiz_id, limit=limit, offset=offset)
    total = await QuizAttemptDAO.count_by_user_quiz(db, user_id, quiz_id)
    return items, total


async def create_question(
    db: AsyncSession,
    quiz_id: int,
    question_text: str,
    options_json: str,
    correct_index: int,
    sort_order: int = 0,
) -> QuizQuestion | None:
    quiz = await QuizDAO.get_by_id(db, quiz_id)
    if not quiz:
        return None
    return await QuizQuestionDAO.create(
        db,
        quiz_id=quiz_id,
        question_text=question_text,
        options_json=options_json,
        correct_index=correct_index,
        sort_order=sort_order,
    )


async def update_question(
    db: AsyncSession,
    quiz_id: int,
    question_id: int,
    question_text: str | None = None,
    options_json: str | None = None,
    correct_index: int | None = None,
    sort_order: int | None = None,
) -> QuizQuestion | None:
    quiz = await QuizDAO.get_by_id(db, quiz_id)
    if not quiz:
        return None
    q = await QuizQuestionDAO.get_by_id(db, question_id)
    if not q or q.quiz_id != quiz_id:
        return None
    return await QuizQuestionDAO.update(
        db,
        question_id,
        question_text=question_text,
        options_json=options_json,
        correct_index=correct_index,
        sort_order=sort_order,
    )


async def delete_question(
    db: AsyncSession, quiz_id: int, question_id: int
) -> bool:
    quiz = await QuizDAO.get_by_id(db, quiz_id)
    if not quiz:
        return False
    q = await QuizQuestionDAO.get_by_id(db, question_id)
    if not q or q.quiz_id != quiz_id:
        return False
    await QuizQuestionDAO.delete(db, question_id)
    return True


# ─── Adaptive Quiz Generation ─────────────────────────────────────────────────

def _topic_matches(value: str | None, requested_topic: str) -> bool:
    """Case-insensitive topic matching with partial-name support."""
    if not value:
        return False
    value = value.strip().casefold()
    requested = requested_topic.strip().casefold()
    return requested in value or value in requested


async def _get_adaptive_performance(
    db: AsyncSession,
    user_id: int,
    requested_topic: str,
    limit: int = 5,
) -> dict:
    """
    Build a performance profile from the authenticated student's own
    submitted attempts. Client-supplied pastAttempts are deliberately ignored.
    """
    result = await db.execute(
        select(QuizAttempt, Quiz)
        .join(Quiz, Quiz.id == QuizAttempt.quiz_id)
        .where(
            QuizAttempt.user_id == user_id,
            QuizAttempt.submitted_at.is_not(None),
            QuizAttempt.score.is_not(None),
        )
        .order_by(QuizAttempt.submitted_at.desc(), QuizAttempt.id.desc())
        .limit(30)
    )

    attempts = []
    topic_stats: dict[str, dict[str, int]] = {}

    for attempt, quiz in result.all():
        questions_result = await db.execute(
            select(QuizQuestion).where(QuizQuestion.quiz_id == quiz.id)
        )
        questions = questions_result.scalars().all()

        quiz_matches = (
            _topic_matches(quiz.topic, requested_topic)
            or _topic_matches(quiz.title, requested_topic)
        )

        try:
            answer_rows = json.loads(attempt.answers_json or "[]")
        except (TypeError, ValueError):
            answer_rows = []

        if not isinstance(answer_rows, list):
            continue

        answers = {
            str(row.get("questionId")): row.get("selectedIndex")
            for row in answer_rows
            if isinstance(row, dict) and row.get("questionId") is not None
        }

        relevant_questions = []
        for question in questions:
            question_topic = question.topic or quiz.topic or quiz.title

            if quiz_matches or _topic_matches(question_topic, requested_topic):
                relevant_questions.append((question, question_topic))

        if not relevant_questions:
            continue

        answered_count = 0
        correct_count = 0

        for question, question_topic in relevant_questions:
            question_id = str(question.id)
            if question_id not in answers:
                continue

            answered_count += 1
            is_correct = answers[question_id] == question.correct_index
            correct_count += int(is_correct)

            key = question_topic.strip() or "General"
            stats = topic_stats.setdefault(key, {"correct": 0, "total": 0})
            stats["total"] += 1
            stats["correct"] += int(is_correct)

        if answered_count:
            attempts.append({
                "title": quiz.title,
                "score": float(attempt.score),
                "submitted_at": attempt.submitted_at.isoformat(),
            })

        if len(attempts) >= limit:
            break

    scores = [item["score"] for item in attempts]
    average_score = round(sum(scores) / len(scores), 1) if scores else None

    weak_areas = [
        name
        for name, stats in topic_stats.items()
        if stats["total"] >= 1
        and stats["correct"] / stats["total"] < 0.70
    ]

    topic_performance = {
        name: {
            "accuracy": round(stats["correct"] / stats["total"] * 100, 1),
            "correct": stats["correct"],
            "total": stats["total"],
        }
        for name, stats in topic_stats.items()
    }

    return {
        "attempts": attempts,
        "average_score": average_score,
        "attempts_analyzed": len(attempts),
        "weak_areas": weak_areas,
        "topic_performance": topic_performance,
    }


def _select_adaptive_difficulty(average_score: float | None) -> str:
    """Choose a conservative target difficulty from recent performance."""
    if average_score is None:
        return "medium"
    if average_score < 40:
        return "easy"
    if average_score < 60:
        return "easy-medium"
    if average_score < 80:
        return "medium"
    if average_score < 90:
        return "medium-hard"
    return "hard"


_QUIZ_GEN_SYSTEM = """You are an expert engineering professor creating adaptive quiz questions.

Return ONLY valid JSON:
{
  "title": "Quiz title",
  "questions": [
    {
      "questionText": "Question text",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correctIndex": 0,
      "explanation": "Why the answer is correct",
      "difficulty": "easy|medium|hard",
      "topic": "Specific sub-topic"
    }
  ]
}

Rules:
- Follow the requested difficulty distribution exactly as closely as possible.
- Focus additional questions on the supplied weak topics.
- Do not invent student performance or weak topics.
- Questions must be technically accurate and unambiguous.
- Each question must have four plausible options and exactly one correct answer.
- Use the supplied course excerpts as the source of truth when provided.
- Do not introduce facts that contradict the supplied course material.
- Avoid repeating the same question in different wording.
"""


async def generate_adaptive_quiz(
    db: AsyncSession,
    user_id: int,
    body: AdaptiveQuizRequest,
) -> AdaptiveQuizResponse:
    """Generate a quiz using the authenticated student's actual quiz history."""
    try:
        from src.safety.guard import scrub_pii
        from src.utils.llm import async_openai, chat_model_name

        topic = scrub_pii(body.topic)[0]

        profile = await _get_adaptive_performance(
            db, user_id, topic, limit=5
        )

        requested_difficulty = body.difficulty or "auto"
        difficulty = (
            _select_adaptive_difficulty(profile["average_score"])
            if requested_difficulty == "auto"
            else requested_difficulty
        )

        distributions = {
            "easy": "70% easy, 30% medium",
            "easy-medium": "50% easy, 50% medium",
            "medium": "20% easy, 60% medium, 20% hard",
            "medium-hard": "20% medium, 80% hard",
            "hard": "100% hard",
        }
        distribution = distributions.get(difficulty, distributions["medium"])

        attempts_text = "\n".join(
            f"- {item['title']}: {item['score']:.1f}%"
            for item in profile["attempts"]
        ) or "No previous submitted attempts found for this topic."

        weak_topics_text = (
            ", ".join(profile["weak_areas"])
            if profile["weak_areas"]
            else "No weak topics identified yet."
        )

        topic_performance_text = json.dumps(
            profile["topic_performance"], ensure_ascii=False
        )

        from saarthi_backend.service.chat_service import _kb_excerpts
        excerpts = _kb_excerpts(topic)[:4000]
        grounding = (
            "\n\nCourse excerpts. Use these as the source of truth:\n"
            + excerpts
            if excerpts else ""
        )

        user_prompt = f"""
Generate exactly {body.numQuestions} adaptive quiz questions.

Subject: {topic}
Course: {body.courseTitle or "Not specified"}

Target difficulty: {difficulty}
Required difficulty distribution: {distribution}

Recent verified student attempts:
{attempts_text}

Average recent score: {profile["average_score"]}
Number of attempts analyzed: {profile["attempts_analyzed"]}

Weak topics detected from individual answers:
{weak_topics_text}

Topic-level accuracy:
{topic_performance_text}

Give additional attention to weak topics while still covering the requested
subject. Do not make every question about a single weak topic.

Focus on conceptual understanding, application, and problem-solving.
{grounding}
"""

        client = async_openai()
        response = await client.chat.completions.create(
            model=chat_model_name(),
            messages=[
                {"role": "system", "content": _QUIZ_GEN_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)

        generated_questions = []
        for question in data.get("questions", []):
            if not isinstance(question, dict):
                continue

            question_text = question.get("questionText")
            options = question.get("options")
            correct_index = question.get("correctIndex")

            if not isinstance(question_text, str) or not question_text.strip():
                continue
            if not isinstance(options, list) or len(options) != 4:
                continue
            if not all(isinstance(option, str) and option.strip() for option in options):
                continue

            try:
                correct_index = int(correct_index)
            except (TypeError, ValueError):
                continue

            if not 0 <= correct_index < len(options):
                continue

            question_difficulty = question.get("difficulty", "medium")
            if question_difficulty not in ("easy", "medium", "hard"):
                question_difficulty = "medium"

            generated_questions.append(
                GeneratedQuizQuestion(
                    questionText=question_text.strip(),
                    options=options,
                    correctIndex=correct_index,
                    explanation=str(question.get("explanation", "")),
                    difficulty=question_difficulty,
                    topic=str(question.get("topic") or topic),
                )
            )

        breakdown = {
            level: sum(1 for q in generated_questions if q.difficulty == level)
            for level in ("easy", "medium", "hard")
        }

        return AdaptiveQuizResponse(
            title=data.get("title", f"Adaptive Quiz: {topic}"),
            topic=topic,
            difficulty=difficulty,
            questions=generated_questions,
            weakAreasDetected=profile["weak_areas"],
            generatedAt=datetime.now(timezone.utc).isoformat(),
            averageScore=profile["average_score"],
            attemptsAnalyzed=profile["attempts_analyzed"],
            difficultyBreakdown=breakdown,
        )

    except Exception as e:
        logger.exception("Adaptive quiz generation failed: %s", e)
        return AdaptiveQuizResponse(
            title=f"Quiz: {body.topic}",
            topic=body.topic,
            difficulty=body.difficulty or "medium",
            questions=[],
            weakAreasDetected=[],
            generatedAt=datetime.now(timezone.utc).isoformat(),
            averageScore=None,
            attemptsAnalyzed=0,
            difficultyBreakdown={"easy": 0, "medium": 0, "hard": 0},
        )
# ─── Peer Comparison ──────────────────────────────────────────────────────────

async def get_peer_comparison(db: AsyncSession, user_id: int) -> PeerComparisonResponse:
    """
    Compute the current user's quiz score percentile against all peers.
    Fetches all submitted attempts, buckets scores into 10-point ranges,
    and returns the distribution + user's position.
    """
    # All submitted attempts (score not null)
    all_result = await db.execute(
        select(QuizAttempt.user_id, QuizAttempt.score, QuizAttempt.quiz_id)
        .where(QuizAttempt.score.isnot(None))
    )
    all_attempts = all_result.fetchall()

    if not all_attempts:
        return PeerComparisonResponse(
            userAvgScore=0.0,
            percentile=0.0,
            peerCount=0,
            distribution=[],
            topicBreakdown=[],
            generatedAt=datetime.now(timezone.utc).isoformat(),
        )

    # Per-user average scores
    user_scores: dict[int, list[float]] = {}
    for row in all_attempts:
        user_scores.setdefault(row.user_id, []).append(row.score)

    user_avgs = {uid: sum(scores) / len(scores) for uid, scores in user_scores.items()}

    user_avg = user_avgs.get(user_id, 0.0)
    peer_count = len(user_avgs)

    # Percentile: what fraction of peers does this user beat
    beaten = sum(1 for uid, avg in user_avgs.items() if uid != user_id and avg < user_avg)
    peers_excluding_self = max(peer_count - 1, 1)
    percentile = round((beaten / peers_excluding_self) * 100, 1)

    # Score distribution bucketed into 10-point ranges
    buckets: dict[str, int] = {}
    for i in range(0, 100, 10):
        buckets[f"{i}-{i+10}"] = 0

    for avg in user_avgs.values():
        bucket_idx = min(int(avg // 10) * 10, 90)
        key = f"{bucket_idx}-{bucket_idx + 10}"
        buckets[key] = buckets.get(key, 0) + 1

    user_bucket_idx = min(int(user_avg // 10) * 10, 90)
    user_bucket_key = f"{user_bucket_idx}-{user_bucket_idx + 10}"

    distribution = [
        ScoreBucket(range=k, count=v, isUser=(k == user_bucket_key))
        for k, v in sorted(buckets.items(), key=lambda x: int(x[0].split("-")[0]))
    ]

    # Topic breakdown: per-quiz avg for user vs all peers
    quiz_user: dict[int, list[float]] = {}
    quiz_all: dict[int, list[float]] = {}
    for row in all_attempts:
        quiz_all.setdefault(row.quiz_id, []).append(row.score)
        if row.user_id == user_id:
            quiz_user.setdefault(row.quiz_id, []).append(row.score)

    # Fetch quiz titles for the quizzes the user has taken
    topic_breakdown = []
    if quiz_user:
        quiz_ids = list(quiz_user.keys())
        quiz_result = await db.execute(select(Quiz).where(Quiz.id.in_(quiz_ids)))
        quizzes = {q.id: q for q in quiz_result.scalars().all()}
        for qid, user_q_scores in quiz_user.items():
            quiz = quizzes.get(qid)
            if not quiz:
                continue
            topic_breakdown.append({
                "topic": quiz.title,
                "userAvg": round(sum(user_q_scores) / len(user_q_scores), 1),
                "peerAvg": round(sum(quiz_all[qid]) / len(quiz_all[qid]), 1),
            })

    return PeerComparisonResponse(
        userAvgScore=round(user_avg, 1),
        percentile=percentile,
        peerCount=peer_count,
        distribution=distribution,
        topicBreakdown=topic_breakdown[:6],
        generatedAt=datetime.now(timezone.utc).isoformat(),
    )
