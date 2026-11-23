import { useEffect, useState } from 'react';
import { AlertCircle, CheckCircle2, ChevronLeft, RotateCcw, XCircle } from 'lucide-react';
import { api } from '../lib/api';
import { renderAnswer } from '../lib/renderAnswer';

interface ReviewQuestion {
    id: string;
    attemptId: number;
    revision: number;
    quizTitle: string;
    courseId: number | null;
    topic: string;
    difficulty: string;
    questionText: string;
    options: string[];
    weakTopic: boolean;
}

interface ReviewQueue {
    items: ReviewQuestion[];
    dueCount: number;
    nextReviewAt: string | null;
}

interface ReviewFeedback {
    selectedIndex: number;
    correctIndex: number;
    isCorrect: boolean;
    explanation: string | null;
    source: string;
    nextReviewAt: string;
}

const reviewTime = (value: string) => new Date(value).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });

export default function QuizReview({ courseId, topic, onExit }: {
    courseId?: string;
    topic?: string;
    onExit: () => void;
}) {
    const [queue, setQueue] = useState<ReviewQueue | null>(null);
    const [reload, setReload] = useState(0);
    const [selected, setSelected] = useState<number | null>(null);
    const [feedback, setFeedback] = useState<ReviewFeedback | null>(null);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [reviewed, setReviewed] = useState(0);
    const question = queue?.items[0];

    useEffect(() => {
        let cancelled = false;
        void api.get<ReviewQueue>('/quizzes/review/queue', { courseId, topic })
            .then((result) => { if (!cancelled) setQueue(result); })
            .catch((failure) => { if (!cancelled) setError(failure instanceof Error ? failure.message : 'Could not load reviews.'); });
        return () => { cancelled = true; };
    }, [courseId, reload, topic]);

    const refresh = () => {
        setError(null);
        setQueue(null);
        setSelected(null);
        setFeedback(null);
        setReload((previous) => previous + 1);
    };

    const checkAnswer = async () => {
        if (!question || selected === null || busy || feedback) return;
        setBusy(true);
        setError(null);
        try {
            const result = await api.post<ReviewFeedback>(`/quizzes/review/${question.id}/answer`, {
                attemptId: question.attemptId, revision: question.revision, selectedIndex: selected,
            });
            setFeedback(result);
            setReviewed((count) => count + 1);
        } catch (failure) {
            setError(failure instanceof Error ? failure.message : 'Could not check your answer.');
        } finally {
            setBusy(false);
        }
    };

    return (
        <section className="quiz-list-page quiz-review">
            <div className="quiz-list-header">
                <button type="button" className="btn btn-ghost btn-sm" onClick={onExit}><ChevronLeft size={16} /> Exam Practice</button>
                <h1><RotateCcw size={22} /> Spaced Review{topic ? ` · ${topic}` : ''}</h1>
                <p>Review questions from completed quizzes. Weak topics come first.</p>
            </div>
            {error && <div className="quiz-review-error" role="alert">{error} <button type="button" className="btn btn-outline btn-sm" onClick={refresh}>Reload reviews</button></div>}
            {!queue && !error && <p role="status">Loading your review queue…</p>}
            {queue && !question && (
                <div className="quiz-list-empty">
                    <CheckCircle2 size={28} />
                    <p>{reviewed > 0 ? `You reviewed ${reviewed} ${reviewed === 1 ? 'question' : 'questions'}.` : 'No questions are due.'}</p>
                    <p>{queue.nextReviewAt ? `Next review ${reviewTime(queue.nextReviewAt)}` : 'Complete a quiz to add questions to your review queue.'}</p>
                    <button type="button" className="btn btn-outline btn-sm" onClick={refresh}>Refresh queue</button>
                </div>
            )}
            {queue && question && (
                <div className="quiz-question-card quiz-review-card">
                    <div className="quiz-question-header">
                        <span className="badge badge-info">{question.topic}</span>
                        <span className="badge badge-info">{question.difficulty}</span>
                        {question.weakTopic && <span className="badge badge-warning">Weak topic</span>}
                    </div>
                    <p className="quiz-list-meta">{question.quizTitle} · {Math.max(queue.dueCount - Number(Boolean(feedback)), 0)} due</p>
                    <div className="quiz-question-text" id="review-question">{renderAnswer(question.questionText)}</div>
                    <div className="quiz-options" role="group" aria-labelledby="review-question">
                        {question.options.map((option, index) => {
                            const correct = feedback?.correctIndex === index;
                            const wrong = Boolean(feedback && feedback.selectedIndex === index && !correct);
                            return <button key={index} type="button" aria-pressed={index === selected}
                                className={`quiz-option ${index === selected ? 'selected' : ''} ${correct ? 'correct' : ''} ${wrong ? 'wrong' : ''}`}
                                disabled={busy || Boolean(feedback)} onClick={() => setSelected(index)}>
                                <span className="quiz-option-letter">{String.fromCharCode(65 + index)}</span>
                                <span className="quiz-option-text">{option}</span>
                                {correct && <CheckCircle2 size={18} />}{wrong && <XCircle size={18} />}
                            </button>;
                        })}
                    </div>
                    {feedback && <div className="quiz-explanation" role="status">
                        <AlertCircle size={18} />
                        <div>
                            <strong>{feedback.isCorrect ? 'Correct.' : `Correct answer is ${question.options[feedback.correctIndex]}.`}</strong>
                            {feedback.explanation && <div>{renderAnswer(feedback.explanation)}</div>}
                            <p>Source · {feedback.source}</p>
                            <p>Next review · {reviewTime(feedback.nextReviewAt)}</p>
                        </div>
                    </div>}
                    <div className="quiz-review-action">
                        {feedback ? <button type="button" className="btn btn-primary" onClick={() => {
                            if (queue.items.length === 1) { refresh(); return; }
                            setQueue({ ...queue, items: queue.items.slice(1), dueCount: queue.dueCount - 1 });
                            setSelected(null);
                            setFeedback(null);
                        }}>Continue review</button> : <button type="button" className="btn btn-primary"
                            disabled={busy || selected === null} onClick={() => void checkAnswer()}>{busy ? 'Checking…' : 'Check answer'}</button>}
                    </div>
                </div>
            )}
        </section>
    );
}
