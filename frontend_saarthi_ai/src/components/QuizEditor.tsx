import { useState } from 'react';
import { ListChecks, Loader2, Plus, Sparkles, Trash2, X } from 'lucide-react';
import { api, type QuizDetailResponse, type QuizDifficulty } from '../lib/api';
import { useToast } from './Toast';

interface DraftQuestion {
    questionText: string;
    options: string[];
    correctIndex: number;
    topic: string;
    difficulty: QuizDifficulty;
    explanation: string;
}

interface Props {
    courseId: string;
    topics: string[];
    onClose: () => void;
    onCreated: (quiz: QuizDetailResponse) => void;
}

const emptyQuestion = (topic: string): DraftQuestion => ({
    questionText: '', options: ['', '', '', ''], correctIndex: 0, topic, difficulty: 'recall', explanation: '',
});

// POST /quizzes/generate-grounded: questions written only from the course notes, each citing its excerpt.
const LEVEL_CODE: Record<QuizDifficulty, string> = { recall: 'L1', application: 'L2', analysis: 'L3' };
interface GeneratedMcqs {
    questions: { question: string; options: string[]; answer_index: number; explanation: string; source: string }[];
    not_covered?: boolean;
    error?: string;
}

const isBlank = (q: DraftQuestion) => !q.questionText.trim() && q.options.every((o) => !o.trim());

/** Teacher form: a quiz with multiple-choice questions, saved in one request. */
export default function QuizEditor({ courseId, topics, onClose, onCreated }: Props) {
    const toast = useToast();
    const [title, setTitle] = useState('');
    const [description, setDescription] = useState('');
    const [topic, setTopic] = useState(topics[0] ?? '');
    const [duration, setDuration] = useState(15);
    const [passing, setPassing] = useState(60);
    const [questions, setQuestions] = useState<DraftQuestion[]>([emptyQuestion('')]);
    const [error, setError] = useState<string | null>(null);
    const [saving, setSaving] = useState(false);
    const [genTopic, setGenTopic] = useState('');
    const [genLevel, setGenLevel] = useState<QuizDifficulty>('recall');
    const [genCount, setGenCount] = useState(5);
    const [generating, setGenerating] = useState(false);
    const [genNote, setGenNote] = useState<string | null>(null);

    const update = (i: number, change: Partial<DraftQuestion>) =>
        setQuestions((prev) => prev.map((q, j) => (j === i ? { ...q, ...change } : q)));

    // Drafts from the notes: the teacher checks and edits them before posting.
    const generate = async () => {
        const t = genTopic.trim();
        if (t.length < 2) { setError('Type a topic to generate questions about, e.g. "FFT".'); return; }
        setGenerating(true);
        setError(null);
        setGenNote(null);
        try {
            const res = await api.post<GeneratedMcqs>('/quizzes/generate-grounded', {
                courseId: Number(courseId), topic: t, numQuestions: genCount, level: LEVEL_CODE[genLevel],
            });
            if (!res.questions?.length) {
                setGenNote(res.not_covered
                    ? `The notes don't cover "${t}". Try a shorter topic name, e.g. "FFT" or "SVM".`
                    : 'No usable questions came back. Try again.');
                return;
            }
            const drafts: DraftQuestion[] = res.questions.map((q) => ({
                questionText: q.question,
                options: q.options.slice(0, 4),
                correctIndex: q.answer_index,
                topic: t,
                difficulty: genLevel,
                explanation: `${q.explanation} Source: ${q.source}`,
            }));
            setQuestions((prev) => [...prev.filter((q) => !isBlank(q)), ...drafts]);
            if (!title.trim()) setTitle(`${t}: practice`);
            setGenNote(`Added ${drafts.length} question${drafts.length === 1 ? '' : 's'} from the notes. Check each one before posting.`);
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Could not generate questions');
        } finally {
            setGenerating(false);
        }
    };

    const save = async () => {
        setError(null);
        if (!title.trim()) { setError('Give the quiz a title.'); return; }
        for (const [i, q] of questions.entries()) {
            const filled = q.options.filter((o) => o.trim());
            if (!q.questionText.trim()) { setError(`Question ${i + 1}: write the question.`); return; }
            if (filled.length < 2) { setError(`Question ${i + 1}: give at least two options.`); return; }
            if (!q.options[q.correctIndex]?.trim()) { setError(`Question ${i + 1}: the correct answer is empty.`); return; }
        }
        setSaving(true);
        try {
            const created = await api.post<QuizDetailResponse>('/quizzes', {
                courseId: Number(courseId),
                title: title.trim(),
                description: description.trim() || undefined,
                durationMinutes: duration,
                passingScore: passing,
                topic: topic || undefined,
                questions: questions.map((q, i) => {
                    // Drop empty options; keep the correct one's position among the rest.
                    const kept = q.options.map((o, idx) => ({ o: o.trim(), idx })).filter((x) => x.o);
                    return {
                        questionText: q.questionText.trim(),
                        optionsJson: JSON.stringify(kept.map((x) => x.o)),
                        correctIndex: kept.findIndex((x) => x.idx === q.correctIndex),
                        sortOrder: i,
                        topic: q.topic.trim() || undefined,
                        difficulty: q.difficulty,
                        explanation: q.explanation.trim() || undefined,
                    };
                }),
            });
            toast.showToast(`Quiz "${created.title}" posted`);
            onCreated(created);
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Could not save the quiz');
        } finally {
            setSaving(false);
        }
    };

    return (
        <div className="cd-modal-overlay" onClick={onClose}>
            <div className="cd-modal cd-modal-wide cd-quiz-editor animate-scale-in" role="dialog" aria-modal="true" aria-labelledby="cd-quiz-editor-title" onClick={(e) => e.stopPropagation()}>
                <div className="cd-modal-header">
                    <h2 id="cd-quiz-editor-title"><ListChecks size={18} /> New quiz</h2>
                    <button className="cd-modal-close" onClick={onClose} aria-label="Close"><X size={20} /></button>
                </div>
                <div className="cd-modal-body">
                    <div className="cd-modal-field">
                        <label>Title *</label>
                        <input className="input" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="e.g. Sampling and aliasing" />
                    </div>
                    <div className="cd-modal-field">
                        <label>Instructions</label>
                        <textarea className="input" rows={2} value={description} onChange={(e) => setDescription(e.target.value)} />
                    </div>
                    <div className="cd-modal-row">
                        <div className="cd-modal-field">
                            <label>Topic</label>
                            <select className="input" value={topic} onChange={(e) => setTopic(e.target.value)}>
                                <option value="">No topic</option>
                                {topics.map((t) => <option key={t} value={t}>{t}</option>)}
                            </select>
                        </div>
                        <div className="cd-modal-field">
                            <label>Time (minutes)</label>
                            <input className="input" type="number" min={1} max={300} value={duration} onChange={(e) => setDuration(Math.max(1, parseInt(e.target.value) || 1))} />
                        </div>
                        <div className="cd-modal-field">
                            <label>Pass mark (%)</label>
                            <input className="input" type="number" min={0} max={100} value={passing} onChange={(e) => setPassing(Math.min(100, Math.max(0, parseInt(e.target.value) || 0)))} />
                        </div>
                    </div>

                    <div className="cd-quiz-generate">
                        <h3><Sparkles size={15} /> Generate questions from the notes</h3>
                        <div className="cd-quiz-generate-row">
                            <input className="input" aria-label="Topic to generate questions about" placeholder="Topic, e.g. FFT" value={genTopic} onChange={(e) => setGenTopic(e.target.value)} maxLength={200} />
                            <select className="input" aria-label="Question level" value={genLevel} onChange={(e) => setGenLevel(e.target.value as QuizDifficulty)}>
                                <option value="recall">Recall</option>
                                <option value="application">Application</option>
                                <option value="analysis">Analysis</option>
                            </select>
                            <select className="input" aria-label="Number of questions" value={genCount} onChange={(e) => setGenCount(Number(e.target.value))}>
                                {[3, 5, 8, 10].map((n) => <option key={n} value={n}>{n} questions</option>)}
                            </select>
                            <button type="button" className="btn btn-outline btn-sm" disabled={generating} onClick={generate}>
                                {generating ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />} {generating ? 'Writing…' : 'Generate'}
                            </button>
                        </div>
                        <p className="cd-quiz-hint">{genNote ?? 'Written only from the course notes; each explanation names its source page.'}</p>
                    </div>

                    {questions.map((q, i) => (
                        <fieldset key={i} className="cd-quiz-question">
                            <legend>Question {i + 1}</legend>
                            {questions.length > 1 && (
                                <button type="button" className="btn-icon cd-quiz-question-remove" aria-label={`Remove question ${i + 1}`} onClick={() => setQuestions((prev) => prev.filter((_, j) => j !== i))}>
                                    <Trash2 size={15} />
                                </button>
                            )}
                            <textarea className="input" rows={2} placeholder="Question" value={q.questionText} onChange={(e) => update(i, { questionText: e.target.value })} />
                            <div className="cd-quiz-options">
                                {q.options.map((opt, k) => (
                                    <label key={k} className="cd-quiz-option">
                                        <input type="radio" name={`correct-${i}`} checked={q.correctIndex === k} onChange={() => update(i, { correctIndex: k })} aria-label={`Option ${String.fromCharCode(65 + k)} is correct`} />
                                        <input className="input" placeholder={`Option ${String.fromCharCode(65 + k)}`} value={opt} onChange={(e) => update(i, { options: q.options.map((o, j) => (j === k ? e.target.value : o)) })} />
                                    </label>
                                ))}
                            </div>
                            <p className="cd-quiz-hint">Select the circle next to the correct answer. Empty options are left out.</p>
                            <div className="cd-modal-row">
                                <div className="cd-modal-field">
                                    <label>Concept</label>
                                    <input className="input" placeholder="e.g. Nyquist rate" value={q.topic} onChange={(e) => update(i, { topic: e.target.value })} />
                                </div>
                                <div className="cd-modal-field">
                                    <label>Level</label>
                                    <select className="input" value={q.difficulty} onChange={(e) => update(i, { difficulty: e.target.value as QuizDifficulty })}>
                                        <option value="recall">Recall</option>
                                        <option value="application">Application</option>
                                        <option value="analysis">Analysis</option>
                                    </select>
                                </div>
                            </div>
                            <div className="cd-modal-field">
                                <label>Explanation (shown after submitting)</label>
                                <textarea className="input" rows={2} value={q.explanation} onChange={(e) => update(i, { explanation: e.target.value })} />
                            </div>
                        </fieldset>
                    ))}
                    <button type="button" className="btn btn-outline btn-sm" onClick={() => setQuestions((prev) => [...prev, emptyQuestion(prev[prev.length - 1]?.topic ?? '')])}>
                        <Plus size={14} /> Add question
                    </button>
                    {error && <p className="cd-assignment-error">{error}</p>}
                </div>
                <div className="cd-modal-footer">
                    <button className="btn btn-outline" onClick={onClose}>Cancel</button>
                    <button className="btn btn-primary" disabled={saving} onClick={save}>{saving ? 'Posting…' : `Post quiz (${questions.length})`}</button>
                </div>
            </div>
        </div>
    );
}
