import { useEffect, useState } from 'react';
import { Code2, Plus, Trash2, X } from 'lucide-react';
import { api, type CodeProblemResponse, type CourseResponse, type ModuleResponse, type PaginatedResponse } from '../lib/api';
import { useToast } from './Toast';
import '../pages/CourseDetail.css'; // shared cd-modal styles

interface DraftCase {
    input: string;
    expectedOutput: string;
    hidden: boolean;
}

interface Props {
    /** null = a new problem. */
    problem: CodeProblemResponse | null;
    languages: { id: string; label: string }[];
    isAdmin: boolean;
    onClose: () => void;
    onSaved: (problem: CodeProblemResponse) => void;
}

const lines = (text: string) => text.split('\n').map((l) => l.trim()).filter(Boolean);
const emptyCase = (): DraftCase => ({ input: '', expectedOutput: '', hidden: false });

/** Teacher form: add or edit a Code Lab problem in a lesson of one of their courses. */
export default function CodeProblemEditor({ problem, languages, isAdmin, onClose, onSaved }: Props) {
    const toast = useToast();
    const [courses, setCourses] = useState<CourseResponse[] | null>(null);
    const [courseId, setCourseId] = useState(problem?.courseId != null ? String(problem.courseId) : '');
    const [lessons, setLessons] = useState<string[]>([]);
    const [topic, setTopic] = useState(problem?.topic ?? '');
    const [title, setTitle] = useState(problem?.title ?? '');
    const [difficulty, setDifficulty] = useState(problem?.difficulty ?? 'medium');
    const [points, setPoints] = useState(problem?.points ?? 50);
    const [description, setDescription] = useState(problem?.description ?? '');
    const [requirements, setRequirements] = useState((problem?.requirements ?? []).join('\n'));
    const [hints, setHints] = useState((problem?.hints ?? []).join('\n'));
    const [starter, setStarter] = useState<Record<string, string>>(problem?.starterCode ?? {});
    const [starterLang, setStarterLang] = useState(Object.keys(problem?.starterCode ?? {})[0] ?? 'python');
    const [cases, setCases] = useState<DraftCase[]>(
        problem?.testCases.length
            ? problem.testCases.map((c) => ({ input: c.input ?? '', expectedOutput: c.expectedOutput ?? '', hidden: c.hidden }))
            : [emptyCase()],
    );
    const [error, setError] = useState<string | null>(null);
    const [saving, setSaving] = useState(false);

    // A new problem goes into one of the teacher's courses (admins: any course, or practice).
    useEffect(() => {
        if (problem) return;
        api.get<PaginatedResponse<CourseResponse>>('/courses', { limit: 100, offset: 0 })
            .then((res) => {
                setCourses(res.items);
                if (!isAdmin) setCourseId((cur) => cur || res.items[0]?.id || '');
            })
            .catch(() => setCourses([]));
    }, [problem, isAdmin]);

    // The course's lessons, offered as suggestions; a new name creates a new lesson.
    useEffect(() => {
        if (!courseId) { setLessons([]); return; }
        api.get<ModuleResponse[]>(`/courses/${courseId}/modules`)
            .then((mods) => setLessons(mods.map((m) => m.title)))
            .catch(() => setLessons([]));
    }, [courseId]);

    const updateCase = (i: number, change: Partial<DraftCase>) =>
        setCases((prev) => prev.map((c, j) => (j === i ? { ...c, ...change } : c)));

    const save = async () => {
        setError(null);
        if (!problem && !courseId && !isAdmin) { setError('Choose one of your courses.'); return; }
        if (!title.trim()) { setError('Give the problem a title.'); return; }
        if (!description.trim()) { setError('Describe the task.'); return; }
        const testCases = cases.filter((c) => c.input.trim() || c.expectedOutput.trim());
        if (testCases.length === 0) { setError('Add at least one test case.'); return; }
        const missing = testCases.findIndex((c) => !c.expectedOutput.trim());
        if (missing >= 0) { setError(`Test ${missing + 1}: write the expected output.`); return; }

        const body = {
            title: title.trim(),
            difficulty,
            points,
            description: description.trim(),
            requirements: lines(requirements),
            hints: lines(hints),
            starterCode: Object.fromEntries(Object.entries(starter).filter(([, code]) => code.trim())),
            testCases,
            topic: topic.trim(), // "" = no lesson
        };
        setSaving(true);
        try {
            const saved = problem
                ? await api.patch<CodeProblemResponse>(`/code/problems/${problem.id}`, body)
                : await api.post<CodeProblemResponse>('/code/problems', { ...body, courseId: courseId ? Number(courseId) : null });
            toast.showToast(problem ? 'Problem updated' : `Problem "${saved.title}" added`);
            onSaved(saved);
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Could not save the problem');
        } finally {
            setSaving(false);
        }
    };

    const noCourses = !problem && !isAdmin && courses !== null && courses.length === 0;

    return (
        <div className="cd-modal-overlay" onClick={onClose}>
            <div className="cd-modal cd-modal-wide animate-scale-in" role="dialog" aria-modal="true" aria-labelledby="cpe-title" onClick={(e) => e.stopPropagation()}>
                <div className="cd-modal-header">
                    <h2 id="cpe-title"><Code2 size={18} /> {problem ? 'Edit problem' : 'New problem'}</h2>
                    <button className="cd-modal-close" onClick={onClose} aria-label="Close"><X size={20} /></button>
                </div>
                <div className="cd-modal-body">
                    {noCourses && <p className="cd-assignment-error">You have no courses yet. Create a course first, then add problems to its lessons.</p>}
                    <div className="cpe-row">
                        <div className="cd-modal-field">
                            <label htmlFor="cpe-course">Course</label>
                            {problem ? (
                                <input id="cpe-course" className="input" value={problem.courseTitle ?? 'Practice (all students)'} disabled />
                            ) : (
                                <select id="cpe-course" className="input" value={courseId} onChange={(e) => { setCourseId(e.target.value); setTopic(''); }}>
                                    {isAdmin && <option value="">Practice (all students)</option>}
                                    {(courses ?? []).map((c) => <option key={c.id} value={c.id}>{c.title}</option>)}
                                </select>
                            )}
                        </div>
                        <div className="cd-modal-field">
                            <label htmlFor="cpe-lesson">Lesson</label>
                            <input
                                id="cpe-lesson"
                                className="input"
                                list="cpe-lessons"
                                value={topic}
                                onChange={(e) => setTopic(e.target.value)}
                                placeholder={courseId ? 'Pick a lesson or type a new one' : 'Practice problems have no lesson'}
                                disabled={!courseId}
                                maxLength={128}
                            />
                            <datalist id="cpe-lessons">{lessons.map((l) => <option key={l} value={l} />)}</datalist>
                        </div>
                    </div>

                    <div className="cd-modal-field">
                        <label htmlFor="cpe-name">Title *</label>
                        <input id="cpe-name" className="input" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="e.g. 8-point FFT" maxLength={200} />
                    </div>
                    <div className="cpe-row">
                        <div className="cd-modal-field">
                            <label htmlFor="cpe-difficulty">Difficulty</label>
                            <select id="cpe-difficulty" className="input" value={difficulty} onChange={(e) => setDifficulty(e.target.value)}>
                                <option value="easy">Easy</option>
                                <option value="medium">Medium</option>
                                <option value="hard">Hard</option>
                            </select>
                        </div>
                        <div className="cd-modal-field">
                            <label htmlFor="cpe-points">Points</label>
                            <input id="cpe-points" className="input" type="number" min={0} max={1000} value={points} onChange={(e) => setPoints(Math.min(1000, Math.max(0, parseInt(e.target.value) || 0)))} />
                        </div>
                    </div>
                    <div className="cd-modal-field">
                        <label htmlFor="cpe-desc">Task *</label>
                        <textarea id="cpe-desc" className="input" rows={4} value={description} onChange={(e) => setDescription(e.target.value)} placeholder="What should the program read and print?" />
                    </div>
                    <div className="cpe-row">
                        <div className="cd-modal-field">
                            <label htmlFor="cpe-req">Requirements (one per line)</label>
                            <textarea id="cpe-req" className="input" rows={3} value={requirements} onChange={(e) => setRequirements(e.target.value)} />
                        </div>
                        <div className="cd-modal-field">
                            <label htmlFor="cpe-hints">Hints (one per line)</label>
                            <textarea id="cpe-hints" className="input" rows={3} value={hints} onChange={(e) => setHints(e.target.value)} />
                        </div>
                    </div>
                    <div className="cd-modal-field">
                        <label htmlFor="cpe-starter">Starter code (optional)</label>
                        <select className="input cpe-starter-lang" aria-label="Starter code language" value={starterLang} onChange={(e) => setStarterLang(e.target.value)}>
                            {languages.map((l) => <option key={l.id} value={l.id}>{l.label}{starter[l.id]?.trim() ? ' ✓' : ''}</option>)}
                        </select>
                        <textarea
                            id="cpe-starter"
                            className="input cpe-code"
                            rows={6}
                            spellCheck={false}
                            value={starter[starterLang] ?? ''}
                            onChange={(e) => setStarter((s) => ({ ...s, [starterLang]: e.target.value }))}
                            placeholder="Students start from this code when they pick this language."
                        />
                    </div>

                    <div>
                        <h3 className="cpe-heading">Test cases</h3>
                        <p className="cd-quiz-hint">
                            The program gets the input on stdin, and its output must match the expected output (extra spaces and
                            trailing blank lines are ignored). Students can't see hidden cases.
                        </p>
                    </div>
                    {cases.map((c, i) => (
                        <fieldset key={i} className="cd-quiz-question">
                            <legend>Test {i + 1}</legend>
                            {cases.length > 1 && (
                                <button type="button" className="btn-icon cd-quiz-question-remove" aria-label={`Remove test ${i + 1}`} onClick={() => setCases((prev) => prev.filter((_, j) => j !== i))}>
                                    <Trash2 size={15} />
                                </button>
                            )}
                            <div className="cpe-row">
                                <div className="cd-modal-field">
                                    <label htmlFor={`cpe-in-${i}`}>Input (stdin)</label>
                                    <textarea id={`cpe-in-${i}`} className="input cpe-code" rows={3} spellCheck={false} value={c.input} onChange={(e) => updateCase(i, { input: e.target.value })} />
                                </div>
                                <div className="cd-modal-field">
                                    <label htmlFor={`cpe-out-${i}`}>Expected output *</label>
                                    <textarea id={`cpe-out-${i}`} className="input cpe-code" rows={3} spellCheck={false} value={c.expectedOutput} onChange={(e) => updateCase(i, { expectedOutput: e.target.value })} />
                                </div>
                            </div>
                            <label className="cpe-hidden-toggle">
                                <input type="checkbox" checked={c.hidden} onChange={(e) => updateCase(i, { hidden: e.target.checked })} /> Hidden from students
                            </label>
                        </fieldset>
                    ))}
                    <button type="button" className="btn btn-outline btn-sm cpe-add-case" onClick={() => setCases((prev) => [...prev, emptyCase()])}>
                        <Plus size={14} /> Add test case
                    </button>
                    {error && <p className="cd-assignment-error">{error}</p>}
                </div>
                <div className="cd-modal-footer">
                    <button className="btn btn-outline" onClick={onClose}>Cancel</button>
                    <button className="btn btn-primary" disabled={saving || noCourses} onClick={save}>
                        {saving ? 'Saving…' : problem ? 'Save changes' : 'Add problem'}
                    </button>
                </div>
            </div>
        </div>
    );
}
