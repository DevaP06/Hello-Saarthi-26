import { useEffect, useState } from 'react';
import { MessageSquare, Save } from 'lucide-react';
import { api, type PaginatedResponse } from '../lib/api';
import Pagination from './Pagination';
import { renderAnswer } from '../lib/renderAnswer';
import { useToast } from './Toast';
import './AIReviewPanel.css';

interface QueryLogItem {
    id: string;
    conversationId: string;
    topic: string;
    query: string;
    answer: string | null;
    confidence: string | null;
    replyMode: string | null;
    rating: string | null;
    instructorNote: string | null;
    createdAt: string;
}

interface QueryLogPage extends PaginatedResponse<QueryLogItem> {
    frequent: { topic: string; query: string; count: number }[];
}

const PAGE_SIZE = 20;

function QueryLogEntry({ item, courseId }: { item: QueryLogItem; courseId: string }) {
    const toast = useToast();
    const [note, setNote] = useState(item.instructorNote ?? '');
    const [correction, setCorrection] = useState('');
    const [saving, setSaving] = useState(false);

    const save = async (kind: 'note' | 'correct') => {
        setSaving(true);
        const path = `/reviews/queries/${item.id}/${kind}?courseId=${courseId}`;
        try {
            if (kind === 'note') {
                await api.patch<{ instructorNote: string | null }>(path, { note });
                toast.showToast('Instructor note saved.');
            } else {
                const result = await api.post<{ indexed: boolean }>(path, { correction });
                toast.showToast(result.indexed ? 'Correction saved and indexed.' : 'Correction saved. Indexing is unavailable.',
                    result.indexed ? undefined : 'error');
            }
        } catch (error) {
            toast.showToast(error instanceof Error ? error.message : 'Could not save your changes.', 'error');
        } finally {
            setSaving(false);
        }
    };

    const issue = item.replyMode === 'not_covered' || item.replyMode === 'error' || !item.answer ? 'Unanswered'
        : item.confidence === 'low' ? 'Low confidence' : item.rating === 'down' ? 'Rated down' : null;
    return (
        <details className="air-query-entry">
            <summary>
                <span>{item.query}</span>
                <small>{item.topic} · {new Date(item.createdAt).toLocaleString()}{issue ? ` · ${issue}` : ''}</small>
            </summary>
            <div className="air-query-content">
                {item.answer && <div>{renderAnswer(item.answer)}</div>}
                <label>Private instructor note<textarea className="input" rows={2} maxLength={4000} value={note}
                    onChange={(event) => setNote(event.target.value)} /></label>
                <button type="button" className="btn btn-outline btn-sm" disabled={saving} onClick={() => void save('note')}>
                    <Save size={14} /> Save note
                </button>
                <label>Corrected answer<textarea className="input" rows={3} maxLength={4000} value={correction}
                    onChange={(event) => setCorrection(event.target.value)} /></label>
                <button type="button" className="btn btn-primary btn-sm" disabled={saving || correction.trim().length < 5}
                    onClick={() => void save('correct')}>Save correction to course</button>
            </div>
        </details>
    );
}

export default function CourseQueryLog({ courseId }: { courseId: string }) {
    const [topic, setTopic] = useState('');
    const [onDate, setOnDate] = useState('');
    const [offset, setOffset] = useState(0);
    const [result, setResult] = useState<{ key: string; data?: QueryLogPage; error?: string } | null>(null);
    const key = JSON.stringify([courseId, topic, onDate, offset]);
    const current = result?.key === key ? result : null;
    const data = current?.data;

    useEffect(() => {
        let cancelled = false;
        const timer = window.setTimeout(() => {
            void api.get<QueryLogPage>('/reviews/queries', {
                courseId, topic: topic.trim() || undefined, onDate: onDate || undefined, limit: PAGE_SIZE, offset,
            }).then((page) => { if (!cancelled) setResult({ key, data: page }); })
                .catch((error) => { if (!cancelled) setResult({ key, error: error instanceof Error ? error.message : 'Could not load course questions.' }); });
        }, 180);
        return () => { cancelled = true; window.clearTimeout(timer); };
    }, [courseId, key, offset, onDate, topic]);

    return (
        <section className="air-panel">
            <div className="air-header"><h2><MessageSquare size={18} /> Course Query Log</h2></div>
            <div className="air-query-filters">
                <label>Topic<input type="search" className="input" value={topic} maxLength={128}
                    onChange={(event) => { setTopic(event.target.value); setOffset(0); }} /></label>
                <label>Date (UTC)<input type="date" className="input" value={onDate}
                    onChange={(event) => { setOnDate(event.target.value); setOffset(0); }} /></label>
            </div>
            {current?.error && <p role="alert">{current.error}</p>}
            {!current && <p role="status">Loading course queries…</p>}
            {data && (
                <>
                    {Boolean(data.frequent.length) && <section className="air-frequent-queries">
                        <h3>Frequently asked by topic</h3>
                        <ul>{data.frequent.map((question) => <li key={`${question.topic}:${question.query}`}>
                            <strong>{question.topic}</strong> · {question.query} · {question.count} times
                        </li>)}</ul>
                    </section>}
                    {data.items.length === 0 && <p>No course queries match.</p>}
                    {data.items.map((item) => <QueryLogEntry key={item.id} item={item} courseId={courseId} />)}
                    {data.total > PAGE_SIZE && <Pagination total={data.total} offset={offset} limit={PAGE_SIZE} onPageChange={setOffset} />}
                </>
            )}
        </section>
    );
}
