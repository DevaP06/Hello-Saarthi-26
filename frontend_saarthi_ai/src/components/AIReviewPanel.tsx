import { useCallback, useEffect, useState } from 'react';
import { CheckCircle2, MessageSquareWarning, XCircle } from 'lucide-react';
import { api, type PaginatedResponse } from '../lib/api';
import { useToast } from './Toast';
import './AIReviewPanel.css';

interface AIReview {
  id: string;
  courseId: string;
  studentName?: string | null;
  query: string;
  answer: string;
  replyMode?: string | null;
  confidence?: string | null;
  reasons: string[];
  source: 'low_confidence' | 'thumbs_down';
  status: 'open' | 'corrected' | 'dismissed';
  occurrences: number;
  correction?: string | null;
  createdAt: string;
}

type Filter = 'open' | 'corrected' | 'dismissed';

const FILTERS: { id: Filter; label: string }[] = [
  { id: 'open', label: 'Needs review' },
  { id: 'corrected', label: 'Corrected' },
  { id: 'dismissed', label: 'Dismissed' },
];

function sourceLabel(r: AIReview): string {
  if (r.source === 'thumbs_down') return 'Student rated it down';
  if (r.replyMode === 'not_covered') return 'Not in the course materials';
  return 'Low confidence';
}

/** Instructor view of AI answers that need a human: write the right answer and Saarthi uses it next time. */
export default function AIReviewPanel({ courseId }: { courseId: string }) {
  const toast = useToast();
  const [filter, setFilter] = useState<Filter>('open');
  const [items, setItems] = useState<AIReview[]>([]);
  const [counts, setCounts] = useState<{ open: number; corrected: number; dismissed: number }>({ open: 0, corrected: 0, dismissed: 0 });
  const [loading, setLoading] = useState(true);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [list, stats] = await Promise.all([
        api.get<PaginatedResponse<AIReview>>('/reviews', { courseId, status: filter, limit: 50, offset: 0 }),
        api.get<{ open: number; corrected: number; dismissed: number }>('/reviews/stats', { courseId }),
      ]);
      setItems(list.items || []);
      setCounts(stats);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not load AI answers to review.');
    } finally {
      setLoading(false);
    }
  }, [courseId, filter]);

  useEffect(() => { void load(); }, [load]);

  const save = async (r: AIReview) => {
    const text = (drafts[r.id] ?? r.correction ?? '').trim();
    if (text.length < 5) {
      toast.showToast('Write the correct answer first', 'error');
      return;
    }
    setBusyId(r.id);
    try {
      const res = await api.post<{ indexed: boolean }>(`/reviews/${r.id}/correct`, { correction: text });
      toast.showToast(res.indexed ? 'Saved. Saarthi will use your answer from now on.' : 'Saved, but it could not be added to the course index yet (is the model server running?).', res.indexed ? undefined : 'error');
      await load();
    } catch (e) {
      toast.showToast(e instanceof Error ? e.message : 'Could not save', 'error');
    } finally {
      setBusyId(null);
    }
  };

  const dismiss = async (r: AIReview) => {
    setBusyId(r.id);
    try {
      await api.post(`/reviews/${r.id}/dismiss`);
      await load();
    } catch (e) {
      toast.showToast(e instanceof Error ? e.message : 'Could not dismiss', 'error');
    } finally {
      setBusyId(null);
    }
  };

  return (
    <div className="air-panel">
      <div className="air-header">
        <h2><MessageSquareWarning size={18} /> AI answers to review</h2>
        <p>
          Questions Saarthi could not answer from your materials, or that students rated down.
          Write the correct answer once: it is added to this course and used for the next student who asks.
        </p>
      </div>

      <div className="air-filters" role="tablist">
        {FILTERS.map((f) => (
          <button
            key={f.id}
            type="button"
            role="tab"
            aria-selected={filter === f.id}
            className={`air-filter ${filter === f.id ? 'active' : ''}`}
            onClick={() => setFilter(f.id)}
          >
            {f.label} <span className="air-count">{counts[f.id]}</span>
          </button>
        ))}
      </div>

      {error && <div className="air-error" role="alert">{error}</div>}
      {loading && <p className="air-muted">Loading…</p>}
      {!loading && !error && items.length === 0 && (
        <p className="air-muted">{filter === 'open' ? 'Nothing needs your attention.' : 'Nothing here yet.'}</p>
      )}

      {items.map((r) => (
        <div key={r.id} className="air-card">
          <div className="air-card-top">
            <span className="air-badge">{sourceLabel(r)}</span>
            {r.occurrences > 1 && <span className="air-badge air-badge-muted">asked {r.occurrences}×</span>}
            <span className="air-student">{r.studentName || 'Student'}</span>
          </div>
          <p className="air-question">{r.query}</p>
          {r.reasons.length > 0 && <p className="air-muted">Why flagged: {r.reasons.join('; ')}</p>}
          {r.answer && (
            <details className="air-answer">
              <summary>What Saarthi answered</summary>
              <pre>{r.answer}</pre>
            </details>
          )}

          {r.status === 'corrected' ? (
            <div className="air-correction">
              <CheckCircle2 size={14} /> <strong>Your answer:</strong> {r.correction}
            </div>
          ) : null}

          {r.status !== 'dismissed' && (
            <textarea
              className="input air-textarea"
              rows={3}
              placeholder="Write the correct answer, in your course's own notation…"
              value={drafts[r.id] ?? r.correction ?? ''}
              onChange={(e) => setDrafts((d) => ({ ...d, [r.id]: e.target.value }))}
            />
          )}
          <div className="air-actions">
            {r.status !== 'dismissed' && (
              <button type="button" className="btn btn-primary btn-sm" disabled={busyId === r.id} onClick={() => void save(r)}>
                {busyId === r.id ? 'Saving…' : r.status === 'corrected' ? 'Update answer' : 'Save answer'}
              </button>
            )}
            {r.status === 'open' && (
              <button type="button" className="btn btn-outline btn-sm" disabled={busyId === r.id} onClick={() => void dismiss(r)}>
                <XCircle size={14} /> Dismiss
              </button>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
