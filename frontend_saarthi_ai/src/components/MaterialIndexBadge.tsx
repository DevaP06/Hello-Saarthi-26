import { useCallback, useEffect, useState } from 'react';
import { Loader2, RefreshCw } from 'lucide-react';
import { api } from '../lib/api';

type IndexState = 'queued' | 'running' | 'done' | 'error' | 'unsupported' | 'none';
interface IndexStatus {
    status: IndexState;
    message?: string;
    chunks?: number;
}

const LABEL: Record<IndexState, string> = {
    queued: 'Queued for the AI',
    running: 'Indexing…',
    done: 'Ready for the AI',
    error: 'Indexing failed',
    unsupported: 'Not searchable',
    none: 'Not indexed',
};

/** Teacher view of a material: can the AI search it yet? Polls while indexing runs. */
export default function MaterialIndexBadge({ courseId, materialId }: { courseId: string; materialId: string }) {
    const [st, setSt] = useState<IndexStatus | null>(null);

    const load = useCallback(() => {
        api.get<IndexStatus>(`/courses/${courseId}/materials/${materialId}/index-status`)
            .then(setSt)
            .catch(() => setSt(null));
    }, [courseId, materialId]);

    useEffect(() => { load(); }, [load]);

    useEffect(() => {
        if (st?.status !== 'queued' && st?.status !== 'running') return;
        const t = window.setTimeout(load, 3000);
        return () => window.clearTimeout(t);
    }, [st, load]);

    const reindex = async () => {
        setSt({ status: 'queued' });
        try {
            const r = await api.post<{ status: string; reason?: string }>(`/courses/${courseId}/materials/${materialId}/reindex`);
            if (r.status === 'skipped') setSt({ status: 'unsupported', message: r.reason });
            else load();
        } catch (e) {
            setSt({ status: 'error', message: e instanceof Error ? e.message : 'Could not start indexing' });
        }
    };

    if (!st) return null;
    const busy = st.status === 'queued' || st.status === 'running';
    return (
        <span className={`cd-index-badge cd-index-${st.status}`} title={st.message || undefined} role="status">
            {busy && <Loader2 size={12} className="animate-spin" />}
            {LABEL[st.status] ?? st.status}
            {st.status === 'done' && st.chunks ? ` · ${st.chunks} chunks` : ''}
            {(st.status === 'error' || st.status === 'none') && (
                <button type="button" className="cd-index-retry" onClick={reindex} title="Index this file so the AI can answer from it">
                    <RefreshCw size={12} /> {st.status === 'error' ? 'Retry' : 'Index now'}
                </button>
            )}
        </span>
    );
}
