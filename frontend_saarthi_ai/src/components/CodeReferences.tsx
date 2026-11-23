import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { BookOpen, ExternalLink } from 'lucide-react';
import { authFetch, getAllPages, getMaterialFileUrl, type MaterialResponse } from '../lib/api';
import { resourceLink } from '../lib/resourceLinks';

export default function CodeReferences({ courseId, topic }: { courseId: number; topic?: string | null }) {
    const [materials, setMaterials] = useState<MaterialResponse[] | null>(null);
    const [selectedId, setSelectedId] = useState('');
    const [preview, setPreview] = useState<{ id: string; url?: string; text?: string; error?: string } | null>(null);
    const [error, setError] = useState<string | null>(null);
    const selected = materials?.find((material) => material.id === selectedId) ?? materials?.[0];

    useEffect(() => {
        let cancelled = false;
        const topicKey = topic?.trim().toLowerCase();
        void getAllPages<MaterialResponse>(`/courses/${courseId}/materials`)
            .then((items) => {
                if (cancelled) return;
                setMaterials(items.filter((item) => item.type !== 'code' && (!topicKey || item.topic?.trim().toLowerCase() === topicKey)));
            })
            .catch(() => { if (!cancelled) setError('Could not load references.'); });
        return () => { cancelled = true; };
    }, [courseId, topic]);

    const uploadedId = selected && /\/uploads\/|\/materials\/\d+\/file$/.test(selected.url) ? selected.id : undefined;
    useEffect(() => {
        if (!uploadedId) return;
        const controller = new AbortController();
        let objectUrl: string | undefined;
        const load = async () => {
            try {
                const response = await authFetch(getMaterialFileUrl(String(courseId), uploadedId), { signal: controller.signal });
                if (!response.ok) throw new Error('Could not preview this reference.');
                const type = response.headers.get('content-type') ?? '';
                if (!type.includes('application/pdf') && !type.startsWith('text/')) {
                    if (!controller.signal.aborted) setPreview({ id: uploadedId });
                    return;
                }
                const blob = await response.blob();
                if (controller.signal.aborted) return;
                if (type.includes('application/pdf')) {
                    objectUrl = URL.createObjectURL(blob);
                    setPreview({ id: uploadedId, url: objectUrl });
                } else {
                    const text = await blob.slice(0, 24_000).text();
                    if (!controller.signal.aborted) setPreview({ id: uploadedId, text });
                }
            } catch {
                if (!controller.signal.aborted) setPreview({ id: uploadedId, error: 'Could not preview this reference.' });
            }
        };
        void load();
        return () => {
            controller.abort();
            if (objectUrl) URL.revokeObjectURL(objectUrl);
        };
    }, [courseId, uploadedId]);

    const link = selected ? resourceLink({ courseId, materialId: selected.id }) : null;
    const current = preview?.id === uploadedId ? preview : null;
    return (
        <aside className="codelab-reference-panel" aria-label="Lesson references">
            <h3><BookOpen size={16} /> Lesson references</h3>
            {error && <p role="alert">{error}</p>}
            {!error && materials === null && <p role="status">Loading references…</p>}
            {materials?.length === 0 && <p>No notes or solved examples are attached to this lesson.</p>}
            {selected && <>
                <label>Resource<select className="input" value={selected.id} onChange={(event) => setSelectedId(event.target.value)}>
                    {materials?.map((item) => <option key={item.id} value={item.id}>{item.title}</option>)}
                </select></label>
                {link && <Link to={link} target="_blank" rel="noopener noreferrer" className="btn btn-outline btn-sm">
                    <ExternalLink size={14} /> Open full resource
                </Link>}
                {current?.url && <iframe title={selected.title} src={current.url} />}
                {current?.text && <pre>{current.text}</pre>}
                {current?.error && <p role="status">{current.error}</p>}
                {uploadedId && !current && <p role="status">Loading preview…</p>}
            </>}
        </aside>
    );
}
