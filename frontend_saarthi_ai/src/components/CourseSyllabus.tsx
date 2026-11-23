import { useRef, useState } from 'react';
import { Download, Upload } from 'lucide-react';
import { apiUrl, authFetch, type CourseResponse } from '../lib/api';

export default function CourseSyllabus({ course, canEdit, onUpdated }: {
    course: CourseResponse;
    canEdit: boolean;
    onUpdated: (course: CourseResponse) => void;
}) {
    const input = useRef<HTMLInputElement>(null);
    const [busy, setBusy] = useState(false);
    const [feedback, setFeedback] = useState('');
    const url = apiUrl(`/courses/${course.id}/syllabus`);

    async function upload(file: File) {
        const form = new FormData();
        form.set('file', file);
        setBusy(true);
        setFeedback('');
        try {
            const response = await authFetch(url, { method: 'PUT', body: form });
            const result = await response.json().catch(() => null);
            if (!response.ok || !result) throw new Error(result?.error?.message ?? 'Could not upload the syllabus.');
            onUpdated(result as CourseResponse);
            setFeedback('Syllabus saved.');
        } catch (error) {
            setFeedback(error instanceof Error ? error.message : 'Could not upload the syllabus.');
        } finally {
            setBusy(false);
        }
    }

    async function download() {
        if (!course.syllabusName) return;
        setBusy(true);
        setFeedback('');
        try {
            const response = await authFetch(url);
            if (!response.ok) throw new Error('Could not download the syllabus.');
            const blobUrl = URL.createObjectURL(await response.blob());
            const link = document.createElement('a');
            link.href = blobUrl;
            link.download = course.syllabusName;
            document.body.append(link);
            link.click();
            link.remove();
            window.setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
        } catch (error) {
            setFeedback(error instanceof Error ? error.message : 'Could not download the syllabus.');
        } finally {
            setBusy(false);
        }
    }

    if (!course.syllabusName && !canEdit) return null;
    return (
        <section className="cd-upcoming-box cd-syllabus" aria-label="Course syllabus">
            <h4>Syllabus</h4>
            {course.syllabusName && (
                <>
                    <p>{course.syllabusName}</p>
                    <button type="button" className="btn btn-outline btn-sm" disabled={busy} onClick={() => void download()}>
                        <Download size={15} /> Download syllabus
                    </button>
                </>
            )}
            {canEdit && (
                <>
                    <input type="file" ref={input} accept=".pdf,application/pdf" hidden onChange={(event) => {
                        const file = event.currentTarget.files?.[0];
                        event.currentTarget.value = '';
                        if (file) void upload(file);
                    }} />
                    <button type="button" className="btn btn-outline btn-sm" disabled={busy} onClick={() => input.current?.click()}>
                        <Upload size={15} /> {course.syllabusName ? 'Replace syllabus' : 'Upload syllabus PDF'}
                    </button>
                </>
            )}
            {feedback && <p role="status">{feedback}</p>}
        </section>
    );
}
