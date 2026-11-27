import { useRef, useState } from 'react';
import { Link2, Paperclip, Plus, X } from 'lucide-react';
import { getUploadFullUrl, uploadFile } from '../lib/api';
import type { AssignmentAttachment } from '../lib/assignmentStatus';
import { useToast } from './Toast';

const MAX_ATTACHMENTS = 20;

interface Props {
    value: AssignmentAttachment[];
    onChange: (next: AssignmentAttachment[]) => void;
    /** A link typed but not yet added, so saving the form can still include it (null when none or invalid). */
    onPendingLinkChange?: (link: AssignmentAttachment | null) => void;
}

/** "example.com/x" → "https://example.com/x"; null when it is not a valid http(s) URL. */
export function normaliseLink(raw: string): string | null {
    let url = raw.trim();
    if (!url) return null;
    if (!/^https?:\/\//i.test(url)) url = `https://${url}`;
    try {
        const u = new URL(url);
        return u.hostname.includes('.') || u.hostname === 'localhost' ? url : null;
    } catch {
        return null;
    }
}

export function attachmentHref(url: string): string {
    return url.startsWith('/') ? getUploadFullUrl(url) : url;
}

/** Teacher form field: upload files and add links to an assignment. */
export default function AssignmentAttachmentsField({ value, onChange, onPendingLinkChange }: Props) {
    const toast = useToast();
    const fileInput = useRef<HTMLInputElement>(null);
    const [uploading, setUploading] = useState(false);
    const [link, setLink] = useState('');
    const [linkName, setLinkName] = useState('');
    const [showLink, setShowLink] = useState(false);
    const full = value.length >= MAX_ATTACHMENTS;

    const addFiles = async (picked: FileList | null) => {
        if (!picked?.length) return;
        setUploading(true);
        const added: AssignmentAttachment[] = [];
        try {
            for (const f of Array.from(picked).slice(0, MAX_ATTACHMENTS - value.length)) {
                const { url } = await uploadFile(f);
                added.push({ name: f.name, url });
            }
        } catch (e) {
            toast.showToast(e instanceof Error ? e.message : 'Upload failed', 'error');
        } finally {
            if (added.length) onChange([...value, ...added]);
            setUploading(false);
            if (fileInput.current) fileInput.current.value = '';
        }
    };

    const reportPending = (nextLink: string, nextName: string) => {
        const url = normaliseLink(nextLink);
        onPendingLinkChange?.(url ? { name: nextName.trim() || url, url } : null);
    };

    const editLink = (nextLink: string, nextName: string) => {
        setLink(nextLink);
        setLinkName(nextName);
        reportPending(nextLink, nextName);
    };

    const addLink = () => {
        if (!link.trim()) return;
        const url = normaliseLink(link);
        if (!url) {
            toast.showToast('Enter a valid link, e.g. https://example.com', 'error');
            return;
        }
        onChange([...value, { name: linkName.trim() || url, url }]);
        editLink('', '');
        setShowLink(false);
    };

    return (
        <div className="cd-modal-field">
            <label>Attachments</label>
            {value.length > 0 && (
                <ul className="cd-work-files">
                    {value.map((a, i) => (
                        <li key={`${a.url}-${i}`}>
                            <a href={attachmentHref(a.url)} target="_blank" rel="noopener noreferrer" className="cd-assignment-file">
                                {a.url.startsWith('/') ? <Paperclip size={14} /> : <Link2 size={14} />} {a.name}
                            </a>
                            <button type="button" className="btn-icon" aria-label={`Remove ${a.name}`} onClick={() => onChange(value.filter((_, j) => j !== i))}>
                                <X size={14} />
                            </button>
                        </li>
                    ))}
                </ul>
            )}
            {showLink && (
                <div className="cd-modal-row">
                    <input className="input" type="url" placeholder="https://…" aria-label="Link" value={link} autoFocus
                        onChange={(e) => editLink(e.target.value, linkName)} onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); addLink(); } }} />
                    <input className="input" placeholder="Name (optional)" aria-label="Link name" value={linkName} maxLength={200}
                        onChange={(e) => editLink(link, e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); addLink(); } }} />
                </div>
            )}
            <div className="cd-work-actions">
                <input ref={fileInput} type="file" multiple hidden onChange={(e) => addFiles(e.target.files)} />
                <button type="button" className="btn btn-outline btn-sm" disabled={uploading || full} onClick={() => fileInput.current?.click()}>
                    <Plus size={14} /> {uploading ? 'Uploading…' : 'Add file'}
                </button>
                {showLink ? (
                    <>
                        <button type="button" className="btn btn-outline btn-sm" disabled={!link.trim()} onClick={addLink}>Add link</button>
                        <button type="button" className="btn btn-ghost btn-sm" onClick={() => { setShowLink(false); editLink('', ''); }}>Cancel</button>
                    </>
                ) : (
                    <button type="button" className="btn btn-outline btn-sm" disabled={full} onClick={() => setShowLink(true)}>
                        <Link2 size={14} /> Add link
                    </button>
                )}
            </div>
        </div>
    );
}
