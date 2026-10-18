import { useCallback, useEffect, useState } from 'react';
import { Copy, Mail, Users, X } from 'lucide-react';
import { api } from '../lib/api';
import { useToast } from './Toast';

interface Invite {
    id: string;
    email: string;
    inviteCode: string;
    accepted: boolean;
    expiresAt: string;
}

interface Props {
    courseId: string;
    courseTitle: string;
    onClose: () => void;
}

const joinLink = (code: string) => `${window.location.origin}${import.meta.env.BASE_URL}join?code=${code}`;

export default function InviteModal({ courseId, courseTitle, onClose }: Props) {
    const { showToast } = useToast();
    const [invites, setInvites] = useState<Invite[]>([]);
    const [email, setEmail] = useState('');
    const [busy, setBusy] = useState(false);

    const load = useCallback(() => {
        api.get<Invite[]>(`/courses/${courseId}/invites`)
            .then(setInvites)
            .catch((e) => showToast(e instanceof Error ? e.message : 'Failed to load invites', 'error'));
    }, [courseId, showToast]);

    useEffect(load, [load]);

    const create = async (target: string) => {
        setBusy(true);
        try {
            await api.post(`/courses/${courseId}/invite`, { email: target });
            setEmail('');
            showToast(target === '*' ? 'Class code created' : `Invite sent to ${target}`);
            load();
        } catch (e) {
            showToast(e instanceof Error ? e.message : 'Failed to create invite', 'error');
        } finally {
            setBusy(false);
        }
    };

    const copy = (text: string) =>
        navigator.clipboard.writeText(text).then(
            () => showToast('Copied'),
            () => showToast('Copy failed', 'error'),
        );

    const now = Date.now();
    const classCodes = invites.filter((i) => i.email === '*' && new Date(i.expiresAt).getTime() > now);
    const emailInvites = invites.filter((i) => i.email !== '*');

    return (
        <div className="course-modal-overlay" onClick={onClose}>
            <div className="course-modal animate-scale-in" onClick={(e) => e.stopPropagation()}>
                <div className="course-modal-header">
                    <h2>Invite students · {courseTitle}</h2>
                    <button className="course-modal-close" onClick={onClose}><X size={20} /></button>
                </div>
                <div className="course-modal-body">
                    <div className="course-modal-field">
                        <label><Users size={14} /> Class code (anyone with it can join, valid 7 days)</label>
                        {classCodes.map((c) => (
                            <div key={c.id} style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', marginBottom: '0.5rem' }}>
                                <code style={{ fontSize: '1.1rem', letterSpacing: '0.05em' }}>{c.inviteCode}</code>
                                <button className="btn btn-outline btn-sm" onClick={() => copy(c.inviteCode)}><Copy size={14} /> Code</button>
                                <button className="btn btn-outline btn-sm" onClick={() => copy(joinLink(c.inviteCode))}><Copy size={14} /> Link</button>
                            </div>
                        ))}
                        <button className="btn btn-primary btn-sm" disabled={busy} onClick={() => create('*')}>
                            {classCodes.length ? 'Generate another code' : 'Generate class code'}
                        </button>
                    </div>

                    <div className="course-modal-field">
                        <label><Mail size={14} /> Invite by email (valid 72 hours)</label>
                        <div style={{ display: 'flex', gap: '0.5rem' }}>
                            <input className="input" type="email" placeholder="student@example.com" value={email} onChange={(e) => setEmail(e.target.value)} />
                            <button className="btn btn-primary btn-sm" disabled={busy || !email.trim()} onClick={() => create(email.trim())}>Invite</button>
                        </div>
                    </div>

                    {emailInvites.length > 0 && (
                        <div className="course-modal-field">
                            <label>Email invites</label>
                            {emailInvites.map((i) => (
                                <div key={i.id} style={{ display: 'flex', justifyContent: 'space-between', gap: '0.5rem', fontSize: '0.85rem', padding: '0.25rem 0' }}>
                                    <span>{i.email}</span>
                                    <span>{i.accepted ? 'Joined' : new Date(i.expiresAt).getTime() < now ? 'Expired' : 'Pending'}</span>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
