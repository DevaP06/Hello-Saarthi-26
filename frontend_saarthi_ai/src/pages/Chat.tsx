import { useState, useRef, useEffect, useCallback } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { Send, Sparkles, User, Bot, Loader2, BookOpen, Lightbulb, Code2, GraduationCap, Plus, Pencil, Trash2, Check, X, ThumbsUp, ThumbsDown } from 'lucide-react';
import { api, getAllPages, type ChatAnswerMetadata, type ChatMessageItemResponse, type ConversationResponse, type ConversationDetailResponse, type EnrollmentWithCourseResponse, type ModuleResponse, type SendMessageResponse, type PaginatedResponse } from '../lib/api';
import ConfirmModal from '../components/ConfirmModal';
import EmptyState from '../components/EmptyState';
import Pagination from '../components/Pagination';
import AnswerSources from '../components/AnswerSources';
import { renderAnswer } from '../lib/renderAnswer';
import { streamChat } from '../lib/streamChat';
import './Chat.css';

const GREETING = "Hello! I'm Saarthi, your AI learning companion. I specialize in:\n\n• **Digital Signal Processing** - FFT, filters, transforms\n• **Machine Learning** - Neural networks, optimization\n• **Data Structures** - Trees, graphs, algorithms\n• **Pattern Recognition** - Classification, feature extraction\n\nHow can I help you today?";
const HISTORY_PAGE_SIZE = 20;

interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
    rating?: 'up' | 'down' | null;
    topic?: string | null;
    replyToId?: number | null;
    answerMetadata?: ChatAnswerMetadata | null;
}

/** Saved messages have numeric ids; the greeting and optimistic messages do not. */
const isSaved = (m: Message) => /^\d+$/.test(m.id);

const suggestedTopics = [
    { icon: BookOpen, label: 'Explain Fourier Transform' },
    { icon: Code2, label: 'Binary Search implementation' },
    { icon: Lightbulb, label: 'What is gradient descent?' },
    { icon: GraduationCap, label: 'DSP exam preparation tips' },
];

function messageFromApi(m: ChatMessageItemResponse): Message {
    return { id: m.id, role: m.role, content: m.content, timestamp: new Date(m.createdAt), rating: m.rating,
        topic: m.topic, replyToId: m.replyToId, answerMetadata: m.answerMetadata };
}

export default function ChatPage() {
    const [searchParams, setSearchParams] = useSearchParams();
    const linkedCourse = searchParams.get('course') ?? '';
    const linkedMaterial = searchParams.get('material') ?? '';
    const [conversations, setConversations] = useState<ConversationResponse[]>([]);
    const [currentId, setCurrentId] = useState<string | null>(null);
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [loadingList, setLoadingList] = useState(true);
    const [loadingChat, setLoadingChat] = useState(false);
    const [sending, setSending] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [editingId, setEditingId] = useState<string | null>(null);
    const [editTitle, setEditTitle] = useState('');
    const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
    const [courseId, setCourseId] = useState(linkedCourse);
    const [moduleId, setModuleId] = useState('');
    const [answerMode, setAnswerMode] = useState<'auto' | 'socratic' | 'direct'>('auto');
    const [enrolledCourses, setEnrolledCourses] = useState<EnrollmentWithCourseResponse[] | null>(null);
    const [courseListError, setCourseListError] = useState(false);
    const courses = enrolledCourses ?? [];
    const [moduleList, setModuleList] = useState<{ courseId: string; items: ModuleResponse[] } | null>(null);
    const [historyTopic, setHistoryTopic] = useState('');
    const [historyDate, setHistoryDate] = useState('');
    const [historyOffset, setHistoryOffset] = useState(0);
    const [historyTotal, setHistoryTotal] = useState(0);
    const [streamStatus, setStreamStatus] = useState('');
    const historyRequest = useRef(0);
    const endRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLTextAreaElement>(null);
    const renameInputRef = useRef<HTMLInputElement>(null);
    const contextMaterialId = courseId && courseId === linkedCourse && linkedMaterial ? Number(linkedMaterial) : undefined;
    const modules = moduleList?.courseId === courseId ? moduleList.items : [];

    const fetchConversations = useCallback(async () => {
        const request = ++historyRequest.current;
        setLoadingList(true);
        try {
            const res = await api.get<PaginatedResponse<ConversationResponse>>('/chat/conversations', {
                limit: HISTORY_PAGE_SIZE, offset: historyOffset, topic: historyTopic.trim() || undefined,
                onDate: historyDate || undefined,
            });
            if (request !== historyRequest.current) return;
            setConversations(res.items || []);
            setHistoryTotal(res.total);
        } catch (e) {
            if (request !== historyRequest.current) return;
            const msg = e instanceof Error ? e.message : 'Failed to load conversations';
            setError(msg);
            if (msg.includes('401') || msg.toLowerCase().includes('unauthorized')) setError('Please log in to use chat.');
        } finally {
            if (request === historyRequest.current) setLoadingList(false);
        }
    }, [historyTopic, historyDate, historyOffset]);

    useEffect(() => {
        const timer = window.setTimeout(() => void fetchConversations(), 180);
        return () => { window.clearTimeout(timer); historyRequest.current += 1; };
    }, [fetchConversations]);

    useEffect(() => {
        let cancelled = false;
        void getAllPages<EnrollmentWithCourseResponse>('/courses/my/enrollments')
            .then((items) => { if (!cancelled) setEnrolledCourses(items); })
            .catch(() => { if (!cancelled) setCourseListError(true); });
        return () => { cancelled = true; };
    }, []);

    useEffect(() => {
        if (!courseId) return;
        let cancelled = false;
        void api.get<ModuleResponse[]>(`/courses/${courseId}/modules`)
            .then((items) => { if (!cancelled) setModuleList({ courseId, items }); })
            .catch(() => { if (!cancelled) setModuleList({ courseId, items: [] }); });
        return () => { cancelled = true; };
    }, [courseId]);

    const loadConversation = useCallback(async (id: string) => {
        setEditingId(null);
        setCurrentId(id);
        setLoadingChat(true);
        setError(null);
        try {
            const res = await api.get<ConversationDetailResponse>(`/chat/conversations/${id}`);
            setCourseId(res.courseId ? String(res.courseId) : '');
            setModuleId('');
            setMessages((res.messages || []).map(messageFromApi));
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Failed to load conversation');
            setMessages([]);
        } finally {
            setLoadingChat(false);
        }
    }, []);

    const startRename = (e: React.MouseEvent, c: ConversationResponse) => {
        e.stopPropagation();
        setEditingId(c.id);
        setEditTitle(c.title || 'New Chat');
        setTimeout(() => renameInputRef.current?.focus(), 0);
    };

    const submitRename = async (id: string) => {
        const title = editTitle.trim() || 'New Chat';
        setError(null);
        try {
            await api.patch<ConversationResponse>(`/chat/conversations/${id}`, { title });
            setConversations((prev) => prev.map((c) => (c.id === id ? { ...c, title } : c)));
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Failed to rename');
        }
        setEditingId(null);
    };

    const cancelRename = () => {
        setEditingId(null);
        setEditTitle('');
    };

    const handleDeleteClick = (e: React.MouseEvent, id: string) => {
        e.stopPropagation();
        setDeleteConfirmId(id);
    };

    const handleDeleteConfirm = async () => {
        if (!deleteConfirmId) return;
        const id = deleteConfirmId;
        setDeleteConfirmId(null);
        setError(null);
        try {
            await api.delete(`/chat/conversations/${id}`);
            setConversations((prev) => prev.filter((c) => c.id !== id));
            if (currentId === id) {
                setCurrentId(null);
                setMessages([]);
            }
            if (conversations.length === 1 && historyOffset > 0) setHistoryOffset(Math.max(0, historyOffset - HISTORY_PAGE_SIZE));
            else void fetchConversations();
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Failed to delete');
        }
    };

    const handleNewChat = () => {
        setError(null);
        setCurrentId(null);
        setMessages([]);
        setInput('');
        inputRef.current?.focus();
    };

    const changeCourse = (nextCourse: string) => {
        if (nextCourse === courseId) return;
        setCourseId(nextCourse);
        setModuleId('');
        setCurrentId(null);
        setMessages([]);
        setError(null);
        setStreamStatus('');
        if (linkedCourse || linkedMaterial) {
            const nextParams = new URLSearchParams(searchParams);
            nextParams.delete('course');
            nextParams.delete('material');
            setSearchParams(nextParams, { replace: true });
        }
    };

    const sendMessage = async () => {
        if (!input.trim() || sending || loadingChat) return;
        const q = input.trim();
        setInput('');
        setError(null);
        setSending(true);
        setStreamStatus('');

        let conversationId = currentId;
        if (!conversationId) {
            try {
                const conv = await api.post<ConversationResponse>('/chat/conversations', {
                    title: 'New Chat', courseId: courseId ? Number(courseId) : undefined,
                });
                setCurrentId(conv.id);
                conversationId = conv.id;
                setMessages([]);
            } catch (e) {
                setError(e instanceof Error ? e.message : 'Failed to create chat');
                setInput(q);
                setSending(false);
                return;
            }
        }

        const optimisticUser: Message = { id: `t-${Date.now()}`, role: 'user', content: q, timestamp: new Date() };
        const pendingId = `${optimisticUser.id}-answer`;
        setMessages((prev) => [...prev, optimisticUser]);
        try {
            const data = courseId || answerMode === 'socratic' ? await streamChat({
                conversationId, message: q, courseId: courseId ? Number(courseId) : undefined,
                moduleId: moduleId ? Number(moduleId) : undefined, contextMaterialId, answerMode,
            }, (delta) => setMessages((prev) => {
                const pending = prev.find((m) => m.id === pendingId);
                if (pending) return prev.map((m) => m.id === pendingId ? { ...m, content: m.content + delta } : m);
                return [...prev, { id: pendingId, role: 'assistant', content: delta, timestamp: new Date() }];
            }), setStreamStatus) : await api.post<SendMessageResponse>(`/chat/conversations/${conversationId}/messages`, { message: q });
            setMessages((prev) => {
                const withoutOptimistic = prev.filter((m) => m.id !== optimisticUser.id && m.id !== pendingId);
                return [...withoutOptimistic, messageFromApi(data.userMessage), messageFromApi(data.assistantMessage)];
            });
            void fetchConversations();
        } catch (e) {
            setMessages((prev) => prev.filter((m) => m.id !== optimisticUser.id && m.id !== pendingId));
            setError(e instanceof Error ? e.message : 'Failed to send message');
            setInput(q);
        } finally {
            setSending(false);
        }
    };

    // Thumbs up/down on an answer: saved on the message, so the teacher and the eval see it.
    const rateAnswer = async (msg: Message, rating: 'up' | 'down') => {
        if (msg.rating === rating) return;
        const i = messages.findIndex((m) => m.id === msg.id);
        const query = messages.find((m) => Number(m.id) === msg.replyToId)?.content
            || messages.slice(0, i).reverse().find((m) => m.role === 'user')?.content || '(no question)';
        setMessages((prev) => prev.map((m) => (m.id === msg.id ? { ...m, rating } : m)));
        try {
            await api.post('/chat/feedback', { query, rating, messageId: Number(msg.id), conversationId: currentId ?? undefined });
        } catch (e) {
            setMessages((prev) => prev.map((m) => (m.id === msg.id ? { ...m, rating: msg.rating } : m)));
            setError(e instanceof Error ? e.message : 'Could not save your rating');
        }
    };

    useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    };

    const displayMessages: Message[] = messages.length === 0 && !loadingChat
        ? [{ id: 'greeting', role: 'assistant', content: GREETING, timestamp: new Date() }] : messages;

    return (
        <div className="chat-page">
            <div className="chat-sidebar">
                <button
                    type="button"
                    className="btn btn-primary chat-new-btn"
                    onClick={handleNewChat}
                    disabled={sending || loadingChat}
                >
                    {sending && !currentId ? <Loader2 size={16} className="animate-spin" /> : <Plus size={16} />}
                    {' '}New Chat
                </button>
                <div className="chat-history">
                    <p className="chat-history-label">Recent Conversations</p>
                    <div className="chat-history-filters">
                        <label>Topic<input type="search" className="input" value={historyTopic} maxLength={128}
                            onChange={(e) => { setHistoryTopic(e.target.value); setHistoryOffset(0); }} /></label>
                        <label>Date (UTC)<input type="date" className="input" value={historyDate}
                            onChange={(e) => { setHistoryDate(e.target.value); setHistoryOffset(0); }} /></label>
                    </div>
                    {loadingList ? (
                        <div className="chat-history-loading"><Loader2 size={14} className="animate-spin" /> Loading...</div>
                    ) : conversations.length === 0 ? (
                        <EmptyState
                            icon={<Sparkles size={28} />}
                            title={historyTopic || historyDate ? 'No conversations match' : 'No conversations yet'}
                            description={historyTopic || historyDate ? 'Try another topic or date.' : 'Start a new chat to get help from Saarthi.'}
                        />
                    ) : (
                        conversations.map((c) => (
                            <div
                                key={c.id}
                                className={`chat-history-item-wrap ${currentId === c.id ? 'active' : ''}`}
                            >
                                {editingId === c.id ? (
                                    <div className="chat-history-item">
                                        <input
                                            ref={renameInputRef}
                                            className="chat-history-rename-input"
                                            aria-label="Conversation title"
                                            value={editTitle}
                                            onChange={(e) => setEditTitle(e.target.value)}
                                            onKeyDown={(e) => {
                                                e.stopPropagation();
                                                if (e.key === 'Enter') submitRename(c.id);
                                                if (e.key === 'Escape') cancelRename();
                                            }}
                                            onClick={(e) => e.stopPropagation()}
                                        />
                                    </div>
                                ) : (
                                    <button type="button" className="chat-history-item" disabled={sending || loadingChat}
                                        onClick={() => void loadConversation(c.id)}>
                                            <Sparkles size={14} className="chat-history-icon" />
                                            <span className="chat-history-title">{c.title || 'New Chat'}</span>
                                    </button>
                                )}
                                {editingId === c.id ? (
                                    <div className="chat-history-actions">
                                        <button type="button" className="chat-history-action" onClick={() => submitRename(c.id)} aria-label="Save"><Check size={14} /></button>
                                        <button type="button" className="chat-history-action" onClick={cancelRename} aria-label="Cancel"><X size={14} /></button>
                                    </div>
                                ) : (
                                    <div className="chat-history-actions">
                                        <button type="button" className="chat-history-action" onClick={(e) => startRename(e, c)} aria-label="Rename"><Pencil size={14} /></button>
                                        <button type="button" className="chat-history-action chat-history-action-delete" onClick={(e) => handleDeleteClick(e, c.id)} aria-label="Delete"><Trash2 size={14} /></button>
                                    </div>
                                )}
                            </div>
                        ))
                    )}
                </div>
                {historyTotal > HISTORY_PAGE_SIZE && <Pagination total={historyTotal} limit={HISTORY_PAGE_SIZE}
                    offset={historyOffset} onPageChange={setHistoryOffset} className="chat-history-pagination" />}
            </div>

            <div className="chat-main">
                {error && <div className="chat-error" role="alert">{error}</div>}
                {loadingChat && messages.length === 0 ? (
                    <div className="chat-loading-state"><Loader2 size={24} className="animate-spin" /> Loading conversation...</div>
                ) : (
                    <>
                        <div className="chat-messages">
                            {displayMessages.map((msg) => (
                                <div key={msg.id} className={`chat-msg ${msg.role}`}>
                                    <div className={`chat-msg-avatar ${msg.role}`}>
                                        {msg.role === 'user' ? <User size={18} /> : <Bot size={18} />}
                                    </div>
                                    <div className="chat-msg-content">
                                        <div className="chat-msg-header">
                                            <span className="chat-msg-name">{msg.role === 'user' ? 'You' : 'Saarthi AI'}</span>
                                            <span className="chat-msg-time">{msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                                            {msg.topic && <span className="chat-turn-badge">{msg.topic}</span>}
                                            {msg.answerMetadata?.socratic === 'hint' && <span className="chat-turn-badge">Socratic hint</span>}
                                            {msg.role === 'assistant' && isSaved(msg) && msg.answerMetadata?.confidence && (
                                                <span className={`chat-turn-badge ${msg.answerMetadata?.confidence === 'low' ? 'low' : ''}`}
                                                    title={msg.answerMetadata?.confidence_reasons?.join('. ')}>
                                                    {msg.answerMetadata.confidence === 'ok' ? 'Confidence · OK' : 'Confidence · Low'}
                                                </span>
                                            )}
                                        </div>
                                        <AnswerSources citations={msg.answerMetadata?.citations} />
                                        {(msg.answerMetadata?.cap?.over || msg.answerMetadata?.cap?.near) && (
                                            <p className="chat-input-hint">{msg.answerMetadata.cap.over ? 'Your daily advanced model quota has been reached.' : 'You are nearing your daily advanced model quota.'}</p>
                                        )}
                                        <div className="chat-msg-text">
                                            {msg.content.split(/```/).map((part, i) => {
                                                if (i % 2 === 1) {
                                                    return (
                                                        <pre key={i} className="chat-code-block">
                                                            <code>{part}</code>
                                                        </pre>
                                                    );
                                                }
                                                if (msg.role === 'assistant') {
                                                    // Equations, lists, bold and [Source: …] badges from the knowledge-base answers.
                                                    return <div key={i} className="chat-msg-rich">{renderAnswer(part, 'chat-msg-bullet')}</div>;
                                                }
                                                return (
                                                    <div key={i}>
                                                        {part.split('\n').map((line, k) => (
                                                            <p key={k} className="chat-msg-p">
                                                                {line.split(/(\[Source: .*?\]|\*\*.*?\*\*)/g).map((subPart, j) => {
                                                                    if (subPart.startsWith('[Source:')) {
                                                                        const source = subPart.match(/\[Source: (.*?)\]/)?.[1];
                                                                        return (
                                                                            <span key={j} className="badge badge-success badge-sm chat-source">
                                                                                📚 {source}
                                                                            </span>
                                                                        );
                                                                    }
                                                                    if (subPart.startsWith('**') && subPart.endsWith('**')) {
                                                                        return <strong key={j}>{subPart.slice(2, -2)}</strong>;
                                                                    }
                                                                    return subPart;
                                                                })}
                                                            </p>
                                                        ))}
                                                    </div>
                                                );
                                            })}
                                        </div>
                                        {msg.role === 'assistant' && isSaved(msg) && (
                                            <div className="chat-msg-rating" role="group" aria-label="Was this answer helpful?">
                                                <button type="button" className={msg.rating === 'up' ? 'active' : ''} aria-pressed={msg.rating === 'up'} aria-label="Helpful" title="Helpful" onClick={() => rateAnswer(msg, 'up')}>
                                                    <ThumbsUp size={14} />
                                                </button>
                                                <button type="button" className={msg.rating === 'down' ? 'active down' : ''} aria-pressed={msg.rating === 'down'} aria-label="Not helpful" title="Not helpful" onClick={() => rateAnswer(msg, 'down')}>
                                                    <ThumbsDown size={14} />
                                                </button>
                                                {msg.rating && <span>Thanks for the feedback</span>}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            ))}
                            {sending && !displayMessages.at(-1)?.id.endsWith('-answer') && (
                                <div className="chat-msg assistant">
                                    <div className="chat-msg-avatar assistant"><Bot size={18} /></div>
                                    <div className="chat-msg-content">
                                        <div className="chat-typing">
                                            <Loader2 size={16} className="animate-spin" />
                                            <span role="status">{streamStatus || 'Saarthi is thinking…'}</span>
                                        </div>
                                    </div>
                                </div>
                            )}
                            <div ref={endRef} />
                        </div>

                        {displayMessages.length <= 1 && (
                            <div className="chat-suggestions">
                                {suggestedTopics.map((topic, i) => {
                                    const Icon = topic.icon;
                                    return (
                                        <button key={i} type="button" className="chat-suggestion-btn" onClick={() => setInput(topic.label)}>
                                            <Icon size={16} /> {topic.label}
                                        </button>
                                    );
                                })}
                            </div>
                        )}

                        <div className="chat-input-area">
                            <div className="chat-turn-controls">
                                <label>Course<select className="input" value={courseId} disabled={loadingChat || sending}
                                    onChange={(e) => changeCourse(e.target.value)}>
                                    <option value="">General chat</option>
                                    {courseId && !courses.some((item) => item.courseId === courseId) && <option value={courseId}>Selected course</option>}
                                    {courses.map((item) => <option key={item.courseId} value={item.courseId}>{item.course.title}</option>)}
                                </select></label>
                                {Boolean(modules.length) && <label>Topic<select className="input" value={moduleId} disabled={sending}
                                    onChange={(e) => setModuleId(e.target.value)}>
                                    <option value="">All topics</option>
                                    {modules.map((item) => <option key={item.id} value={item.id}>{item.title}</option>)}
                                </select></label>}
                                <label>Answer style<select className="input" value={answerMode} disabled={sending}
                                    onChange={(e) => setAnswerMode(e.target.value as typeof answerMode)}>
                                    <option value="auto">Automatic</option>
                                    <option value="socratic">Hints first</option>
                                    <option value="direct">Direct answer</option>
                                </select></label>
                                {contextMaterialId && <button type="button" className="btn btn-outline btn-sm" disabled={sending}
                                    onClick={() => { const next = new URLSearchParams(searchParams); next.delete('material'); setSearchParams(next, { replace: true }); }}>
                                    <X size={14} /> Clear document context
                                </button>}
                            </div>
                            {enrolledCourses?.length === 0 && !courseId && <p className="chat-input-hint">
                                <Link to="/courses">Join a course</Link> to ask questions about its materials.
                            </p>}
                            {courseListError && <p className="chat-input-hint" role="status">Could not load your courses. Reload to try again.</p>}
                            <div className="chat-input-wrapper">
                                <textarea
                                    ref={inputRef}
                                    value={input}
                                    onChange={(e) => setInput(e.target.value)}
                                    onKeyDown={handleKeyDown}
                                    placeholder="Ask Saarthi anything about your subjects..."
                                    aria-label="Message to Saarthi"
                                    rows={1}
                                    disabled={sending}
                                />
                                <button type="button" className="chat-send-btn" aria-label="Send message" onClick={sendMessage} disabled={!input.trim() || sending}>
                                    <Send size={18} />
                                </button>
                            </div>
                            <p className="chat-input-hint">Press Enter to send • Shift+Enter for new line</p>
                        </div>
                    </>
                )}
            </div>

            <ConfirmModal
                open={deleteConfirmId !== null}
                title="Delete conversation"
                message="This conversation will be permanently deleted. This cannot be undone."
                confirmLabel="Delete"
                variant="danger"
                onConfirm={handleDeleteConfirm}
                onCancel={() => setDeleteConfirmId(null)}
            />
        </div>
    );
}
