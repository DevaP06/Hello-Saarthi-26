import { useEffect, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { BookOpen, FileText, Play, ExternalLink } from 'lucide-react';
import { api, type SearchResponse } from '../lib/api';
import { resourceLink } from '../lib/resourceLinks';
import Pagination from '../components/Pagination';
import './Search.css';

interface SemanticHit {
    content_type: string;
    source_id: string;
    text: string;
    metadata: {
        title?: string; source_file?: string; page?: number; start_sec?: number;
        course_id?: number; material_id?: number; video_id?: number;
    };
}

const CONTENT_TYPES: Record<string, string> = { notes: 'Notes', video: 'Videos', exercise: 'Solved exercises', code: 'Code' };

function ContentResults({ query, type }: { query: string; type: string }) {
    const [result, setResult] = useState<{ items?: SemanticHit[]; error?: string } | null>(null);
    useEffect(() => {
        let cancelled = false;
        void api.post<{ results: SemanticHit[] }>('/v1/saarthi/retrieval/search', {
            query, include_courses: true, top_k: 20, include_scores: false, content_types: type ? [type] : undefined,
        }).then((response) => { if (!cancelled) setResult({ items: response.results }); })
            .catch((failure) => { if (!cancelled) setResult({ error: failure instanceof Error ? failure.message : 'Could not search course content.' }); });
        return () => { cancelled = true; };
    }, [query, type]);

    if (!result) return <p className="search-loading" role="status">Searching indexed content…</p>;
    if (result.error) return <p className="search-error" role="alert">{result.error}</p>;
    if (!result.items?.length) return <p className="search-empty">No indexed content matches this search.</p>;

    return <ul className="search-list">
        {result.items.map((hit, index) => {
            const source = hit.metadata;
            const title = source.title || source.source_file?.split(/[/\\]/).pop() || 'Course source';
            const href = resourceLink({ courseId: source.course_id, materialId: source.material_id,
                videoId: source.video_id, page: source.page, startSeconds: source.start_sec });
            const timestamp = source.start_sec == null ? '' : ` · ${Math.floor(source.start_sec / 60)}:${String(source.start_sec % 60).padStart(2, '0')}`;
            return <li key={`${hit.source_id}:${index}`} className="search-content-result">
                <h2>{href ? <Link to={href}>{title} <ExternalLink size={14} aria-hidden="true" /></Link> : title}</h2>
                <p className="search-item-sub">{CONTENT_TYPES[hit.content_type] ?? 'Content'}{source.page ? ` · p. ${source.page}` : ''}{timestamp}</p>
                <p>{hit.text.length > 440 ? `${hit.text.slice(0, 440).trimEnd()}…` : hit.text}</p>
            </li>;
        })}
    </ul>;
}

export default function SearchPage() {
    const [searchParams, setSearchParams] = useSearchParams();
    const q = (searchParams.get('q') ?? '').trim();
    const offset = Math.max(0, parseInt(searchParams.get('offset') ?? '0', 10) || 0);
    const limit = 20;
    const contentMode = searchParams.get('mode') === 'content';
    const contentType = searchParams.get('type') ?? '';

    const [result, setResult] = useState<{ q: string; offset: number; data?: SearchResponse; error?: string } | null>(null);
    const current = result?.q === q && result.offset === offset ? result : null;
    const data = current?.data;
    const error = current?.error;

    useEffect(() => {
        if (q.length < 2 || contentMode) return;
        let cancelled = false;
        api.get<SearchResponse>('/courses/search', { q, limit, offset })
            .then((res) => {
                if (!cancelled) setResult({ q, offset, data: res });
            })
            .catch((e) => {
                if (!cancelled) setResult({ q, offset, error: e instanceof Error ? e.message : 'Search failed' });
            });
        return () => { cancelled = true; };
    }, [q, offset, contentMode]);

    const setMode = (mode: 'titles' | 'content') => setSearchParams((previous) => {
        const next = new URLSearchParams(previous);
        next.set('mode', mode);
        next.delete('offset');
        return next;
    });

    return (
        <div className="search-page">
            <h1>Search</h1>
            <form key={q} className="search-controls" onSubmit={(event) => {
                event.preventDefault();
                const query = String(new FormData(event.currentTarget).get('query') ?? '').trim();
                setSearchParams((previous) => {
                    const next = new URLSearchParams(previous);
                    next.set('q', query);
                    next.delete('offset');
                    return next;
                });
            }}>
                <input name="query" type="search" className="input" defaultValue={q} maxLength={2000}
                    aria-label="Search query" placeholder="Enter a topic or title" />
                <button type="submit" className="btn btn-primary btn-sm">Search</button>
            </form>
            <div className="search-controls" role="group" aria-label="Search mode">
                <button type="button" className={`btn btn-sm ${contentMode ? 'btn-outline' : 'btn-primary'}`} aria-pressed={!contentMode}
                    onClick={() => setMode('titles')}>Titles</button>
                <button type="button" className={`btn btn-sm ${contentMode ? 'btn-primary' : 'btn-outline'}`} aria-pressed={contentMode}
                    onClick={() => setMode('content')}>Content</button>
                {contentMode && <label>Content type <select className="input" value={contentType} onChange={(event) => {
                    setSearchParams((previous) => {
                        const next = new URLSearchParams(previous);
                        next.set('type', event.target.value);
                        return next;
                    });
                }}>
                    <option value="">All types</option>
                    {Object.entries(CONTENT_TYPES).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
                </select></label>}
            </div>
            {contentMode ? (q.length >= 2 ? <ContentResults key={`${q}:${contentType}`} query={q} type={contentType} />
                : <p className="search-placeholder">Enter at least 2 characters to search indexed course content.</p>) : <>
            {error && <div className="search-error" role="alert">{error}</div>}
            {q.length < 2 ? (
                <p className="search-placeholder">Enter at least 2 characters to search.</p>
            ) : !current ? (
                <p className="search-loading">Searching…</p>
            ) : data ? (
                <>
                    <div className="search-results">
                        {data.courses.length > 0 && (
                            <section className="search-section">
                                <h2><BookOpen size={18} /> Courses ({data.totalCourses})</h2>
                                <ul className="search-list">
                                    {data.courses.map((item) => (
                                        <li key={`c-${item.id}`}>
                                            <a href={item.link ?? `/courses/${item.id}`} className="search-item">
                                                <span className="search-item-title">{item.title}</span>
                                                {item.subtitle && <span className="search-item-sub">{item.subtitle}</span>}
                                                <ExternalLink size={14} />
                                            </a>
                                        </li>
                                    ))}
                                </ul>
                            </section>
                        )}
                        {data.materials.length > 0 && (
                            <section className="search-section">
                                <h2><FileText size={18} /> Materials ({data.totalMaterials})</h2>
                                <ul className="search-list">
                                    {data.materials.map((item) => {
                                        const link = resourceLink({ courseId: item.courseId, materialId: item.id });
                                        const label = <>
                                                <span className="search-item-title">{item.title}</span>
                                                {item.subtitle && <span className="search-item-sub">{item.subtitle}</span>}
                                                <ExternalLink size={14} />
                                        </>;
                                        return (
                                            <li key={`m-${item.id}`}>
                                                {link ? <Link to={link} className="search-item">{label}</Link>
                                                    : <a href={item.link ?? '#'} className="search-item">{label}</a>}
                                            </li>
                                        );
                                    })}
                                </ul>
                            </section>
                        )}
                        {data.videos.length > 0 && (
                            <section className="search-section">
                                <h2><Play size={18} /> Videos ({data.totalVideos})</h2>
                                <ul className="search-list">
                                    {data.videos.map((item) => (
                                        <li key={`v-${item.id}`}>
                                            <a href={item.link ?? `/videos/${item.id}`} className="search-item">
                                                <span className="search-item-title">{item.title}</span>
                                                {item.subtitle && <span className="search-item-sub">{item.subtitle}</span>}
                                                <ExternalLink size={14} />
                                            </a>
                                        </li>
                                    ))}
                                </ul>
                            </section>
                        )}
                    </div>
                    {(data.totalCourses > limit || data.totalMaterials > limit || data.totalVideos > limit) && (
                        <Pagination
                            total={Math.max(data.totalCourses, data.totalMaterials, data.totalVideos)}
                            limit={data.limit}
                            offset={data.offset}
                            onPageChange={(newOffset) => setSearchParams((prev) => {
                                prev.set('offset', String(newOffset));
                                return prev;
                            })}
                        />
                    )}
                    {data.courses.length === 0 && data.materials.length === 0 && data.videos.length === 0 && (
                        <p className="search-empty">No results for &quot;{q}&quot;</p>
                    )}
                </>
            ) : null}
            </>}
        </div>
    );
}
