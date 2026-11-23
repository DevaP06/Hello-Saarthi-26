import { useEffect, useRef, useState } from 'react';
import { ChevronLeft, ChevronRight, Highlighter, Trash2 } from 'lucide-react';
import { getDocument, GlobalWorkerOptions, TextLayer, type PDFDocumentProxy, type RenderTask } from 'pdfjs-dist';
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
import 'pdfjs-dist/web/pdf_viewer.css';
import { api } from '../lib/api';
import './MaterialPdf.css';

GlobalWorkerOptions.workerSrc = workerUrl;

interface HighlightRectangle {
    x: number;
    y: number;
    width: number;
    height: number;
}

interface SelectedText {
    page: number;
    quote: string;
    rectangles: HighlightRectangle[];
}

interface Annotation extends SelectedText {
    id: string;
    documentVersion: string;
    note: string;
}

function readSelection(layer: HTMLElement, bounds: DOMRect, page: number): SelectedText | null {
    const selection = window.getSelection();
    if (!selection?.rangeCount || selection.isCollapsed || !bounds.width || !bounds.height) return null;
    const range = selection.getRangeAt(0);
    if (!layer.contains(range.startContainer) || !layer.contains(range.endContainer)) return null;
    const quote = selection.toString().trim();
    if (!quote) return null;
    const rectangles = Array.from(range.getClientRects()).flatMap((rect) => {
        const left = Math.max(rect.left, bounds.left);
        const top = Math.max(rect.top, bounds.top);
        const width = Math.min(rect.right, bounds.right) - left;
        const height = Math.min(rect.bottom, bounds.bottom) - top;
        return width > 0 && height > 0 ? [{
            x: (left - bounds.left) / bounds.width,
            y: (top - bounds.top) / bounds.height,
            width: width / bounds.width,
            height: height / bounds.height,
        }] : [];
    });
    return rectangles.length ? { page, quote, rectangles } : null;
}

export default function MaterialPdf({ src, title, initialPage, annotationPath, documentVersion }: {
    src: string;
    title: string;
    initialPage: number;
    annotationPath: string;
    documentVersion: string | null;
}) {
    const [pdf, setPdf] = useState<PDFDocumentProxy | null>(null);
    const [requestedPage, setRequestedPage] = useState(initialPage);
    const [documentError, setDocumentError] = useState(false);
    const [pageActivity, setPageActivity] = useState<{ key: string; hasText?: boolean; error?: boolean } | null>(null);
    const [viewportWidth, setViewportWidth] = useState(0);
    const [annotations, setAnnotations] = useState<Annotation[] | null>(null);
    const [selection, setSelection] = useState<SelectedText | null>(null);
    const [note, setNote] = useState('');
    const [activeHighlight, setActiveHighlight] = useState<string | null>(null);
    const [busy, setBusy] = useState(false);
    const [feedback, setFeedback] = useState('');
    const scroller = useRef<HTMLDivElement>(null);
    const pageBox = useRef<HTMLDivElement>(null);
    const canvas = useRef<HTMLCanvasElement>(null);
    const textLayer = useRef<HTMLDivElement>(null);
    const page = pdf ? Math.min(requestedPage, pdf.numPages) : requestedPage;
    const pageKey = `${page}:${viewportWidth}`;
    const pageReady = pageActivity?.key === pageKey && !pageActivity.error;

    useEffect(() => {
        const task = getDocument({ url: src });
        let disposed = false;
        void task.promise.then((loaded) => { if (!disposed) setPdf(loaded); })
            .catch(() => { if (!disposed) setDocumentError(true); });
        return () => {
            disposed = true;
            void task.destroy().catch(() => {});
        };
    }, [src]);

    useEffect(() => {
        if (!documentVersion) return;
        let disposed = false;
        void api.get<Annotation[]>(annotationPath).then((items) => {
            if (!disposed) setAnnotations(items.filter((item) => item.documentVersion === documentVersion));
        }).catch(() => { if (!disposed) setFeedback('Could not load saved highlights. Reopen the PDF to retry.'); });
        return () => { disposed = true; };
    }, [annotationPath, documentVersion]);

    useEffect(() => {
        const element = scroller.current;
        if (!element || documentError) return;
        const observer = new ResizeObserver(([entry]) => setViewportWidth(Math.floor(entry.contentRect.width)));
        observer.observe(element);
        return () => observer.disconnect();
    }, [documentError]);

    useEffect(() => {
        const box = pageBox.current;
        const surface = canvas.current;
        const layer = textLayer.current;
        if (!pdf || !viewportWidth || !box || !surface || !layer || documentError) return;
        let disposed = false;
        let renderTask: RenderTask | undefined;
        let layerTask: TextLayer | undefined;
        void pdf.getPage(page).then(async (pdfPage) => {
            if (disposed) return;
            const base = pdfPage.getViewport({ scale: 1 });
            const scale = Math.max(1, Math.min(960, viewportWidth - 24)) / base.width;
            const viewport = pdfPage.getViewport({ scale });
            const ratio = Math.min(2, window.devicePixelRatio || 1);
            box.style.width = `${Math.floor(viewport.width)}px`;
            box.style.height = `${Math.floor(viewport.height)}px`;
            box.style.setProperty('--total-scale-factor', String(scale * viewport.userUnit));
            surface.width = Math.ceil(viewport.width * ratio);
            surface.height = Math.ceil(viewport.height * ratio);
            layer.replaceChildren();
            renderTask = pdfPage.render({ canvas: surface, viewport, transform: [ratio, 0, 0, ratio, 0, 0] });
            layerTask = new TextLayer({ container: layer, viewport, textContentSource: pdfPage.streamTextContent() });
            await Promise.all([renderTask.promise, layerTask.render()]);
            if (!disposed) setPageActivity({ key: pageKey, hasText: layerTask.textContentItemsStr.some((text) => text.trim()) });
        }).catch(() => { if (!disposed) setPageActivity({ key: pageKey, error: true }); });
        return () => {
            disposed = true;
            renderTask?.cancel();
            layerTask?.cancel();
        };
    }, [pdf, page, pageKey, viewportWidth, documentError]);

    useEffect(() => {
        if (!pageReady || !annotations) return;
        const capture = () => {
            const layer = textLayer.current;
            const box = pageBox.current;
            if (!layer || !box) return;
            const selected = readSelection(layer, box.getBoundingClientRect(), page);
            if (selected) setSelection(selected);
        };
        document.addEventListener('selectionchange', capture);
        return () => document.removeEventListener('selectionchange', capture);
    }, [pageReady, annotations, page]);

    useEffect(() => {
        const element = scroller.current;
        const highlight = pageBox.current?.querySelector('.material-pdf-highlights .active');
        if (!pageReady || !activeHighlight || !element || !highlight) return;
        const offset = highlight.getBoundingClientRect().top - element.getBoundingClientRect().top;
        element.scrollTop += offset - 36;
    }, [activeHighlight, pageReady]);

    function goToPage(next: number, highlightId: string | null = null) {
        setRequestedPage(next);
        setSelection(null);
        setNote('');
        setActiveHighlight(highlightId);
        scroller.current?.scrollTo({ top: 0 });
    }

    async function saveHighlight() {
        if (!selection || !documentVersion || busy) return;
        if (selection.quote.length > 5000 || selection.rectangles.length > 100) {
            setFeedback('Select a shorter passage to highlight.');
            return;
        }
        setBusy(true);
        setFeedback('');
        try {
            const saved = await api.post<Annotation>(annotationPath, { ...selection, documentVersion, note });
            setAnnotations((items) => [...(items ?? []), saved]);
            setSelection(null);
            setNote('');
            setActiveHighlight(saved.id);
            setFeedback('Highlight saved.');
        } catch (error) {
            setFeedback(error instanceof Error ? error.message : 'Could not save the highlight.');
        } finally {
            setBusy(false);
        }
    }

    async function deleteHighlight(id: string) {
        setBusy(true);
        setFeedback('');
        try {
            await api.delete(`${annotationPath}/${id}`);
            setAnnotations((items) => items?.filter((item) => item.id !== id) ?? null);
            setFeedback('Highlight removed.');
        } catch (error) {
            setFeedback(error instanceof Error ? error.message : 'Could not remove the highlight.');
        } finally {
            setBusy(false);
        }
    }

    if (documentError || (pageActivity?.key === pageKey && pageActivity.error)) {
        return <iframe title={title} src={`${src}#page=${page}`} className="cd-doc-view-iframe" />;
    }

    return (
        <section className="material-pdf" aria-label={`PDF viewer for ${title}`}>
            <div className="material-pdf-toolbar">
                <button type="button" className="btn btn-ghost btn-sm" aria-label="Previous page"
                    disabled={!pdf || page <= 1 || busy} onClick={() => goToPage(page - 1)}><ChevronLeft size={16} /></button>
                <span role="status">{pdf ? `Page ${page} of ${pdf.numPages}` : 'Loading PDF…'}</span>
                <button type="button" className="btn btn-ghost btn-sm" aria-label="Next page"
                    disabled={!pdf || page >= pdf.numPages || busy} onClick={() => goToPage(page + 1)}><ChevronRight size={16} /></button>
            </div>
            <div className="material-pdf-scroll" ref={scroller} aria-busy={!pageReady}>
                <div className="material-pdf-page" ref={pageBox} style={{ visibility: pageReady ? 'visible' : 'hidden' }}>
                    <canvas ref={canvas} aria-hidden="true" />
                    <div className="material-pdf-highlights" aria-hidden="true">
                        {annotations?.filter((item) => item.page === page).flatMap(({ id, rectangles }) => rectangles.map((rect, index) => (
                            <span className={id === activeHighlight ? 'active' : undefined} key={`${id}:${index}`} style={{
                                left: `${rect.x * 100}%`, top: `${rect.y * 100}%`, width: `${rect.width * 100}%`, height: `${rect.height * 100}%`,
                            }} />
                        )))}
                    </div>
                    <div className="textLayer" ref={textLayer} />
                </div>
            </div>
            <div className="material-pdf-notes">
                <details open={selection ? true : undefined}>
                    <summary>My highlights{annotations?.length ? ` (${annotations.length})` : ''}</summary>
                    {selection ? (
                        <div className="material-pdf-draft">
                            <blockquote>{selection.quote}</blockquote>
                            <label>Private note (optional)<textarea rows={2} maxLength={4000} value={note}
                                className="input" onChange={(event) => setNote(event.target.value)} /></label>
                            <div className="material-pdf-actions">
                                <button type="button" className="btn btn-primary btn-sm" disabled={busy} onClick={() => void saveHighlight()}>
                                    <Highlighter size={15} /> Save highlight
                                </button>
                                <button type="button" className="btn btn-ghost btn-sm" disabled={busy} onClick={() => { setSelection(null); setNote(''); }}>
                                    Clear selection
                                </button>
                            </div>
                        </div>
                    ) : annotations && pageReady && (
                        <p>{pageActivity.hasText ? 'Select text on the page to save a highlight.' : 'This page is available as an image.'}</p>
                    )}
                    <ul>
                        {annotations?.map((item) => (
                            <li key={item.id}>
                                <button type="button" className="material-pdf-note-link" disabled={busy}
                                    onClick={() => goToPage(item.page, item.id)}><span>Page {item.page}</span>{item.quote}</button>
                                {item.note && <p>{item.note}</p>}
                                <button type="button" className="btn btn-ghost btn-sm" disabled={busy}
                                    aria-label={`Remove highlight on page ${item.page}`} onClick={() => void deleteHighlight(item.id)}><Trash2 size={15} /></button>
                            </li>
                        ))}
                    </ul>
                </details>
                {feedback && <p role="status">{feedback}</p>}
            </div>
        </section>
    );
}
