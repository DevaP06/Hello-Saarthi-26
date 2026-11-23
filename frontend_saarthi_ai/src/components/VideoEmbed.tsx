import { useEffect, useEffectEvent, useRef } from 'react';

interface YouTubePlayer {
    getCurrentTime(): number;
    destroy(): void;
}

interface YouTubeApi {
    Player: new (frame: HTMLIFrameElement, options: {
        events: { onStateChange: (event: { data: number; target: YouTubePlayer }) => void };
    }) => YouTubePlayer;
}

declare global {
    interface Window {
        YT?: YouTubeApi;
        onYouTubeIframeAPIReady?: () => void;
    }
}

let apiLoading: Promise<YouTubeApi | undefined> | undefined;

function loadYouTubeApi() {
    if (window.YT?.Player) return Promise.resolve(window.YT);
    if (apiLoading) return apiLoading;

    apiLoading = new Promise<YouTubeApi | undefined>((resolve) => {
        const script = document.createElement('script');
        const previous = window.onYouTubeIframeAPIReady;
        window.onYouTubeIframeAPIReady = () => {
            resolve(window.YT);
            previous?.();
        };
        script.src = 'https://www.youtube.com/iframe_api';
        script.onerror = () => resolve(undefined);
        document.head.append(script);
    });
    return apiLoading;
}

export default function VideoEmbed({ src, title, startSeconds, onPlayback }: {
    src: string;
    title: string;
    startSeconds: number | null;
    onPlayback: (playing: boolean, completedAt?: number) => void;
}) {
    const container = useRef<HTMLDivElement>(null);
    const playback = useEffectEvent(onPlayback);

    useEffect(() => {
        const parent = container.current;
        if (!parent) return;
        let url: URL;
        try {
            url = new URL(src, window.location.href);
        } catch {
            return;
        }
        if (url.protocol !== 'http:' && url.protocol !== 'https:') return;
        const isYouTube = ['www.youtube.com', 'youtube.com', 'www.youtube-nocookie.com'].includes(url.hostname)
            && url.pathname.startsWith('/embed/');
        if (isYouTube) {
            url.searchParams.set('enablejsapi', '1');
            url.searchParams.set('origin', window.location.origin);
            if (startSeconds !== null) url.searchParams.set('start', String(startSeconds));
        }

        // The player API owns the frame and may remove it when destroyed.
        const frame = document.createElement('iframe');
        frame.src = url.href;
        frame.title = title;
        frame.className = 'video-element video-iframe';
        frame.allow = 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture';
        frame.allowFullscreen = true;
        parent.replaceChildren(frame);

        let cancelled = false;
        let player: YouTubePlayer | undefined;
        if (isYouTube) {
            void loadYouTubeApi().then((api) => {
                if (cancelled || !api?.Player) return;
                player = new api.Player(frame, {
                    events: { onStateChange: ({ data, target }) => {
                        if (!cancelled) playback(data === 1, data === 0 ? target.getCurrentTime() : undefined);
                    } },
                });
            }).catch(() => { /* The iframe can still be used without playback metrics. */ });
        }

        return () => {
            cancelled = true;
            playback(false);
            player?.destroy();
            parent.replaceChildren();
        };
    }, [src, title, startSeconds]);

    return <div className="video-element video-iframe" ref={container} />;
}
