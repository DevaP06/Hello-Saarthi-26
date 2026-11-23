import { useEffect, useEffectEvent } from 'react';
import { useAuthStore } from '../stores/auth.store';
import { apiUrl, authFetch } from './api';

function eventId(): string {
    // getRandomValues also works when developing over plain HTTP on the local network.
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

export function useResourceActivity(
    resourceType: 'material' | 'video', resourceId: string | undefined, isActive: () => boolean,
) {
    const studentId = useAuthStore((state) => state.user?.role === 'student' ? state.user.id : undefined);
    const mayCountTime = useEffectEvent(isActive);

    useEffect(() => {
        if (!studentId || !resourceId) return;

        const send = async (activeSeconds: number) => {
            const body = JSON.stringify({ eventId: eventId(), resourceType, resourceId: Number(resourceId), activeSeconds });
            // Keep the same ID if the server received the event but the response was lost.
            for (let attempt = 0; attempt < 2; attempt += 1) {
                if (useAuthStore.getState().user?.id !== studentId) return;
                try {
                    const response = await authFetch(apiUrl('/progress/activity'), {
                        method: 'POST', headers: { 'Content-Type': 'application/json' }, body, keepalive: true,
                    });
                    if (response.status < 500) return;
                } catch {
                    // Activity recording should never interrupt the reader or player.
                }
            }
        };

        let lastTick = performance.now();
        let activeMs = 0;
        let wasActive = document.visibilityState === 'visible' && document.hasFocus() && mayCountTime();
        const flush = () => {
            const seconds = Math.min(Math.floor(activeMs / 1000), 30);
            if (!seconds) return;
            activeMs -= seconds * 1000;
            void send(seconds);
        };
        const tick = () => {
            const now = performance.now();
            if (wasActive) activeMs += Math.min(now - lastTick, 1500);
            lastTick = now;
            wasActive = document.visibilityState === 'visible' && document.hasFocus() && mayCountTime();
            if (activeMs >= 15_000) flush();
        };
        const onLeave = () => { tick(); flush(); };
        // Deferring the open avoids a duplicate event during React's development effect replay.
        const open = window.setTimeout(() => void send(0), 0);
        const timer = window.setInterval(tick, 1000);
        document.addEventListener('visibilitychange', onLeave);
        window.addEventListener('blur', onLeave);
        window.addEventListener('focus', tick);
        window.addEventListener('pagehide', onLeave);
        return () => {
            window.clearTimeout(open);
            window.clearInterval(timer);
            document.removeEventListener('visibilitychange', onLeave);
            window.removeEventListener('blur', onLeave);
            window.removeEventListener('focus', tick);
            window.removeEventListener('pagehide', onLeave);
            if (wasActive) activeMs += Math.min(performance.now() - lastTick, 1500);
            flush();
        };
    }, [resourceType, resourceId, studentId]);
}
