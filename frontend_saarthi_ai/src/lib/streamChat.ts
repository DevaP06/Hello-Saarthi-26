import { apiUrl, authFetch, type SendMessageResponse } from './api';

export async function streamChat(
    turn: {
        conversationId: string; message: string; courseId?: number; moduleId?: number;
        contextMaterialId?: number; answerMode: 'auto' | 'socratic' | 'direct';
    },
    onDelta: (delta: string) => void,
    onStatus: (detail: string) => void,
): Promise<SendMessageResponse> {
    // Quick Tunnels do not support Server-Sent Events.
    const jsonDelivery = window.location.hostname.endsWith('.trycloudflare.com');
    const response = await authFetch(apiUrl('/chat/stream'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: jsonDelivery ? 'application/json' : 'text/event-stream' },
        body: JSON.stringify({ ...turn, structured: true }),
    });
    if (!response.ok) {
        const error = await response.json().catch(() => null);
        const detail = typeof error?.detail === 'string' ? error.detail : error?.detail?.[0]?.msg;
        const fallback = response.status >= 500 ? 'Saarthi could not complete this request. Please try again.'
            : `Could not start this chat turn (${response.status}).`;
        throw new Error(error?.error?.message ?? detail ?? fallback);
    }
    if (jsonDelivery) return await response.json() as SendMessageResponse;
    if (!response.body) throw new Error('No chat response was received.');

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let answer: SendMessageResponse | undefined;
    try {
        for (;;) {
            const { done, value } = await reader.read();
            buffer += decoder.decode(value, { stream: !done });
            let boundary = buffer.indexOf('\n\n');
            while (boundary >= 0) {
                const frame = buffer.slice(0, boundary);
                buffer = buffer.slice(boundary + 2);
                boundary = buffer.indexOf('\n\n');
                const [header, data] = frame.split('\n', 2);
                if (!data?.startsWith('data: ')) continue;
                const payload = data.slice(6);
                if (header === 'event: delta') onDelta(JSON.parse(payload) as string);
                if (header === 'event: status') onStatus((JSON.parse(payload) as { detail: string }).detail);
                if (header === 'event: saved') answer = JSON.parse(payload) as SendMessageResponse;
                if (header === 'event: error') throw new Error(payload);
            }
            if (done) break;
        }
    } catch (error) {
        if (!answer) throw error;
    } finally {
        reader.releaseLock();
    }
    if (!answer) throw new Error('The connection closed before the answer was saved.');
    return answer;
}
