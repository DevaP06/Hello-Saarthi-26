import { Link } from 'react-router-dom';
import type { ChatCitation } from '../lib/api';
import { resourceLink } from '../lib/resourceLinks';

export default function AnswerSources({ citations }: { citations: ChatCitation[] | undefined }) {
    if (!citations?.length) return null;
    return (
        <ul className="chat-citation-list" aria-label="Answer sources">
            {citations.map((source, index) => {
                const link = resourceLink({ courseId: source.course_id, materialId: source.material_id,
                    videoId: source.video_id, startSeconds: source.start_sec, page: source.page_number });
                const name = source.source_file.split(/[/\\]/).pop() || 'Course source';
                const timestamp = source.start_sec != null
                    ? `${Math.floor(source.start_sec / 60)}:${String(source.start_sec % 60).padStart(2, '0')}` : null;
                const text = `${name}${timestamp ? ` · ${timestamp}` : source.page_number ? ` · p. ${source.page_number}` : ''}`;
                return <li key={index}>{link ? <Link to={link} target="_blank" rel="noopener noreferrer">{text}</Link> : text}</li>;
            })}
        </ul>
    );
}
