import { useState } from 'react';
import type { ModuleResponse } from '../lib/api';

export default function ModulePrerequisites({ module, modules, onSave, onCancel }: {
    module: ModuleResponse;
    modules: ModuleResponse[];
    onSave: (ids: string[]) => Promise<void>;
    onCancel: () => void;
}) {
    const [selected, setSelected] = useState(module.prerequisiteIds ?? []);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState('');
    const choices = modules.filter((item) => item.id !== module.id);

    async function save() {
        setBusy(true);
        setError('');
        try {
            await onSave(selected);
        } catch (error) {
            setError(error instanceof Error ? error.message : 'Could not save prerequisites.');
        } finally {
            setBusy(false);
        }
    }

    return (
        <form className="cd-prerequisite-editor" onSubmit={(event) => { event.preventDefault(); void save(); }}>
            <fieldset disabled={busy}>
                <legend>Prerequisites for {module.title}</legend>
                {choices.length === 0 && <p>Add another topic to set a prerequisite.</p>}
                {choices.map((item) => (
                    <label key={item.id}>
                        <input type="checkbox" checked={selected.includes(item.id)} onChange={(event) => {
                            const checked = event.currentTarget.checked;
                            setSelected((ids) => checked ? [...ids, item.id] : ids.filter((id) => id !== item.id));
                        }} />
                        {item.title}
                    </label>
                ))}
            </fieldset>
            {error && <p role="alert">{error}</p>}
            <div className="cd-topic-actions">
                <button type="submit" className="btn btn-primary btn-sm" disabled={busy || choices.length === 0}>Save prerequisites</button>
                <button type="button" className="btn btn-ghost btn-sm" disabled={busy} onClick={onCancel}>Cancel</button>
            </div>
        </form>
    );
}
