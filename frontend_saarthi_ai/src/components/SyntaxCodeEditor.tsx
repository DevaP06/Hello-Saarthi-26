import CodeMirror from '@uiw/react-codemirror';
import { javascript } from '@codemirror/lang-javascript';
import { python } from '@codemirror/lang-python';
import { StreamLanguage } from '@codemirror/language';
import { c, cpp, java } from '@codemirror/legacy-modes/mode/clike';
import { go } from '@codemirror/legacy-modes/mode/go';
import { julia } from '@codemirror/legacy-modes/mode/julia';
import { octave } from '@codemirror/legacy-modes/mode/octave';
import { rust } from '@codemirror/legacy-modes/mode/rust';
import { EditorView } from '@codemirror/view';

const editorAttributes = EditorView.contentAttributes.of({ 'aria-label': 'Code editor', spellcheck: 'false' });
const cPlusPlus = StreamLanguage.define(cpp);
const modes = {
    python: python(), javascript: javascript(), matlab: StreamLanguage.define(octave),
    c: StreamLanguage.define(c), cpp: cPlusPlus, 'c++': cPlusPlus, java: StreamLanguage.define(java),
    go: StreamLanguage.define(go), julia: StreamLanguage.define(julia), rust: StreamLanguage.define(rust),
};

export default function SyntaxCodeEditor({ code, language, onChange }: {
    code: string;
    language: string;
    onChange: (value: string) => void;
}) {
    const syntax = modes[language as keyof typeof modes];
    return <CodeMirror className="codelab-syntax-editor" value={code} height="100%" theme="dark"
        extensions={syntax ? [editorAttributes, syntax] : [editorAttributes]} onChange={onChange} indentWithTab={false} />;
}
