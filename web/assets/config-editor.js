/** CodeMirror-based config.yaml editor for settings page */

window.CoolClawConfigEditor = {
    _cm: null,
    _onDirty: null,

    init(containerId, options = {}) {
        const el = document.getElementById(containerId);
        if (!el || typeof CodeMirror === 'undefined') return null;

        if (this._cm) {
            this._cm.toTextArea();
            this._cm = null;
        }

        this._onDirty = options.onDirty || null;
        this._cm = CodeMirror(el, {
            mode: 'yaml',
            theme: 'material-darker',
            lineNumbers: true,
            indentUnit: 2,
            tabSize: 2,
            indentWithTabs: false,
            lineWrapping: false,
            foldGutter: true,
            gutters: ['CodeMirror-linenumbers', 'CodeMirror-foldgutter'],
            matchBrackets: true,
            autoCloseBrackets: true,
            styleActiveLine: true,
            viewportMargin: Infinity,
            extraKeys: {
                Tab(cm) {
                    if (cm.somethingSelected()) {
                        cm.indentSelection('add');
                    } else {
                        cm.replaceSelection('  ', 'end');
                    }
                },
                'Shift-Tab'(cm) {
                    cm.indentSelection('subtract');
                },
                'Ctrl-S'() {
                    window.dispatchEvent(new CustomEvent('config-yaml-save', { detail: { reload: true } }));
                    return false;
                },
                'Cmd-S'() {
                    window.dispatchEvent(new CustomEvent('config-yaml-save', { detail: { reload: true } }));
                    return false;
                },
            },
        });

        this._cm.on('change', () => {
            if (this._onDirty) this._onDirty(true);
        });

        return this._cm;
    },

    getValue() {
        return this._cm ? this._cm.getValue() : '';
    },

    setValue(text) {
        if (this._cm) {
            this._cm.setValue(text || '');
            this._cm.clearHistory();
            if (this._onDirty) this._onDirty(false);
        }
    },

    focus() {
        if (this._cm) this._cm.focus();
    },

    refresh() {
        if (this._cm) this._cm.refresh();
    },
};
