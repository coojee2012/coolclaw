/** CoolClaw shared web utilities */

window.CoolClaw = {
    tailwindConfig: {
        darkMode: 'class',
        theme: {
            extend: {
                colors: {
                    primary: '#00d9ff',
                    secondary: '#0f3460',
                    dark: '#1a1a2e',
                    surface: '#16213e',
                    card: '#0d1b2a',
                }
            }
        }
    },

    toast(message, type = 'info') {
        window.dispatchEvent(new CustomEvent('toast', { detail: { message, type } }));
    },

    formatContent(content) {
        if (!content) return '';
        let html = content
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
        html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) =>
            `<pre><code class="language-${lang || 'text'}">${code.trim()}</code></pre>`
        );
        html = html.replace(/`([^`\n]+)`/g, '<code>$1</code>');
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\n/g, '<br>');
        return html;
    },

    formatRelativeTime(iso) {
        if (!iso) return '';
        const diff = Date.now() - new Date(iso).getTime();
        const mins = Math.floor(diff / 60000);
        if (mins < 1) return '刚刚';
        if (mins < 60) return `${mins} 分钟前`;
        const hours = Math.floor(mins / 60);
        if (hours < 24) return `${hours} 小时前`;
        const days = Math.floor(hours / 24);
        if (days < 7) return `${days} 天前`;
        return new Date(iso).toLocaleDateString('zh-CN');
    },

    truncatePath(path, max = 42) {
        if (!path) return '未设置工作目录';
        if (path.length <= max) return path;
        const parts = path.split('/');
        if (parts.length <= 2) return '…' + path.slice(-max + 1);
        return parts[0] + '/…/' + parts.slice(-2).join('/');
    },

    async checkAuth() {
        const res = await fetch('/api/auth/me');
        if (!res.ok) {
            window.location.href = '/login.html';
            return null;
        }
        return res.json();
    },

    async logout() {
        await fetch('/api/auth/logout', { method: 'POST' });
        window.location.href = '/login.html';
    },

    async parseSSEStream(response, onEvent) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const parts = buffer.split('\n\n');
            buffer = parts.pop() || '';

            for (const part of parts) {
                if (!part.trim()) continue;
                let event = 'message';
                let data = '';
                for (const line of part.split('\n')) {
                    if (line.startsWith('event:')) event = line.slice(6).trim();
                    else if (line.startsWith('data:')) {
                        data += (data ? '\n' : '') + line.slice(5);
                    }
                }
                if (data || event !== 'message') {
                    await onEvent(event, data);
                }
            }
        }
    },

    formatSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / 1048576).toFixed(1) + ' MB';
    },
};
