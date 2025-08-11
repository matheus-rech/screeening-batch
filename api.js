// api.js - Frontend API integration

class ScreeningAPI {
    constructor(baseURL = 'http://localhost:8000') {
        this.baseURL = baseURL;
        this.sessionId = null;
        this.ws = null;
    }

    async importReferences(references, criteria) {
        const response = await fetch(`${this.baseURL}/api/import`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ references, criteria })
        });
        
        const data = await response.json();
        this.sessionId = data.session_id;
        this.connectWebSocket();
        return data;
    }

    async screenSingle(reference, criteria, useAI = true) {
        const response = await fetch(`${this.baseURL}/api/screen/single`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                reference,
                criteria,
                use_ai: useAI
            })
        });
        
        return await response.json();
    }

    async getNextReference() {
        if (!this.sessionId) throw new Error('No active session');
        
        const response = await fetch(`${this.baseURL}/api/screen/next/${this.sessionId}`);
        return await response.json();
    }

    async resolveConflict(result1, result2, paper, criteria) {
        const response = await fetch(`${this.baseURL}/api/conflict/resolve`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ result1, result2, paper, criteria })
        });
        
        return await response.json();
    }

    async exportResults(format = 'csv') {
        if (!this.sessionId) throw new Error('No active session');
        
        const response = await fetch(`${this.baseURL}/api/export/${this.sessionId}?format=${format}`);
        const data = await response.json();
        
        // Trigger download
        const blob = new Blob([data.content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = data.filename;
        a.click();
    }

    connectWebSocket() {
        if (!this.sessionId) return;
        
        this.ws = new WebSocket(`ws://localhost:8000/ws/${this.sessionId}`);
        
        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            if (message.type === 'progress') {
                this.onProgressUpdate(message.data);
            }
        };
    }

    onProgressUpdate(progress) {
        // Override this method to handle progress updates
        console.log('Progress:', progress);
    }
}
