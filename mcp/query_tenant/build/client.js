import { signRequest, } from './auth.js';
const DEFAULT_TIMEOUT_MS = 10_000;
const MAX_RETRIES = 3;
const CIRCUIT_THRESHOLD = 5;
const CIRCUIT_RESET_MS = 30_000;
export class AuthenticatedClient {
    baseUrl;
    credentials;
    circuit = { failures: 0, lastFailure: 0, isOpen: false };
    constructor(baseUrl, credentials, allowHttp = false) {
        if (!baseUrl.startsWith('https://') && !allowHttp) {
            throw new Error(`Refusing to connect to non-HTTPS endpoint: ${baseUrl}. ` +
                'Set ALLOW_HTTP=true for internal/dev networks, or use an https:// address.');
        }
        this.baseUrl = baseUrl.replace(/\/+$/, '');
        this.credentials = credentials;
    }
    async request(options = {}) {
        this.checkCircuit();
        const method = (options.method ?? 'GET').toUpperCase();
        const path = options.path ?? '/';
        const url = this.buildUrl(path, options.query);
        const bodyStr = options.body ? JSON.stringify(options.body) : '';
        const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
        const headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        };
        this.applyAuth(headers, method, url.toString(), bodyStr);
        let lastError = null;
        for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
            try {
                const controller = new AbortController();
                const timer = setTimeout(() => controller.abort(), timeoutMs);
                const response = await fetch(url.toString(), {
                    method,
                    headers,
                    body: method !== 'GET' && bodyStr ? bodyStr : undefined,
                    signal: controller.signal,
                });
                clearTimeout(timer);
                this.recordSuccess();
                const text = await response.text();
                let data;
                try {
                    data = JSON.parse(text);
                }
                catch {
                    data = text || null;
                }
                if (!response.ok) {
                    return {
                        status: response.status,
                        data: { error: true, message: `HTTP ${response.status}`, details: data },
                    };
                }
                return { status: response.status, data };
            }
            catch (err) {
                lastError = err instanceof Error ? err : new Error(String(err));
                if (lastError.name === 'AbortError') {
                    lastError = new Error(`Request timed out after ${timeoutMs}ms`);
                }
                this.recordFailure();
                if (attempt < MAX_RETRIES - 1) {
                    const backoff = Math.min(1000 * 2 ** attempt, 8000);
                    const jitter = Math.random() * backoff * 0.3;
                    await this.sleep(backoff + jitter);
                    this.applyAuth(headers, method, url.toString(), bodyStr);
                }
            }
        }
        return {
            status: 502,
            data: {
                error: true,
                message: `All ${MAX_RETRIES} attempts failed`,
                details: lastError?.message,
            },
        };
    }
    buildUrl(path, query) {
        const fullPath = `/api/v1/${path.replace(/^\/+/, '')}`;
        const url = new URL(fullPath, this.baseUrl);
        if (query) {
            for (const [k, v] of Object.entries(query)) {
                url.searchParams.set(k, v);
            }
        }
        return url;
    }
    applyAuth(headers, method, url, body) {
        if (this.credentials.mode === 'session') {
            const creds = this.credentials;
            headers[creds.headerName] = creds.token;
        }
        else {
            const creds = this.credentials;
            const timestamp = new Date().toISOString();
            const authHeaders = signRequest(creds, method, url, body, timestamp);
            Object.assign(headers, authHeaders);
        }
    }
    checkCircuit() {
        if (!this.circuit.isOpen)
            return;
        const elapsed = Date.now() - this.circuit.lastFailure;
        if (elapsed > CIRCUIT_RESET_MS) {
            this.circuit = { failures: 0, lastFailure: 0, isOpen: false };
            return;
        }
        throw new Error(`Circuit breaker is OPEN — too many failures. ` +
            `Retry after ${Math.ceil((CIRCUIT_RESET_MS - elapsed) / 1000)}s.`);
    }
    recordFailure() {
        this.circuit.failures++;
        this.circuit.lastFailure = Date.now();
        if (this.circuit.failures >= CIRCUIT_THRESHOLD) {
            this.circuit.isOpen = true;
        }
    }
    recordSuccess() {
        this.circuit.failures = 0;
        this.circuit.isOpen = false;
    }
    sleep(ms) {
        return new Promise((resolve) => setTimeout(resolve, ms));
    }
}
//# sourceMappingURL=client.js.map