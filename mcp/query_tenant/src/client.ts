import {
  type Credentials,
  type AkskCredentials,
  type SessionCredentials,
  signRequest,
} from './auth.js';

interface QueryOptions {
  method?: string;
  path?: string;
  query?: Record<string, string>;
  body?: unknown;
  timeoutMs?: number;
}

interface ApiResponse {
  status: number;
  data: unknown;
}

interface CircuitState {
  failures: number;
  lastFailure: number;
  isOpen: boolean;
}

const DEFAULT_TIMEOUT_MS = 10_000;
const MAX_RETRIES = 3;
const CIRCUIT_THRESHOLD = 5;
const CIRCUIT_RESET_MS = 30_000;

export class AuthenticatedClient {
  private baseUrl: string;
  private credentials: Credentials;
  private circuit: CircuitState = { failures: 0, lastFailure: 0, isOpen: false };

  constructor(baseUrl: string, credentials: Credentials, allowHttp = false) {
    if (!baseUrl.startsWith('https://') && !allowHttp) {
      throw new Error(
        `Refusing to connect to non-HTTPS endpoint: ${baseUrl}. ` +
        'Set ALLOW_HTTP=true for internal/dev networks, or use an https:// address.'
      );
    }
    this.baseUrl = baseUrl.replace(/\/+$/, '');
    this.credentials = credentials;
  }

  async request(options: QueryOptions = {}): Promise<ApiResponse> {
    this.checkCircuit();

    const method = (options.method ?? 'GET').toUpperCase();
    const path = options.path ?? '/';
    const url = this.buildUrl(path, options.query);
    const bodyStr = options.body ? JSON.stringify(options.body) : '';
    const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    };

    this.applyAuth(headers, method, url.toString(), bodyStr);

    let lastError: Error | null = null;

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
        let data: unknown;
        try {
          data = JSON.parse(text);
        } catch {
          data = text || null;
        }

        if (!response.ok) {
          return {
            status: response.status,
            data: { error: true, message: `HTTP ${response.status}`, details: data },
          };
        }

        return { status: response.status, data };
      } catch (err) {
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

  private buildUrl(path: string, query?: Record<string, string>): URL {
    const fullPath = `/api/v1/${path.replace(/^\/+/, '')}`;
    const url = new URL(fullPath, this.baseUrl);
    if (query) {
      for (const [k, v] of Object.entries(query)) {
        url.searchParams.set(k, v);
      }
    }
    return url;
  }

  private applyAuth(
    headers: Record<string, string>,
    method: string,
    url: string,
    body: string,
  ): void {
    if (this.credentials.mode === 'session') {
      const creds = this.credentials as SessionCredentials;
      headers[creds.headerName] = creds.token;
    } else {
      const creds = this.credentials as AkskCredentials;
      const timestamp = new Date().toISOString();
      const authHeaders = signRequest(creds, method, url, body, timestamp);
      Object.assign(headers, authHeaders);
    }
  }

  private checkCircuit(): void {
    if (!this.circuit.isOpen) return;

    const elapsed = Date.now() - this.circuit.lastFailure;
    if (elapsed > CIRCUIT_RESET_MS) {
      this.circuit = { failures: 0, lastFailure: 0, isOpen: false };
      return;
    }

    throw new Error(
      `Circuit breaker is OPEN — too many failures. ` +
      `Retry after ${Math.ceil((CIRCUIT_RESET_MS - elapsed) / 1000)}s.`
    );
  }

  private recordFailure(): void {
    this.circuit.failures++;
    this.circuit.lastFailure = Date.now();
    if (this.circuit.failures >= CIRCUIT_THRESHOLD) {
      this.circuit.isOpen = true;
    }
  }

  private recordSuccess(): void {
    this.circuit.failures = 0;
    this.circuit.isOpen = false;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}
