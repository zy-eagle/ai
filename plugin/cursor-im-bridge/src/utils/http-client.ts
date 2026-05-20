import * as https from 'https';
import * as http from 'http';
import { URL } from 'url';

/**
 * 跨平台 HTTP 客户端封装
 * 使用 Node.js 内置 https/http 模块，兼容所有 Electron 和 Node.js 环境
 */
export interface HttpClientOptions {
  timeout?: number;
  defaultHeaders?: Record<string, string>;
}

export class HttpClient {
  private baseUrl: string;
  private timeout: number;
  private defaultHeaders: Record<string, string>;

  constructor(baseUrl: string, options: HttpClientOptions = {}) {
    this.baseUrl = baseUrl;
    this.timeout = options.timeout || 10000;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
      ...options.defaultHeaders,
    };
  }

  async get<T>(path: string, headers?: Record<string, string>): Promise<T> {
    return this.request<T>('GET', path, undefined, headers);
  }

  async post<T>(path: string, body: unknown, headers?: Record<string, string>): Promise<T> {
    return this.request<T>('POST', path, body, headers);
  }

  async postAbsolute<T>(url: string, body: unknown, headers?: Record<string, string>): Promise<T> {
    return this.doRequest<T>('POST', url, body, headers);
  }

  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
    headers?: Record<string, string>
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    return this.doRequest<T>(method, url, body, headers);
  }

  private doRequest<T>(
    method: string,
    url: string,
    body?: unknown,
    headers?: Record<string, string>
  ): Promise<T> {
    return new Promise((resolve, reject) => {
      const parsed = new URL(url);
      const isHttps = parsed.protocol === 'https:';
      const lib = isHttps ? https : http;

      const payload = body ? JSON.stringify(body) : undefined;
      const reqHeaders: Record<string, string> = {
        ...this.defaultHeaders,
        ...headers,
      };
      if (payload) {
        reqHeaders['Content-Length'] = Buffer.byteLength(payload).toString();
      }

      const req = lib.request(
        {
          hostname: parsed.hostname,
          port: parsed.port || (isHttps ? 443 : 80),
          path: parsed.pathname + parsed.search,
          method,
          headers: reqHeaders,
        },
        (res) => {
          const MAX_BODY = 2 * 1024 * 1024; // 2 MB guard
          let data = '';
          let exceeded = false;
          res.on('data', (chunk) => {
            if (exceeded) return;
            data += chunk;
            if (data.length > MAX_BODY) {
              exceeded = true;
              res.destroy();
              reject(new Error(`Response body exceeds ${MAX_BODY} bytes`));
            }
          });
          res.on('end', () => {
            if (exceeded) return;
            if (res.statusCode && res.statusCode >= 200 && res.statusCode < 300) {
              try {
                resolve(JSON.parse(data) as T);
              } catch {
                reject(new Error(`Invalid JSON response: ${data.slice(0, 200)}`));
              }
            } else {
              reject(new Error(`HTTP ${res.statusCode}: ${data.slice(0, 200)}`));
            }
          });
        }
      );

      req.setTimeout(this.timeout, () => {
        req.destroy();
        reject(new Error(`Request timeout after ${this.timeout}ms`));
      });

      req.on('error', (err) => {
        reject(new Error(`HTTP request failed: ${err.message}`));
      });

      if (payload) {
        req.write(payload);
      }
      req.end();
    });
  }
}
