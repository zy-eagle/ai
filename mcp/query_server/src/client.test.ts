import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { AuthenticatedClient } from './client.js';
import type { SessionCredentials, AkskCredentials } from './auth.js';

const sessionCreds: SessionCredentials = {
  mode: 'session',
  token: 'Bearer test-session-token',
  headerName: 'Authorization',
};

const akskCreds: AkskCredentials = {
  mode: 'aksk',
  accessKey: 'AK_TEST',
  secretKey: 'SK_TEST',
};

describe('AuthenticatedClient', () => {
  describe('constructor', () => {
    it('rejects non-HTTPS base URL by default', () => {
      expect(() => new AuthenticatedClient('http://insecure.com', sessionCreds))
        .toThrow('non-HTTPS');
    });

    it('allows HTTP when allowHttp=true', () => {
      expect(() => new AuthenticatedClient('http://internal.local', sessionCreds, true))
        .not.toThrow();
    });

    it('accepts HTTPS base URL', () => {
      expect(() => new AuthenticatedClient('https://secure.com', sessionCreds))
        .not.toThrow();
    });
  });

  describe('request with session auth', () => {
    let fetchSpy: ReturnType<typeof vi.fn>;

    beforeEach(() => {
      fetchSpy = vi.fn();
      vi.stubGlobal('fetch', fetchSpy);
    });

    afterEach(() => {
      vi.restoreAllMocks();
    });

    it('injects session token header on GET', async () => {
      fetchSpy.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ result: 'ok' }),
      });

      const client = new AuthenticatedClient('https://api.example.com', sessionCreds);
      const res = await client.request({ path: '/data', method: 'GET' });

      expect(res.status).toBe(200);
      expect(res.data).toEqual({ result: 'ok' });

      const [url, opts] = fetchSpy.mock.calls[0];
      expect(url).toContain('https://api.example.com/data');
      expect(opts.headers['Authorization']).toBe('Bearer test-session-token');
    });

    it('does NOT expose token in request body', async () => {
      fetchSpy.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({}),
      });

      const client = new AuthenticatedClient('https://api.example.com', sessionCreds);
      await client.request({ path: '/data', method: 'POST', body: { query: 'test' } });

      const [, opts] = fetchSpy.mock.calls[0];
      expect(opts.body).not.toContain('test-session-token');
    });
  });

  describe('request with aksk auth', () => {
    let fetchSpy: ReturnType<typeof vi.fn>;

    beforeEach(() => {
      fetchSpy = vi.fn();
      vi.stubGlobal('fetch', fetchSpy);
    });

    afterEach(() => {
      vi.restoreAllMocks();
    });

    it('injects HMAC signature headers', async () => {
      fetchSpy.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ data: [1, 2, 3] }),
      });

      const client = new AuthenticatedClient('https://api.example.com', akskCreds);
      await client.request({ path: '/data' });

      const [, opts] = fetchSpy.mock.calls[0];
      expect(opts.headers['X-Access-Key']).toBe('AK_TEST');
      expect(opts.headers['X-Timestamp']).toBeDefined();
      expect(opts.headers['X-Signature']).toMatch(/^[0-9a-f]{64}$/);
    });

    it('does NOT include secret key in any header', async () => {
      fetchSpy.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({}),
      });

      const client = new AuthenticatedClient('https://api.example.com', akskCreds);
      await client.request({ path: '/test' });

      const [, opts] = fetchSpy.mock.calls[0];
      const allHeaderValues = Object.values(opts.headers as Record<string, string>).join(' ');
      expect(allHeaderValues).not.toContain('SK_TEST');
    });
  });

  describe('error handling', () => {
    let fetchSpy: ReturnType<typeof vi.fn>;

    beforeEach(() => {
      fetchSpy = vi.fn();
      vi.stubGlobal('fetch', fetchSpy);
    });

    afterEach(() => {
      vi.restoreAllMocks();
    });

    it('returns structured error on HTTP 4xx/5xx', async () => {
      fetchSpy.mockResolvedValue({
        ok: false,
        status: 403,
        json: () => Promise.resolve({ reason: 'forbidden' }),
      });

      const client = new AuthenticatedClient('https://api.example.com', sessionCreds);
      const res = await client.request({ path: '/secret' });

      expect(res.status).toBe(403);
      expect((res.data as Record<string, unknown>).error).toBe(true);
    });

    it('retries on network failure and returns 502 after exhaustion', async () => {
      fetchSpy.mockRejectedValue(new Error('network error'));

      const client = new AuthenticatedClient('https://api.example.com', sessionCreds);
      const res = await client.request({ path: '/fail', timeoutMs: 100 });

      expect(res.status).toBe(502);
      expect((res.data as Record<string, unknown>).message).toContain('All 3 attempts failed');
      expect(fetchSpy).toHaveBeenCalledTimes(3);
    }, 30000);

    it('trips circuit breaker after repeated failures', async () => {
      fetchSpy.mockRejectedValue(new Error('down'));

      const client = new AuthenticatedClient('https://api.example.com', sessionCreds);

      for (let i = 0; i < 2; i++) {
        await client.request({ path: '/fail', timeoutMs: 100 });
      }

      await expect(client.request({ path: '/fail', timeoutMs: 100 }))
        .rejects.toThrow('Circuit breaker is OPEN');
    }, 60000);
  });

  describe('timeout', () => {
    afterEach(() => {
      vi.restoreAllMocks();
    });

    it('aborts request after timeout', async () => {
      const fetchSpy = vi.fn().mockImplementation(
        () => new Promise((_, reject) => {
          setTimeout(() => reject(new DOMException('The operation was aborted', 'AbortError')), 50);
        })
      );
      vi.stubGlobal('fetch', fetchSpy);

      const client = new AuthenticatedClient('https://api.example.com', sessionCreds);
      const res = await client.request({ path: '/slow', timeoutMs: 50 });

      expect(res.status).toBe(502);
    }, 30000);
  });
});
