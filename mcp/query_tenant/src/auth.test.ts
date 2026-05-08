import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { loadCredentials, signRequest, maskSecret, type AkskCredentials } from './auth.js';

describe('maskSecret', () => {
  it('masks long strings keeping first/last 3 chars', () => {
    expect(maskSecret('abcdefghij')).toBe('abc***hij');
  });

  it('masks short strings completely', () => {
    expect(maskSecret('abc')).toBe('***');
    expect(maskSecret('')).toBe('***');
  });
});

describe('signRequest', () => {
  it('produces deterministic HMAC-SHA256 signature', () => {
    const creds: AkskCredentials = { mode: 'aksk', accessKey: 'AK001', secretKey: 'SK_SECRET' };
    const ts = '2026-01-01T00:00:00Z';
    const h1 = signRequest(creds, 'GET', 'https://api.example.com/data', '', ts);
    const h2 = signRequest(creds, 'GET', 'https://api.example.com/data', '', ts);

    expect(h1['X-Access-Key']).toBe('AK001');
    expect(h1['X-Timestamp']).toBe(ts);
    expect(h1['X-Signature']).toMatch(/^[0-9a-f]{64}$/);
    expect(h1['X-Signature']).toBe(h2['X-Signature']);
  });

  it('produces different signature for different body', () => {
    const creds: AkskCredentials = { mode: 'aksk', accessKey: 'AK001', secretKey: 'SK_SECRET' };
    const ts = '2026-01-01T00:00:00Z';
    const h1 = signRequest(creds, 'POST', 'https://api.example.com/data', '{"a":1}', ts);
    const h2 = signRequest(creds, 'POST', 'https://api.example.com/data', '{"a":2}', ts);

    expect(h1['X-Signature']).not.toBe(h2['X-Signature']);
  });

  it('does not include secretKey in output headers', () => {
    const creds: AkskCredentials = { mode: 'aksk', accessKey: 'AK001', secretKey: 'SK_SECRET' };
    const headers = signRequest(creds, 'GET', 'https://api.example.com', '', '2026-01-01T00:00:00Z');
    const headerValues = Object.values(headers).join(' ');

    expect(headerValues).not.toContain('SK_SECRET');
  });
});

describe('loadCredentials', () => {
  const originalEnv = { ...process.env };

  beforeEach(() => {
    process.env = { ...originalEnv };
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('loads session credentials', () => {
    process.env.AUTH_MODE = 'session';
    process.env.SESSION_TOKEN = 'Bearer test-token-123';
    const creds = loadCredentials();

    expect(creds.mode).toBe('session');
    if (creds.mode === 'session') {
      expect(creds.token).toBe('Bearer test-token-123');
      expect(creds.headerName).toBe('Authorization');
    }
  });

  it('loads session with custom header name', () => {
    process.env.AUTH_MODE = 'session';
    process.env.SESSION_TOKEN = 'my-session';
    process.env.SESSION_HEADER_NAME = 'X-Session-Token';
    const creds = loadCredentials();

    if (creds.mode === 'session') {
      expect(creds.headerName).toBe('X-Session-Token');
    }
  });

  it('loads aksk credentials', () => {
    process.env.AUTH_MODE = 'aksk';
    process.env.ACCESS_KEY = 'AK_TEST';
    process.env.SECRET_KEY = 'SK_TEST';
    const creds = loadCredentials();

    expect(creds.mode).toBe('aksk');
    if (creds.mode === 'aksk') {
      expect(creds.accessKey).toBe('AK_TEST');
      expect(creds.secretKey).toBe('SK_TEST');
    }
  });

  it('throws when session mode lacks token', () => {
    process.env.AUTH_MODE = 'session';
    process.env.SESSION_TOKEN = '';

    expect(() => loadCredentials()).toThrow('SESSION_TOKEN');
  });

  it('throws when aksk mode lacks keys', () => {
    process.env.AUTH_MODE = 'aksk';
    process.env.ACCESS_KEY = '';
    process.env.SECRET_KEY = '';

    expect(() => loadCredentials()).toThrow('ACCESS_KEY');
  });
});
