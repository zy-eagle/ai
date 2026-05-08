import { describe, it, expect } from 'vitest';
import { sanitizeResponse } from './sanitize.js';

describe('sanitizeResponse', () => {
  it('masks sensitive keys in flat object', () => {
    const input = { name: 'test', authorization: 'Bearer secret-token-123456', count: 5 };
    const result = sanitizeResponse(input) as Record<string, unknown>;

    expect(result.name).toBe('test');
    expect(result.count).toBe(5);
    expect(result.authorization).not.toContain('secret-token');
  });

  it('masks sensitive keys in nested object', () => {
    const input = { headers: { cookie: 'sid=abc123456789' }, data: { ok: true } };
    const result = sanitizeResponse(input) as Record<string, unknown>;
    const headers = result.headers as Record<string, unknown>;

    expect(headers.cookie).not.toContain('abc123456789');
  });

  it('masks sensitive keys in arrays', () => {
    const input = [{ token: 'my-long-secret-token-value' }, { name: 'safe' }];
    const result = sanitizeResponse(input) as Record<string, unknown>[];

    expect(result[0].token).not.toContain('secret-token');
    expect(result[1].name).toBe('safe');
  });

  it('passes through primitives unchanged', () => {
    expect(sanitizeResponse('hello')).toBe('hello');
    expect(sanitizeResponse(42)).toBe(42);
    expect(sanitizeResponse(null)).toBe(null);
    expect(sanitizeResponse(undefined)).toBe(undefined);
  });

  it('masks password field', () => {
    const input = { user: 'admin', password: 'super-secret-password-123' };
    const result = sanitizeResponse(input) as Record<string, unknown>;

    expect(result.user).toBe('admin');
    expect(result.password).not.toContain('super-secret');
  });
});
