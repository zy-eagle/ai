import { maskSecret } from './auth.js';

const SENSITIVE_KEYS = new Set([
  'authorization', 'cookie', 'set-cookie', 'x-access-key',
  'x-signature', 'x-timestamp', 'session_token', 'secret_key',
  'access_key', 'token', 'api_key', 'apikey', 'password',
]);

export function sanitizeResponse(data: unknown): unknown {
  if (data == null) return data;
  if (typeof data !== 'object') return data;

  if (Array.isArray(data)) {
    return data.map(sanitizeResponse);
  }

  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(data as Record<string, unknown>)) {
    if (SENSITIVE_KEYS.has(key.toLowerCase())) {
      result[key] = typeof value === 'string' ? maskSecret(value) : '***';
    } else if (typeof value === 'object' && value !== null) {
      result[key] = sanitizeResponse(value);
    } else {
      result[key] = value;
    }
  }
  return result;
}
