import { createHmac } from 'node:crypto';
import { readFileSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
function loadEnvFile(filePath) {
    if (!existsSync(filePath))
        return;
    const content = readFileSync(filePath, 'utf-8');
    for (const line of content.split('\n')) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('#'))
            continue;
        const eqIndex = trimmed.indexOf('=');
        if (eqIndex === -1)
            continue;
        const key = trimmed.slice(0, eqIndex).trim();
        const value = trimmed.slice(eqIndex + 1).trim().replace(/^["']|["']$/g, '');
        if (!(key in process.env)) {
            process.env[key] = value;
        }
    }
}
export function loadCredentials() {
    const thisDir = dirname(fileURLToPath(import.meta.url));
    const defaultEnv = resolve(thisDir, '..', '.env');
    const envFile = process.env.ENV_FILE;
    loadEnvFile(envFile ? resolve(envFile) : defaultEnv);
    const mode = (process.env.AUTH_MODE ?? 'session');
    if (mode === 'aksk') {
        const accessKey = process.env.ACCESS_KEY?.trim();
        const secretKey = process.env.SECRET_KEY?.trim();
        if (!accessKey || !secretKey) {
            throw new Error('AUTH_MODE=aksk requires ACCESS_KEY and SECRET_KEY environment variables');
        }
        return { mode: 'aksk', accessKey, secretKey };
    }
    const token = process.env.SESSION_TOKEN?.trim();
    const headerName = process.env.SESSION_HEADER_NAME ?? 'Authorization';
    if (!token) {
        throw new Error('AUTH_MODE=session requires SESSION_TOKEN environment variable');
    }
    return { mode: 'session', token, headerName };
}
export function signRequest(creds, method, url, body, timestamp) {
    const stringToSign = [method.toUpperCase(), url, timestamp, body].join('\n');
    const signature = createHmac('sha256', creds.secretKey)
        .update(stringToSign)
        .digest('hex');
    return {
        'X-Access-Key': creds.accessKey,
        'X-Timestamp': timestamp,
        'X-Signature': signature,
    };
}
export function maskSecret(value) {
    if (value.length <= 6)
        return '***';
    return value.slice(0, 3) + '***' + value.slice(-3);
}
//# sourceMappingURL=auth.js.map