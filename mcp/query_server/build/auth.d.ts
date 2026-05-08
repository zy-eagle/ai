export type AuthMode = 'session' | 'aksk';
export interface SessionCredentials {
    mode: 'session';
    token: string;
    headerName: string;
}
export interface AkskCredentials {
    mode: 'aksk';
    accessKey: string;
    secretKey: string;
}
export type Credentials = SessionCredentials | AkskCredentials;
export declare function loadCredentials(): Credentials;
export declare function signRequest(creds: AkskCredentials, method: string, url: string, body: string, timestamp: string): Record<string, string>;
export declare function maskSecret(value: string): string;
//# sourceMappingURL=auth.d.ts.map