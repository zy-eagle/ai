import { type Credentials } from './auth.js';
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
export declare class AuthenticatedClient {
    private baseUrl;
    private credentials;
    private circuit;
    constructor(baseUrl: string, credentials: Credentials, allowHttp?: boolean);
    request(options?: QueryOptions): Promise<ApiResponse>;
    private buildUrl;
    private applyAuth;
    private checkCircuit;
    private recordFailure;
    private recordSuccess;
    private sleep;
}
export {};
//# sourceMappingURL=client.d.ts.map