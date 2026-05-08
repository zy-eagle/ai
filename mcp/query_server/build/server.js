import { McpServer, StdioServerTransport } from '@modelcontextprotocol/server';
import * as z from 'zod/v4';
import { loadCredentials, maskSecret } from './auth.js';
import { AuthenticatedClient } from './client.js';
import { sanitizeResponse } from './sanitize.js';
export async function createServer() {
    const credentials = loadCredentials();
    const baseUrl = process.env.BASE_URL ?? 'https://mydomain.com';
    const allowHttp = process.env.ALLOW_HTTP === 'true';
    const client = new AuthenticatedClient(baseUrl, credentials, allowHttp);
    const authDescription = credentials.mode === 'session'
        ? 'session (token header)'
        : `aksk (AK: ${maskSecret(credentials.accessKey)})`;
    const server = new McpServer({ name: 'query-server', version: '0.1.0' }, {
        instructions: `Authenticated data query server. Auth mode: ${authDescription}. ` +
            'Credentials are managed server-side — never ask the user for tokens or keys. ' +
            'Use query_data to fetch data from the remote API.',
    });
    server.registerTool('query_data', {
        title: 'Query Data',
        description: 'Fetch data from the authenticated API. ' +
            'The server handles authentication automatically — do NOT include any auth tokens or credentials in parameters.',
        inputSchema: z.object({
            path: z
                .string()
                .default('/')
                .describe('API endpoint path, e.g. "/users" or "/data/reports"'),
            method: z
                .enum(['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
                .default('GET')
                .describe('HTTP method'),
            query: z
                .record(z.string(), z.string())
                .optional()
                .describe('URL query parameters as key-value pairs'),
            body: z
                .record(z.string(), z.unknown())
                .optional()
                .describe('Request body for POST/PUT/PATCH requests'),
        }),
    }, async ({ path, method, query, body }) => {
        try {
            const response = await client.request({ method, path, query, body });
            const sanitized = sanitizeResponse(response.data);
            return {
                content: [
                    {
                        type: 'text',
                        text: JSON.stringify({ status: response.status, data: sanitized }, null, 2),
                    },
                ],
            };
        }
        catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            return {
                content: [{ type: 'text', text: JSON.stringify({ error: true, message }) }],
                isError: true,
            };
        }
    });
    server.registerTool('check_health', {
        title: 'Health Check',
        description: 'Check if the remote API is reachable. No authentication details are exposed.',
        inputSchema: z.object({}),
    }, async () => {
        try {
            const response = await client.request({ method: 'GET', path: '/', timeoutMs: 5000 });
            return {
                content: [
                    {
                        type: 'text',
                        text: JSON.stringify({
                            healthy: response.status >= 200 && response.status < 500,
                            status: response.status,
                            auth_mode: credentials.mode,
                        }),
                    },
                ],
            };
        }
        catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            return {
                content: [
                    {
                        type: 'text',
                        text: JSON.stringify({ healthy: false, error: message }),
                    },
                ],
                isError: true,
            };
        }
    });
    return server;
}
export async function startServer() {
    const server = await createServer();
    const transport = new StdioServerTransport();
    await server.connect(transport);
}
//# sourceMappingURL=server.js.map