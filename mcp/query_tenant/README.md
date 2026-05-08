# MCP Query Server

A Model Context Protocol (MCP) server for querying authenticated APIs. Supports **session token** and **AK/SK (HMAC signature)** authentication modes, with credentials fully isolated from the AI — they never appear in tool parameters or return values.

## Security Design

- **Credential isolation**: Auth tokens are loaded from environment variables / `.env` file only; they are never passed as MCP tool parameters or exposed in responses.
- **HMAC signing**: In AK/SK mode, the Secret Key is used to compute an HMAC-SHA256 signature — the SK itself never appears in any HTTP request.
- **Response sanitization**: All API responses are scrubbed of sensitive headers/fields before being returned to the AI.
- **HTTPS enforced** by default. Set `ALLOW_HTTP=true` for internal/dev networks.
- **Circuit breaker**: Automatically stops requests after repeated failures, with auto-recovery.
- **Retry with backoff**: Failed requests are retried up to 3 times with exponential backoff and jitter.

---

## Installation

### From GitHub (Recommended)

Clone the repository and install dependencies:

```bash
git clone https://github.com/zy-eagle/ai.git
cd ai/mcp/query_server
npm install
```

The `build/` directory is pre-compiled and included in the repository — no build step required for basic use.

### From a local copy

Copy the `mcp/query_server/` directory to your project, then:

```bash
cd mcp/query_server
npm install
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

### Session Mode

```env
AUTH_MODE=session
BASE_URL=http://192.168.231.129:3000
ALLOW_HTTP=true
SESSION_TOKEN=Bearer your-jwt-token
SESSION_HEADER_NAME=Authorization
```

### AK/SK Mode

```env
AUTH_MODE=aksk
BASE_URL=https://api.example.com
ACCESS_KEY=your-access-key
SECRET_KEY=your-secret-key
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `BASE_URL` | Yes | — | Target API base URL |
| `AUTH_MODE` | No | `session` | `session` or `aksk` |
| `SESSION_TOKEN` | If session | — | Full token value (e.g. `Bearer xxx`) |
| `SESSION_HEADER_NAME` | No | `Authorization` | HTTP header name for the token |
| `ACCESS_KEY` | If aksk | — | Access Key for HMAC signing |
| `SECRET_KEY` | If aksk | — | Secret Key for HMAC signing |
| `ALLOW_HTTP` | No | `false` | Set `true` for internal/dev HTTP endpoints |
| `ENV_FILE` | No | — | Custom `.env` file path. If set, loads from this path instead of default `mcp/query_server/.env` |

---

## Usage with Cursor

Add to `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "query-server": {
      "command": "node",
      "args": ["mcp/query_server/build/index.js"]
    }
  }
}
```

Credentials are loaded from `mcp/query_server/.env` (create from `.env.example`).

If you want to store the `.env` file outside the project:

```json
{
  "mcpServers": {
    "query-server": {
      "command": "node",
      "args": ["mcp/query_server/build/index.js"],
      "env": {
        "ENV_FILE": "/path/to/your-credentials.env"
      }
    }
  }
}
```

---

## Available Tools

| Tool | Description |
|------|-------------|
| `query_data` | Fetch data from the authenticated API. Pass `path`, `method`, `query`, and `body` — auth is injected automatically. |
| `check_health` | Check if the remote API is reachable. |

### Example: query_data

```
Tool: query_data
Input: { "path": "/api/v1/tenants", "method": "GET" }
```

The server automatically adds the authentication header. The AI never sees or handles credentials.

---

## Development

```bash
npm install          # Install dependencies
npm run build        # Compile TypeScript → build/
npm run dev          # Watch mode
npx vitest run       # Run tests
```
