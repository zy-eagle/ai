# MCP Query Server (Go)

A Model Context Protocol (MCP) server for querying authenticated APIs, implemented in Go with **zero external dependencies**. Supports **session token** and **AK/SK (HMAC signature)** authentication modes, with credentials fully isolated from the AI.

This is a Go port of the [TypeScript version](../query_tenant/).

## Security Design

- **Credential isolation**: Auth tokens are loaded from environment variables / `.env` file only; they are never passed as MCP tool parameters or exposed in responses.
- **HMAC signing**: In AK/SK mode, the Secret Key is used to compute an HMAC-SHA256 signature — the SK itself never appears in any HTTP request.
- **Response sanitization**: All API responses are scrubbed of sensitive headers/fields before being returned to the AI.
- **HTTPS enforced** by default. Set `ALLOW_HTTP=true` for internal/dev networks.
- **Circuit breaker**: Automatically stops requests after repeated failures (5 consecutive), with 30s auto-recovery.
- **Retry with backoff**: Failed requests are retried up to 3 times with exponential backoff and jitter.

## Build

Requires Go 1.24+. No external dependencies.

```bash
cd mcp/query_tenant_go
go build -o query_tenant_go.exe .
```

Or on Linux/macOS:

```bash
go build -o query_tenant_go .
```

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
| `ENV_FILE` | No | — | Custom `.env` file path |

## Usage with Cursor

Add to `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "query-tenant-go": {
      "command": "path/to/query_tenant_go.exe",
      "args": [],
      "env": {
        "ENV_FILE": "/path/to/your.env"
      }
    }
  }
}
```

## Available Tools

| Tool | Description |
|------|-------------|
| `query_data` | Fetch data from the authenticated API. Pass `path`, `method`, `query`, and `body` — auth is injected automatically. |
| `check_health` | Check if the remote API is reachable. |

## MCP Protocol

This server implements the MCP JSON-RPC 2.0 stdio protocol from scratch (zero dependencies), supporting:

- `initialize` / `notifications/initialized` — handshake & capability negotiation
- `tools/list` — tool discovery
- `tools/call` — tool invocation
- `ping` — health check

Protocol version: `2025-11-25`

## Testing

```bash
go test -v ./...
```

## Architecture

| File | Description |
|------|-------------|
| `main.go` | Entry point |
| `server.go` | Tool registration (query_data, check_health) |
| `mcp.go` | JSON-RPC 2.0 stdio MCP protocol layer |
| `client.go` | HTTP client with retry, backoff, circuit breaker |
| `auth.go` | .env loading, credential parsing, HMAC-SHA256 signing |
| `sanitize.go` | Response sensitive field masking |
