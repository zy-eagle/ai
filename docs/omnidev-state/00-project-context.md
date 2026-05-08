---
project_type: legacy
stack: python
stability_level: high
---

## Stack & Layers

- **Language**: TypeScript (ES2022)
- **Framework**: @modelcontextprotocol/server (McpServer + StdioServerTransport)
- **HTTP Client**: Node.js fetch / undici
- **Schema Validation**: zod/v4
- **Package Manager**: npm
- **Transport**: stdio
- **Project Layout**: `mcp/<server_name>/` — 每个 MCP Server 独立子目录

## Conventions (TypeScript MCP SDK)

- 使用 `new McpServer({ name, version })` 初始化服务器
- 工具用 `server.registerTool(name, { inputSchema, description }, handler)` 注册
- 使用 `StdioServerTransport` + `server.connect(transport)` 启动
- Zod v4 定义输入/输出 schema

## Dependency Topology

- **Third-Party HTTP API**: 外部需鉴权接口 (Session / AK-SK)

## Stability Level

**high** — 用户明确要求安全级别高，凭证不得暴露给 AI
