---
total_tasks: 5
parallel_groups: 3
critical_path: [T1 → T3 → T4 → T5]
frontend_impact: no
tech_stack: TypeScript, @modelcontextprotocol/server, zod/v4
---

## Group 1 (parallel — no prerequisites)

- [ ] **T1** [backend] 项目脚手架 — 创建 `mcp/query_server/` 目录，初始化 `package.json`、`tsconfig.json`、入口文件，安装依赖 · outputs: `mcp/query_server/package.json`, `mcp/query_server/tsconfig.json`, `mcp/query_server/src/index.ts`
- [ ] **T2** [backend] 凭证管理模块 — 从环境变量 / `.env` 加载鉴权凭证（AK/SK、Session Token），凭证仅在服务端内存中，绝不通过 MCP 工具参数或返回值暴露给 AI · outputs: `mcp/query_server/src/auth.ts`

## Group 2 (serial — after Group 1)

- [ ] **T3** [backend] HTTP 客户端 & 鉴权拦截器 — 封装 HTTP 客户端，支持两种鉴权模式：①会话模式（Cookie/Session Token 注入 Header）②AK/SK 模式（HMAC-SHA256 签名注入 Header）。包含超时控制、重试、结构化错误 · depends: T1, T2 · outputs: `mcp/query_server/src/client.ts`
- [ ] **T4** [backend] MCP Server & 工具注册 — 注册 `query_data` 等工具，AI 只传入业务参数（查询条件），鉴权由服务端自动注入，返回值脱敏 · depends: T3 · outputs: `mcp/query_server/src/server.ts`

## Group 3 (serial — after Group 2)

- [ ] **T5** [backend] 安全加固 & 文档 — `.env.example`、README、确保 `.gitignore` 排除 `.env` · depends: T4 · outputs: `mcp/query_server/.env.example`, `mcp/query_server/README.md`

## 安全设计要点

1. **凭证隔离**: AK/SK 和 Session Token 仅通过环境变量注入，永远不作为 MCP 工具参数
2. **签名不可逆**: AK/SK 模式使用 HMAC-SHA256 签名，SK 不出现在任何请求中
3. **返回值脱敏**: 工具返回结果中移除鉴权相关 header/cookie
4. **日志安全**: 日志中 mask 所有凭证字段
5. **传输安全**: 强制 HTTPS，拒绝 HTTP 明文
