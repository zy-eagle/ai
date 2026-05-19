# Cursor IM Bridge

双向通讯桥接插件，支持飞书、企微、Telegram、钉钉及自定义内部 IM 工具与 Cursor/VS Code 的集成。

[English](./README.md)

---

## 功能特性

- **IM → Cursor CLI → IM 全自动处理**: 从飞书/企微/钉钉发消息 → 自动调用 Cursor CLI Agent 处理 → 结果回复到 IM
- **多平台 IM 支持**: 飞书 (Feishu/Lark)、企业微信 (WeCom)、Telegram、钉钉 (DingTalk)
- **自定义适配器**: 可接入任何内部 IM 系统（通过 HTTP API + 字段映射配置）
- **双向通讯**: 在 Cursor 中接收和发送 IM 消息
- **跨操作系统**: 支持 Windows、macOS、Linux
- **长连接优先**: 飞书(WebSocket)、钉钉(Stream)、企微(Polling)、Telegram(Long Polling) — 不需要公网 URL
- **安全凭证管理**: 使用操作系统钥匙链存储敏感凭证
- **消息面板**: WebView 面板展示消息历史

## 架构

```
┌─────────────────────────────────────────────┐
│              VS Code Extension               │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  │
│  │ 命令注册│  │ 状态栏   │  │ WebView   │  │
│  └────┬────┘  └────┬─────┘  └─────┬─────┘  │
│       └─────────────┼──────────────┘        │
│              ┌──────┴──────┐                │
│              │  IM Bridge  │                │
│              │  (主控制器) │                │
│              └──────┬──────┘                │
│       ┌─────────────┼──────────────┐        │
│  ┌────┴────┐  ┌─────┴─────┐  ┌────┴────┐  │
│  │消息总线 │  │ 适配器注册│  │Webhook  │  │
│  │(路由器) │  │  (工厂)   │  │ 服务器  │  │
│  └────┬────┘  └─────┬─────┘  └────┬────┘  │
│       └─────────────┼──────────────┘        │
│              ┌──────┴──────┐                │
│              │   适配器层   │                │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌─────┐ │
│  │  飞书  │ │  企微  │ │Telegram│ │ ... │ │
│  └────────┘ └────────┘ └────────┘ └─────┘ │
└─────────────────────────────────────────────┘
```

## 安装

```bash
cd plugin/cursor-im-bridge
npm install
npm run compile
```

开发模式下，在 VS Code 中按 F5 启动 Extension Development Host。

## 配置

在 VS Code Settings 中配置 `cursorImBridge.adapters`：

```json
{
  "cursorImBridge.adapters": [
    {
      "type": "feishu",
      "name": "我的飞书机器人",
      "enabled": true,
      "config": {
        "appId": "cli_xxx",
        "appSecret": "xxx"
      }
    },
    {
      "type": "wecom",
      "name": "我的企微机器人",
      "enabled": true,
      "config": {
        "corpId": "ww_xxx",
        "corpSecret": "xxx",
        "agentId": 1000001
      }
    },
    {
      "type": "telegram",
      "name": "我的 Telegram 机器人",
      "enabled": true,
      "config": {
        "botToken": "123456:ABC-DEF",
        "allowedChatIds": ["12345678"]
      }
    },
    {
      "type": "dingtalk",
      "name": "我的钉钉机器人",
      "enabled": true,
      "config": {
        "appKey": "xxx",
        "appSecret": "xxx",
        "robotWebhookUrl": "https://oapi.dingtalk.com/robot/send?access_token=xxx",
        "robotSecret": "SEC_xxx"
      }
    },
    {
      "type": "custom",
      "name": "内部 IM",
      "enabled": true,
      "config": {
        "baseUrl": "https://im.internal.company.com/api",
        "sendMessagePath": "/messages/send",
        "getChannelsPath": "/channels",
        "getHistoryPath": "/messages/history",
        "auth": {
          "type": "bearer",
          "credentials": "your-token-here"
        },
        "fieldMapping": {
          "channelIdField": "channel_id",
          "contentField": "content",
          "senderIdField": "sender_id",
          "senderNameField": "sender_name",
          "messageIdField": "id",
          "timestampField": "timestamp"
        },
        "pollingInterval": 5000,
        "pollingPath": "/messages/poll"
      }
    }
  ],
  "cursorImBridge.webhookPort": 3927,
  "cursorImBridge.autoConnect": false,
  "cursorImBridge.logLevel": "info"
}
```

## Webhook 端点

启动后，本地 Webhook 服务监听 `http://127.0.0.1:3927/`：

| 路径 | 适配器 | 说明 |
|------|--------|------|
| `/feishu` | 飞书 | 事件订阅回调 |
| `/wecom` | 企业微信 | 消息回调 |
| `/telegram` | Telegram | Bot webhook 更新 |
| `/dingtalk` | 钉钉 | 机器人回调 |
| `/custom` | 自定义 | 通用 webhook 端点 |

## 自定义适配器

自定义适配器支持通过配置连接任何 HTTP API 形式的 IM 系统：

| 配置字段 | 说明 |
|----------|------|
| `baseUrl` | API 基础地址 |
| `sendMessagePath` | 发送消息的 POST 端点 |
| `getChannelsPath` | 获取通道列表的 GET 端点 |
| `getHistoryPath` | 获取历史消息的 GET 端点 |
| `auth` | 认证配置，支持 Bearer / API Key / Basic / Custom Header |
| `fieldMapping` | 自定义字段名映射（适配不同 API 响应格式） |
| `pollingInterval` | 轮询间隔（毫秒），设 0 表示仅使用 Webhook |

### 认证方式

| 类型 | 说明 | 配置示例 |
|------|------|---------|
| `bearer` | Bearer Token 认证 | `{ "type": "bearer", "credentials": "token" }` |
| `apikey` | API Key 认证（X-API-Key 头） | `{ "type": "apikey", "credentials": "key" }` |
| `basic` | HTTP Basic 认证 | `{ "type": "basic", "credentials": "base64(user:pass)" }` |
| `custom-header` | 自定义 Header | `{ "type": "custom-header", "headerName": "X-Auth", "credentials": "value" }` |

## 核心功能：IM → Cursor CLI → IM 自动处理

这是本插件的核心场景：**用飞书（或其他 IM）代替 Cursor 聊天窗口**，实现远程通过 IM 向 Cursor 发任务、获取结果。

### 工作流程

```
┌──────────┐         ┌─────────────────┐         ┌─────────────┐         ┌──────────┐
│ 飞书用户  │ 发消息  │  IM Bridge 插件  │ 调用    │ Cursor CLI   │ 回复    │ 飞书用户  │
│          │ ──────→ │  (接收消息)      │ ──────→ │ Agent 处理   │ ──────→ │ (收结果) │
└──────────┘         └─────────────────┘         └─────────────┘         └──────────┘
```

### 启用自动回复

在 `settings.json` 中配置：

```json
{
  "cursorImBridge.autoReply.enabled": true,
  "cursorImBridge.autoReply.triggerPrefix": "",
  "cursorImBridge.autoReply.timeout": 120000,
  "cursorImBridge.autoReply.sendProcessingStatus": true,
  "cursorImBridge.autoReply.maxConcurrent": 3
}
```

### 配置说明

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `autoReply.enabled` | `false` | 开启后，所有收到的 IM 消息自动触发 Cursor CLI 处理 |
| `autoReply.cliPath` | 自动检测 | Cursor CLI 路径（Windows: `cursor.cmd`, macOS: `/usr/local/bin/cursor`） |
| `autoReply.cwd` | 当前工作区 | CLI 工作目录（决定 Cursor Agent 操作的项目） |
| `autoReply.timeout` | `120000` | 执行超时（毫秒） |
| `autoReply.triggerPrefix` | `""` | 消息前缀过滤（空=所有消息；设为如 `/ask ` 则仅响应以此开头的消息） |
| `autoReply.sendProcessingStatus` | `true` | 处理前先回复一条「正在处理」状态消息 |
| `autoReply.processingTemplate` | `⏳ 正在处理您的请求，请稍候...` | 处理中消息模板 |
| `autoReply.successTemplate` | `{output}` | 成功回复模板（`{output}` = CLI输出, `{duration}` = 耗时） |
| `autoReply.errorTemplate` | `❌ 处理失败: {error}` | 失败回复模板 |
| `autoReply.ignoreSenders` | `["self"]` | 忽略这些发送者（防止回复自己的消息造成循环） |
| `autoReply.maxConcurrent` | `3` | 最大并发任务数 |

### 使用示例

飞书用户在群中发送：
```
帮我分析 src/utils.ts 的性能问题
```

Cursor IM Bridge 自动：
1. 回复 `⏳ 正在处理您的请求，请稍候...`
2. 调用 `cursor agent --message "帮我分析 src/utils.ts 的性能问题"`
3. 等待 Cursor CLI Agent 处理完成
4. 将 Agent 输出结果回复到飞书群

### 安全建议

- 设置 `triggerPrefix`（如 `/cursor `）避免所有消息都触发处理
- 配置 `ignoreSenders` 排除不需要响应的人
- 限制 `maxConcurrent` 防止资源耗尽

## 命令

| 命令 | 说明 |
|------|------|
| `IM Bridge: Connect` | 连接 IM 适配器 |
| `IM Bridge: Disconnect` | 断开所有适配器 |
| `IM Bridge: Send Message` | 通过适配器发送消息 |
| `IM Bridge: Show Panel` | 打开消息面板 |
| `IM Bridge: Configure` | 打开适配器配置 |
| `IM Bridge: Toggle Auto-Reply` | 开关自动回复 |
| `IM Bridge: Process Prompt via CLI` | 手动输入 prompt 调用 CLI 并发送结果 |

## 跨平台说明

| 方面 | 说明 |
|------|------|
| 凭证存储 | 使用 VS Code SecretStorage → Windows Credential Manager / macOS Keychain / Linux libsecret |
| 配置路径 | 自动适配各平台标准目录（Windows: AppData, macOS: Library, Linux: XDG） |
| Webhook 绑定 | 仅绑定 `127.0.0.1`，无外部暴露风险 |
| 网络超时 | 所有请求含 10 秒超时控制 |
| Node.js | 需要 Node.js 18+（使用原生 fetch） |

## 项目结构

```
plugin/cursor-im-bridge/
├── src/
│   ├── extension.ts            # 插件入口
│   ├── types/index.ts          # 类型定义
│   ├── adapters/
│   │   ├── base-adapter.ts     # 适配器抽象基类
│   │   ├── feishu-adapter.ts   # 飞书适配器 (WebSocket)
│   │   ├── wecom-adapter.ts    # 企业微信适配器 (Polling)
│   │   ├── telegram-adapter.ts # Telegram 适配器 (Long Polling)
│   │   ├── dingtalk-adapter.ts # 钉钉适配器 (Stream)
│   │   └── custom-adapter.ts   # 自定义适配器（内部 IM）
│   ├── core/
│   │   ├── im-bridge.ts        # 主控制器 (集成任务处理)
│   │   ├── adapter-registry.ts # 适配器工厂注册表
│   │   ├── message-bus.ts      # 消息路由总线
│   │   └── webhook-server.ts   # 本地 Webhook HTTP 服务
│   ├── processor/
│   │   ├── cursor-cli.ts       # Cursor CLI 执行器
│   │   └── task-processor.ts   # 任务处理器 (消息→CLI→回复)
│   └── utils/
│       ├── http-client.ts      # HTTP 客户端封装
│       ├── platform.ts         # 平台检测工具
│       └── secure-store.ts     # 安全凭证存储
├── resources/
│   └── icon.svg                # 侧边栏图标
├── package.json
├── tsconfig.json
├── README.md                   # 英文文档
└── README.zh-CN.md             # 中文文档
```

## 开发指南

### 添加新适配器

1. 在 `src/adapters/` 下创建新文件，继承 `BaseAdapter`
2. 实现 `doConnect`、`doDisconnect`、`sendMessage`、`getChannels`、`getHistory` 方法
3. 在 `src/adapters/index.ts` 中导出
4. 在 `src/core/adapter-registry.ts` 的 `builtinFactories` 中注册
5. 在 `AdapterType` 枚举中添加新类型

### 调试

1. 在 VS Code 中按 F5 启动调试
2. 在 Extension Development Host 中执行命令
3. 查看 Output Channel "IM Bridge" 获取日志

## 许可证

MIT
