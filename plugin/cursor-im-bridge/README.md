# Cursor IM Bridge

Bidirectional messaging bridge plugin for Cursor/VS Code — integrates with Feishu, WeCom, Telegram, DingTalk, and custom internal IM tools.

[中文文档](./README.zh-CN.md)

---

## Features

- **Multi-platform IM support**: Feishu (Lark), WeCom, Telegram, DingTalk
- **Custom adapter**: Connect to any internal IM system via HTTP API + field mapping
- **Bidirectional messaging**: Send and receive IM messages directly in Cursor
- **Cross-OS**: Windows, macOS, Linux
- **Secure credential storage**: Uses OS keychain via VS Code SecretStorage
- **Webhook receiver**: Built-in local HTTP server for IM callbacks
- **Message panel**: WebView panel for viewing message history

## Architecture

```
┌─────────────────────────────────────────────┐
│              VS Code Extension               │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Commands│  │StatusBar │  │ WebView   │  │
│  └────┬────┘  └────┬─────┘  └─────┬─────┘  │
│       └─────────────┼──────────────┘        │
│              ┌──────┴──────┐                │
│              │  IM Bridge  │                │
│              │ (Controller)│                │
│              └──────┬──────┘                │
│       ┌─────────────┼──────────────┐        │
│  ┌────┴────┐  ┌─────┴─────┐  ┌────┴────┐  │
│  │MessageBus│  │  Registry │  │Webhook  │  │
│  │(Router) │  │ (Factory) │  │ Server  │  │
│  └────┬────┘  └─────┬─────┘  └────┬────┘  │
│       └─────────────┼──────────────┘        │
│              ┌──────┴──────┐                │
│              │  Adapters   │                │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌─────┐ │
│  │ Feishu │ │ WeCom  │ │Telegram│ │ ... │ │
│  └────────┘ └────────┘ └────────┘ └─────┘ │
└─────────────────────────────────────────────┘
```

## Installation

```bash
cd plugin/cursor-im-bridge
npm install
npm run compile
```

To run in development mode, press F5 in VS Code to launch Extension Development Host.

## Configuration

Configure `cursorImBridge.adapters` in VS Code Settings:

```json
{
  "cursorImBridge.adapters": [
    {
      "type": "feishu",
      "name": "My Feishu Bot",
      "enabled": true,
      "config": {
        "appId": "cli_xxx",
        "appSecret": "xxx"
      }
    },
    {
      "type": "wecom",
      "name": "My WeCom Bot",
      "enabled": true,
      "config": {
        "corpId": "ww_xxx",
        "corpSecret": "xxx",
        "agentId": 1000001
      }
    },
    {
      "type": "telegram",
      "name": "My Telegram Bot",
      "enabled": true,
      "config": {
        "botToken": "123456:ABC-DEF",
        "allowedChatIds": ["12345678"]
      }
    },
    {
      "type": "dingtalk",
      "name": "My DingTalk Bot",
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
      "name": "Internal IM",
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

## Webhook Endpoints

After activation, the local webhook server listens on `http://127.0.0.1:3927/`:

| Path | Adapter | Description |
|------|---------|-------------|
| `/feishu` | Feishu | Event subscription callback |
| `/wecom` | WeCom | Message callback |
| `/telegram` | Telegram | Bot webhook updates |
| `/dingtalk` | DingTalk | Robot callback |
| `/custom` | Custom | Generic webhook endpoint |

## Custom Adapter

The custom adapter connects to any HTTP API-based IM system via configuration:

| Field | Description |
|-------|-------------|
| `baseUrl` | API base URL |
| `sendMessagePath` | POST endpoint for sending messages |
| `getChannelsPath` | GET endpoint for channel list |
| `getHistoryPath` | GET endpoint for message history |
| `auth` | Auth config (Bearer / API Key / Basic / Custom Header) |
| `fieldMapping` | Custom field name mapping |
| `pollingInterval` | Polling interval in ms (0 = webhook only) |

## Commands

| Command | Description |
|---------|-------------|
| `IM Bridge: Connect` | Connect to IM adapters |
| `IM Bridge: Disconnect` | Disconnect all adapters |
| `IM Bridge: Send Message` | Send a message via adapter |
| `IM Bridge: Show Panel` | Open message panel |
| `IM Bridge: Configure` | Open adapter settings |

## Cross-Platform Notes

| Aspect | Details |
|--------|---------|
| Credential storage | VS Code SecretStorage → Windows Credential Manager / macOS Keychain / Linux libsecret |
| Config paths | Auto-adapts to platform standard directories |
| Webhook binding | `127.0.0.1` only — no external exposure |
| Network timeouts | All requests include 10s timeout |
| Node.js | Requires Node.js 18+ (for native fetch) |

## Project Structure

```
plugin/cursor-im-bridge/
├── src/
│   ├── extension.ts            # Entry point
│   ├── types/index.ts          # Type definitions
│   ├── adapters/
│   │   ├── base-adapter.ts     # Abstract base class
│   │   ├── feishu-adapter.ts   # Feishu (Lark)
│   │   ├── wecom-adapter.ts    # WeCom
│   │   ├── telegram-adapter.ts # Telegram
│   │   ├── dingtalk-adapter.ts # DingTalk
│   │   └── custom-adapter.ts   # Custom / Internal IM
│   ├── core/
│   │   ├── im-bridge.ts        # Main controller
│   │   ├── adapter-registry.ts # Factory registry
│   │   ├── message-bus.ts      # Message router
│   │   └── webhook-server.ts   # Local HTTP server
│   └── utils/
│       ├── http-client.ts      # HTTP client wrapper
│       ├── platform.ts         # Platform detection
│       └── secure-store.ts     # Credential store
├── resources/
│   └── icon.svg
├── package.json
├── tsconfig.json
└── README.md
```

## License

MIT
