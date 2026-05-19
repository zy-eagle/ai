import * as http from 'http';
import { AdapterType, WebhookPayload } from '../types';
import { EventEmitter } from 'events';

/**
 * 本地 Webhook 接收服务器
 * 接收来自 IM 平台的回调消息并路由到对应适配器
 * 跨平台兼容：使用 Node.js 原生 http 模块
 */
export class WebhookServer extends EventEmitter {
  private server: http.Server | null = null;
  private port: number;

  constructor(port: number = 3927) {
    super();
    this.port = port;
  }

  async start(): Promise<void> {
    if (this.server) return;

    return new Promise((resolve, reject) => {
      this.server = http.createServer((req, res) => {
        this.handleRequest(req, res);
      });

      this.server.on('error', (err) => {
        if ((err as NodeJS.ErrnoException).code === 'EADDRINUSE') {
          this.port++;
          this.server!.listen(this.port, '127.0.0.1');
        } else {
          reject(err);
        }
      });

      this.server.listen(this.port, '127.0.0.1', () => {
        resolve();
      });
    });
  }

  async stop(): Promise<void> {
    if (!this.server) return;

    return new Promise((resolve) => {
      this.server!.close(() => {
        this.server = null;
        resolve();
      });
    });
  }

  getPort(): number {
    return this.port;
  }

  private handleRequest(req: http.IncomingMessage, res: http.ServerResponse): void {
    if (req.method !== 'POST') {
      res.writeHead(405);
      res.end('Method Not Allowed');
      return;
    }

    const chunks: Buffer[] = [];
    req.on('data', (chunk) => chunks.push(chunk));
    req.on('end', () => {
      try {
        const body = Buffer.concat(chunks).toString('utf-8');
        const data = JSON.parse(body);
        const path = req.url || '/';

        const adapterType = this.resolveAdapterType(path);
        if (!adapterType) {
          res.writeHead(404);
          res.end('Unknown adapter path');
          return;
        }

        // 飞书 URL 验证挑战
        if (data.challenge) {
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ challenge: data.challenge }));
          return;
        }

        const payload: WebhookPayload = {
          adapter: adapterType,
          event: data.event_type || data.EventType || 'message',
          data,
          timestamp: Date.now(),
        };

        this.emit('webhook', payload);

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ code: 0, msg: 'ok' }));
      } catch {
        res.writeHead(400);
        res.end('Invalid request body');
      }
    });
  }

  private resolveAdapterType(path: string): AdapterType | null {
    const normalized = path.toLowerCase().replace(/^\/+|\/+$/g, '');
    const segment = normalized.split('/')[0];

    const mapping: Record<string, AdapterType> = {
      feishu: AdapterType.Feishu,
      lark: AdapterType.Feishu,
      wecom: AdapterType.WeCom,
      wechat: AdapterType.WeCom,
      telegram: AdapterType.Telegram,
      dingtalk: AdapterType.DingTalk,
      custom: AdapterType.Custom,
    };

    return mapping[segment] || null;
  }
}
