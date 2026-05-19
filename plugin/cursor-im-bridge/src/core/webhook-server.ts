import * as http from 'http';
import { AdapterType, WebhookPayload } from '../types';
import { EventEmitter } from 'events';
import {
  verifyFeishuSignature,
  verifyFeishuToken,
  verifyDingTalkSignature,
  verifyWeComSignature,
} from '../utils/webhook-verifier';

export interface WebhookSecrets {
  feishu?: { verificationToken?: string; encryptKey?: string };
  dingtalk?: { robotSecret?: string };
  wecom?: { token?: string };
}

/**
 * 本地 Webhook 接收服务器
 * 接收来自 IM 平台的回调消息并路由到对应适配器
 * 跨平台兼容：使用 Node.js 原生 http 模块
 */
export class WebhookServer extends EventEmitter {
  private server: http.Server | null = null;
  private port: number;
  private secrets: WebhookSecrets;

  constructor(port: number = 3927, secrets: WebhookSecrets = {}) {
    super();
    this.port = port;
    this.secrets = secrets;
  }

  updateSecrets(secrets: WebhookSecrets): void {
    this.secrets = secrets;
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

    // 限制请求体大小（防止超大 payload 攻击）
    const MAX_BODY = 1024 * 512; // 512 KB
    let bodySize = 0;
    const chunks: Buffer[] = [];

    req.on('data', (chunk: Buffer) => {
      bodySize += chunk.length;
      if (bodySize > MAX_BODY) {
        res.writeHead(413);
        res.end('Payload Too Large');
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });

    req.on('end', () => {
      try {
        const bodyStr = Buffer.concat(chunks).toString('utf-8');
        const data = JSON.parse(bodyStr);
        const reqPath = req.url || '/';

        const adapterType = this.resolveAdapterType(reqPath);
        if (!adapterType) {
          res.writeHead(404);
          res.end('Unknown adapter path');
          return;
        }

        // ─── 签名验证 ────────────────────────────────────────────
        if (!this.verifyRequest(adapterType, req, bodyStr, data)) {
          res.writeHead(401);
          res.end('Unauthorized: invalid signature');
          return;
        }

        // 飞书 URL 验证挑战（challenge 在签名验证通过后响应）
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

  private verifyRequest(
    adapterType: AdapterType,
    req: http.IncomingMessage,
    bodyStr: string,
    data: Record<string, unknown>
  ): boolean {
    switch (adapterType) {
      case AdapterType.Feishu: {
        const feishuCfg = this.secrets.feishu;
        if (!feishuCfg) return true; // 未配置密钥则跳过验证（兼容模式）

        // 优先使用签名验证（更安全）
        if (feishuCfg.encryptKey) {
          const timestamp = req.headers['x-lark-request-timestamp'] as string;
          const nonce = req.headers['x-lark-request-nonce'] as string;
          const signature = req.headers['x-lark-signature'] as string;
          return verifyFeishuSignature(timestamp, nonce, feishuCfg.encryptKey, bodyStr, signature);
        }

        // 降级：token 验证
        if (feishuCfg.verificationToken) {
          const bodyToken = (data.token as string) || (data.verification_token as string);
          return verifyFeishuToken(bodyToken, feishuCfg.verificationToken);
        }

        return true;
      }

      case AdapterType.DingTalk: {
        const ddCfg = this.secrets.dingtalk;
        if (!ddCfg?.robotSecret) return true;

        const timestamp = req.headers['timestamp'] as string;
        const signature = req.headers['sign'] as string;
        return verifyDingTalkSignature(timestamp, ddCfg.robotSecret, signature);
      }

      case AdapterType.WeCom: {
        const wecomCfg = this.secrets.wecom;
        if (!wecomCfg?.token) return true;

        const url = new URL(req.url || '/', `http://${req.headers.host}`);
        const timestamp = url.searchParams.get('timestamp') || '';
        const nonce = url.searchParams.get('nonce') || '';
        const msgSignature = url.searchParams.get('msg_signature') || '';
        const echostr = url.searchParams.get('echostr') || '';
        return verifyWeComSignature(wecomCfg.token, timestamp, nonce, echostr, msgSignature);
      }

      default:
        return true;
    }
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
