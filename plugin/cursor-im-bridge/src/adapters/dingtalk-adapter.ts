import { BaseAdapter } from './base-adapter';
import {
  AdapterType,
  AdapterConfig,
  IMMessage,
  IMChannel,
  MessageDirection,
} from '../types';
import { HttpClient } from '../utils/http-client';
import WebSocket from 'ws';

interface DingTalkConfig {
  appKey: string;
  appSecret: string;
  /**
   * 连接模式:
   * - 'stream' (默认): 使用钉钉 Stream 长连接，不需要公网 URL
   * - 'webhook': 传统回调模式，需要公网 URL
   */
  mode?: 'stream' | 'webhook';
  /** 群机器人 Webhook URL (用于主动推送) */
  robotWebhookUrl?: string;
  /** 群机器人加签密钥 */
  robotSecret?: string;
  /** 订阅的事件类型 (stream 模式) */
  subscriptions?: Array<{ type: string; topic: string }>;
}

export class DingTalkAdapter extends BaseAdapter {
  readonly type = AdapterType.DingTalk;
  readonly displayName = '钉钉 (DingTalk)';

  private accessToken: string = '';
  private tokenExpiry: number = 0;
  private http: HttpClient;
  private ws: WebSocket | null = null;
  private wsReconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private tokenRefreshTimer: ReturnType<typeof setInterval> | null = null;
  private reconnectCount: number = 0;

  private get dingtalkConfig(): DingTalkConfig {
    return this._config.config as unknown as DingTalkConfig;
  }

  private get mode(): 'stream' | 'webhook' {
    return this.dingtalkConfig.mode || 'stream';
  }

  constructor(config: AdapterConfig) {
    super(config);
    this.http = new HttpClient('https://oapi.dingtalk.com', {
      timeout: 10000,
    });
  }

  protected async doConnect(): Promise<void> {
    await this.refreshToken();

    if (this.mode === 'stream') {
      await this.connectStream();
    }

    this.startTokenRefresh();
  }

  protected async doDisconnect(): Promise<void> {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    if (this.wsReconnectTimer) {
      clearTimeout(this.wsReconnectTimer);
      this.wsReconnectTimer = null;
    }
    if (this.tokenRefreshTimer) {
      clearInterval(this.tokenRefreshTimer);
      this.tokenRefreshTimer = null;
    }
    this.accessToken = '';
  }

  async sendMessage(
    channelId: string,
    content: string,
    contentType: IMMessage['contentType'] = 'text'
  ): Promise<IMMessage> {
    if (this.dingtalkConfig.robotWebhookUrl) {
      return this.sendViaRobotWebhook(channelId, content, contentType);
    }

    await this.ensureToken();
    const body: Record<string, unknown> = {
      msgtype: contentType === 'markdown' ? 'markdown' : 'text',
      chatid: channelId,
    };

    if (contentType === 'markdown') {
      body.markdown = { title: 'Message', text: content };
    } else {
      body.text = { content };
    }

    await this.http.post(
      `/chat/send?access_token=${this.accessToken}`,
      body
    );

    return this.createOutboundMessage(channelId, content, contentType);
  }

  async getChannels(): Promise<IMChannel[]> {
    await this.ensureToken();
    return [];
  }

  async getHistory(
    _channelId: string,
    _limit?: number,
    _before?: string
  ): Promise<IMMessage[]> {
    return [];
  }

  /** 处理来自 webhook 回调的事件 (仅 webhook 模式) */
  handleWebhookEvent(event: Record<string, unknown>): void {
    this.processEvent(event);
  }

  // ─── Stream 长连接 (钉钉 Stream Mode) ─────────────────────────────

  private async connectStream(): Promise<void> {
    const endpoint = await this.getStreamEndpoint();
    await this.establishStreamConnection(endpoint);
  }

  private async getStreamEndpoint(): Promise<{ endpoint: string; ticket: string }> {
    const apiHttp = new HttpClient('https://api.dingtalk.com', { timeout: 10000 });
    const subscriptions = this.dingtalkConfig.subscriptions || [
      { type: 'EVENT', topic: '/v1.0/im/bot/messages/get' },
      { type: 'CALLBACK', topic: '/v1.0/im/bot/messages/get' },
    ];

    const resp = await apiHttp.post<{
      endpoint: string;
      ticket: string;
    }>(
      '/v1.0/gateway/connections/open',
      {
        clientId: this.dingtalkConfig.appKey,
        clientSecret: this.dingtalkConfig.appSecret,
        subscriptions,
        ua: 'CursorIMBridge/0.1.0',
      }
    );

    if (!resp.endpoint) {
      throw new Error('DingTalk Stream: failed to get connection endpoint');
    }

    return resp;
  }

  private async establishStreamConnection(conn: { endpoint: string; ticket: string }): Promise<void> {
    return new Promise((resolve, reject) => {
      const url = `${conn.endpoint}?ticket=${encodeURIComponent(conn.ticket)}`;
      const ws = new WebSocket(url);

      ws.on('open', () => {
        this.ws = ws;
        this.reconnectCount = 0;
        resolve();
      });

      ws.on('message', (data) => {
        try {
          const payload = JSON.parse(data.toString());
          this.handleStreamMessage(payload, ws);
        } catch (err) {
          this.emit('error', new Error(`DingTalk Stream parse error: ${err}`));
        }
      });

      ws.on('close', () => {
        this.ws = null;
        if (!this.abortController?.signal.aborted) {
          this.scheduleReconnect();
        }
      });

      ws.on('error', (err) => {
        if (!this.ws) {
          reject(err);
        } else {
          this.emit('error', err);
        }
      });

      ws.on('ping', () => {
        ws.pong();
      });
    });
  }

  private handleStreamMessage(payload: Record<string, unknown>, ws: WebSocket): void {
    const type = payload.type as string;

    // 钉钉 Stream 协议：系统消息
    if (type === 'SYSTEM') {
      const headers = payload.headers as Record<string, string> | undefined;
      if (headers?.topic === 'ping') {
        ws.send(JSON.stringify({
          code: 200,
          headers: payload.headers,
          message: 'OK',
          data: payload.data,
        }));
      }
      return;
    }

    // 钉钉 Stream 协议：事件/回调消息
    if (type === 'EVENT' || type === 'CALLBACK') {
      const data = payload.data as string | undefined;
      if (data) {
        try {
          const event = JSON.parse(data);
          this.processEvent(event);
        } catch {
          // ignore parse failures
        }
      }

      // ACK 确认
      ws.send(JSON.stringify({
        code: 200,
        headers: payload.headers,
        message: 'OK',
        data: '',
      }));
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectCount >= 10) {
      this.emit('error', new Error('DingTalk Stream: max reconnection attempts exceeded'));
      return;
    }

    this.reconnectCount++;
    const delay = Math.min(1000 * Math.pow(2, this.reconnectCount) + Math.random() * 1000, 30000);

    this.wsReconnectTimer = setTimeout(async () => {
      try {
        await this.connectStream();
      } catch {
        this.scheduleReconnect();
      }
    }, delay);
  }

  // ─── 通用事件处理 ──────────────────────────────────────────────────

  private processEvent(event: Record<string, unknown>): void {
    const msgtype = (event.msgtype as string) || (event.msgType as string);
    const text = event.text as Record<string, unknown> | undefined;
    const content = event.content as Record<string, unknown> | undefined;

    const textContent = (text?.content as string) || (content?.content as string) || '';

    if (textContent || msgtype === 'text') {
      const msg: IMMessage = {
        id: (event.msgId as string) || (event.chatbotCorpId as string) || `dingtalk-${Date.now()}`,
        adapterId: this.id,
        adapterType: AdapterType.DingTalk,
        direction: MessageDirection.Inbound,
        channelId: (event.conversationId as string) || 'unknown',
        channelName: event.conversationTitle as string,
        senderId: (event.senderStaffId as string) || (event.senderId as string) || 'unknown',
        senderName: event.senderNick as string,
        content: textContent,
        contentType: 'text',
        timestamp: parseInt(event.createAt as string) || Date.now(),
      };
      this.emit('message', msg);
    }
  }

  // ─── 发送消息 ──────────────────────────────────────────────────────

  private async sendViaRobotWebhook(
    channelId: string,
    content: string,
    contentType: IMMessage['contentType']
  ): Promise<IMMessage> {
    let url = this.dingtalkConfig.robotWebhookUrl!;

    if (this.dingtalkConfig.robotSecret) {
      const timestamp = Date.now();
      const sign = await this.computeSign(timestamp, this.dingtalkConfig.robotSecret);
      url += `&timestamp=${timestamp}&sign=${encodeURIComponent(sign)}`;
    }

    const robotHttp = new HttpClient('', { timeout: 10000 });
    const body: Record<string, unknown> = {
      msgtype: contentType === 'markdown' ? 'markdown' : 'text',
    };

    if (contentType === 'markdown') {
      body.markdown = { title: 'Message', text: content };
    } else {
      body.text = { content };
    }

    await robotHttp.postAbsolute(url, body);
    return this.createOutboundMessage(channelId, content, contentType);
  }

  private async computeSign(timestamp: number, secret: string): Promise<string> {
    const crypto = await import('crypto');
    const stringToSign = `${timestamp}\n${secret}`;
    const hmac = crypto.createHmac('sha256', secret);
    hmac.update(stringToSign);
    return hmac.digest('base64');
  }

  // ─── Token 管理 ────────────────────────────────────────────────────

  private async refreshToken(): Promise<void> {
    const resp = await this.http.get<{
      errcode: number;
      errmsg: string;
      access_token: string;
      expires_in: number;
    }>(`/gettoken?appkey=${this.dingtalkConfig.appKey}&appsecret=${this.dingtalkConfig.appSecret}`);

    if (resp.errcode !== 0) {
      throw new Error(`DingTalk auth failed: ${resp.errmsg}`);
    }

    this.accessToken = resp.access_token;
    this.tokenExpiry = Date.now() + (resp.expires_in - 300) * 1000;
  }

  private async ensureToken(): Promise<void> {
    if (Date.now() >= this.tokenExpiry) {
      await this.refreshToken();
    }
  }

  private startTokenRefresh(): void {
    this.tokenRefreshTimer = setInterval(async () => {
      try {
        await this.ensureToken();
      } catch (err) {
        this.emit('error', err instanceof Error ? err : new Error(String(err)));
      }
    }, 60000);
  }
}
