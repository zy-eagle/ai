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

interface FeishuConfig {
  appId: string;
  appSecret: string;
  /**
   * 连接模式:
   * - 'websocket' (默认): 使用飞书长连接，不需要公网 URL
   * - 'webhook': 传统回调模式，需要公网 URL
   */
  mode?: 'websocket' | 'webhook';
  /** Webhook 验证 token (仅 webhook 模式) */
  verificationToken?: string;
  /** 事件加密 key (仅 webhook 模式) */
  encryptKey?: string;
}

interface FeishuTokenResponse {
  code: number;
  msg: string;
  tenant_access_token: string;
  expire: number;
}

interface FeishuWSEndpointResponse {
  code: number;
  msg: string;
  data: {
    URL: string;
    ClientConfig: {
      ReconnectCount: number;
      ReconnectInterval: number;
      ReconnectNonce: number;
      PingInterval: number;
    };
  };
}

export class FeishuAdapter extends BaseAdapter {
  readonly type = AdapterType.Feishu;
  readonly displayName = '飞书 (Feishu)';

  private accessToken: string = '';
  private tokenExpiry: number = 0;
  private http: HttpClient;
  private ws: WebSocket | null = null;
  private wsReconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private tokenRefreshTimer: ReturnType<typeof setInterval> | null = null;
  private reconnectCount: number = 0;
  private maxReconnects: number = 10;

  private get feishuConfig(): FeishuConfig {
    return this._config.config as unknown as FeishuConfig;
  }

  private get mode(): 'websocket' | 'webhook' {
    return this.feishuConfig.mode || 'websocket';
  }

  constructor(config: AdapterConfig) {
    super(config);
    this.http = new HttpClient('https://open.feishu.cn/open-apis', {
      timeout: 10000,
    });
  }

  protected async doConnect(): Promise<void> {
    await this.refreshToken();

    if (this.mode === 'websocket') {
      await this.connectWebSocket();
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
  }

  async sendMessage(
    channelId: string,
    content: string,
    contentType: IMMessage['contentType'] = 'text'
  ): Promise<IMMessage> {
    await this.ensureToken();
    const body = this.buildMessageBody(content, contentType);

    await this.http.post(
      `/im/v1/messages?receive_id_type=chat_id`,
      {
        receive_id: channelId,
        msg_type: contentType === 'markdown' ? 'interactive' : 'text',
        content: JSON.stringify(body),
      },
      { Authorization: `Bearer ${this.accessToken}` }
    );

    return this.createOutboundMessage(channelId, content, contentType);
  }

  async getChannels(): Promise<IMChannel[]> {
    await this.ensureToken();
    const resp = await this.http.get<{
      data: { items: Array<{ chat_id: string; name: string; chat_type: string; member_count: number }> };
    }>('/im/v1/chats', {
      Authorization: `Bearer ${this.accessToken}`,
    });

    return (resp.data?.items || []).map((item) => ({
      id: item.chat_id,
      adapterId: this.id,
      name: item.name || 'Unnamed',
      type: item.chat_type === 'p2p' ? 'private' as const : 'group' as const,
      members: item.member_count,
    }));
  }

  async getHistory(
    channelId: string,
    limit: number = 20,
    _before?: string
  ): Promise<IMMessage[]> {
    await this.ensureToken();
    const resp = await this.http.get<{
      data: {
        items: Array<{
          message_id: string;
          sender: { sender_id: { user_id: string }; sender_type: string };
          body: { content: string };
          create_time: string;
          chat_id: string;
        }>;
      };
    }>(`/im/v1/messages?container_id_type=chat&container_id=${channelId}&page_size=${limit}`, {
      Authorization: `Bearer ${this.accessToken}`,
    });

    return (resp.data?.items || []).map((item) => ({
      id: item.message_id,
      adapterId: this.id,
      adapterType: AdapterType.Feishu,
      direction: MessageDirection.Inbound,
      channelId: item.chat_id,
      senderId: item.sender?.sender_id?.user_id || 'unknown',
      content: this.parseMessageContent(item.body?.content),
      contentType: 'text' as const,
      timestamp: parseInt(item.create_time) || Date.now(),
    }));
  }

  /** 处理来自 webhook 的事件 (仅 webhook 模式) */
  handleWebhookEvent(event: Record<string, unknown>): void {
    this.processEvent(event);
  }

  // ─── WebSocket 长连接 ───────────────────────────────────────────────

  private async connectWebSocket(): Promise<void> {
    const endpoint = await this.getWSEndpoint();
    await this.establishWSConnection(endpoint);
  }

  private async getWSEndpoint(): Promise<FeishuWSEndpointResponse['data']> {
    const resp = await this.http.post<FeishuWSEndpointResponse>(
      '/callback/ws/endpoint',
      {},
      { Authorization: `Bearer ${this.accessToken}` }
    );

    if (resp.code !== 0) {
      throw new Error(`Feishu WS endpoint failed: ${resp.msg}`);
    }

    this.maxReconnects = resp.data.ClientConfig.ReconnectCount || 10;
    return resp.data;
  }

  private async establishWSConnection(endpoint: FeishuWSEndpointResponse['data']): Promise<void> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(endpoint.URL, {
        headers: {
          'User-Agent': 'CursorIMBridge/0.1.0',
        },
      });

      ws.on('open', () => {
        this.ws = ws;
        this.reconnectCount = 0;
        resolve();
      });

      ws.on('message', (data) => {
        try {
          const payload = JSON.parse(data.toString());
          this.handleWSMessage(payload);
        } catch (err) {
          this.emit('error', new Error(`WS parse error: ${err}`));
        }
      });

      ws.on('close', (code, reason) => {
        this.ws = null;
        if (!this.abortController?.signal.aborted) {
          this.scheduleReconnect(endpoint);
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

  private handleWSMessage(payload: Record<string, unknown>): void {
    const type = payload.type as string;

    if (type === 'pong' || type === 'ping') {
      return;
    }

    // 飞书 WS 事件格式
    if (payload.header || payload.event) {
      this.processEvent(payload);
    }

    // 封装格式：data 字段包含实际事件
    if (payload.data && typeof payload.data === 'object') {
      this.processEvent(payload.data as Record<string, unknown>);
    }
  }

  private scheduleReconnect(endpoint: FeishuWSEndpointResponse['data']): void {
    if (this.reconnectCount >= this.maxReconnects) {
      this.emit('error', new Error('Feishu WS max reconnection attempts exceeded'));
      this.setStatus('error' as never);
      return;
    }

    this.reconnectCount++;
    const delay = Math.min(1000 * Math.pow(2, this.reconnectCount) + Math.random() * 1000, 30000);

    this.wsReconnectTimer = setTimeout(async () => {
      try {
        const newEndpoint = await this.getWSEndpoint();
        await this.establishWSConnection(newEndpoint);
      } catch {
        this.scheduleReconnect(endpoint);
      }
    }, delay);
  }

  // ─── 通用事件处理 ───────────────────────────────────────────────────

  private processEvent(event: Record<string, unknown>): void {
    const header = event.header as Record<string, unknown> | undefined;
    const eventType = (header?.event_type as string) || (event.type as string);

    if (eventType === 'im.message.receive_v1') {
      const eventData = event.event as Record<string, unknown>;
      const message = eventData?.message as Record<string, unknown>;
      const sender = eventData?.sender as Record<string, unknown>;

      if (message && sender) {
        const msg: IMMessage = {
          id: message.message_id as string,
          adapterId: this.id,
          adapterType: AdapterType.Feishu,
          direction: MessageDirection.Inbound,
          channelId: message.chat_id as string,
          senderId: (sender.sender_id as Record<string, string>)?.user_id || 'unknown',
          senderName: sender.sender_type as string,
          content: this.parseMessageContent(message.content as string),
          contentType: 'text',
          timestamp: parseInt(message.create_time as string) || Date.now(),
        };
        this.emit('message', msg);
      }
    }
  }

  // ─── Token 管理 ────────────────────────────────────────────────────

  private async refreshToken(): Promise<void> {
    const resp = await this.http.post<FeishuTokenResponse>(
      '/auth/v3/tenant_access_token/internal',
      {
        app_id: this.feishuConfig.appId,
        app_secret: this.feishuConfig.appSecret,
      }
    );

    if (resp.code !== 0) {
      throw new Error(`Feishu auth failed: ${resp.msg}`);
    }

    this.accessToken = resp.tenant_access_token;
    this.tokenExpiry = Date.now() + (resp.expire - 300) * 1000;
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

  // ─── 工具方法 ──────────────────────────────────────────────────────

  private buildMessageBody(content: string, contentType: IMMessage['contentType']): unknown {
    if (contentType === 'markdown' || contentType === 'card') {
      return {
        elements: [{ tag: 'markdown', content }],
      };
    }
    return { text: content };
  }

  private parseMessageContent(raw: string | undefined): string {
    if (!raw) return '';
    try {
      const parsed = JSON.parse(raw);
      return parsed.text || parsed.content || raw;
    } catch {
      return raw;
    }
  }
}
