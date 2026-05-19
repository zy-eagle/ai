import { BaseAdapter } from './base-adapter';
import {
  AdapterType,
  AdapterConfig,
  IMMessage,
  IMChannel,
  MessageDirection,
} from '../types';
import { HttpClient } from '../utils/http-client';

interface WeComConfig {
  corpId: string;
  corpSecret: string;
  agentId: number;
  /**
   * 连接模式:
   * - 'polling' (默认): 长轮询拉取消息，不需要公网 URL
   * - 'webhook': 传统回调模式，需要公网 URL
   */
  mode?: 'polling' | 'webhook';
  /** 轮询间隔 (ms), 默认 3000 */
  pollingInterval?: number;
  /** 回调 URL 验证 token (仅 webhook 模式) */
  token?: string;
  /** 消息加解密 key (仅 webhook 模式) */
  encodingAESKey?: string;
}

export class WeComAdapter extends BaseAdapter {
  readonly type = AdapterType.WeCom;
  readonly displayName = '企业微信 (WeCom)';

  private accessToken: string = '';
  private tokenExpiry: number = 0;
  private http: HttpClient;
  private pollingTimer: ReturnType<typeof setInterval> | null = null;
  private tokenRefreshTimer: ReturnType<typeof setInterval> | null = null;
  private lastMsgId: string = '';

  private get wecomConfig(): WeComConfig {
    return this._config.config as unknown as WeComConfig;
  }

  private get mode(): 'polling' | 'webhook' {
    return this.wecomConfig.mode || 'polling';
  }

  constructor(config: AdapterConfig) {
    super(config);
    this.http = new HttpClient('https://qyapi.weixin.qq.com/cgi-bin', {
      timeout: 10000,
    });
  }

  protected async doConnect(): Promise<void> {
    await this.refreshToken();

    if (this.mode === 'polling') {
      this.startPolling();
    }

    this.startTokenRefresh();
  }

  protected async doDisconnect(): Promise<void> {
    if (this.pollingTimer) {
      clearInterval(this.pollingTimer);
      this.pollingTimer = null;
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
    await this.ensureToken();

    const msgType = contentType === 'markdown' ? 'markdown' : 'text';
    const body: Record<string, unknown> = {
      touser: channelId.startsWith('@') ? channelId.slice(1) : undefined,
      toparty: !channelId.startsWith('@') ? channelId : undefined,
      msgtype: msgType,
      agentid: this.wecomConfig.agentId,
    };

    if (msgType === 'markdown') {
      body.markdown = { content };
    } else {
      body.text = { content };
    }

    await this.http.post(
      `/message/send?access_token=${this.accessToken}`,
      body
    );

    return this.createOutboundMessage(channelId, content, contentType);
  }

  async getChannels(): Promise<IMChannel[]> {
    await this.ensureToken();
    const resp = await this.http.get<{
      errcode: number;
      department: Array<{ id: number; name: string; parentid: number }>;
    }>(`/department/list?access_token=${this.accessToken}`);

    if (resp.errcode !== 0) return [];

    return (resp.department || []).map((dept) => ({
      id: String(dept.id),
      adapterId: this.id,
      name: dept.name,
      type: 'group' as const,
    }));
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

  // ─── 长轮询模式 ────────────────────────────────────────────────────

  private startPolling(): void {
    const interval = this.wecomConfig.pollingInterval || 3000;

    const poll = async () => {
      if (this.abortController?.signal.aborted) return;

      try {
        await this.ensureToken();

        // 企微通过"同步消息"接口拉取未读消息
        // 使用 /cgi-bin/message/get_statistics 或应用的消息拉取能力
        const resp = await this.http.post<{
          errcode: number;
          errmsg: string;
          msglist?: Array<{
            msgid: string;
            action: string;
            tolist: string[];
            msgtype: string;
            text?: { content: string };
            markdown?: { content: string };
            from: string;
            send_time: string;
          }>;
        }>(`/message/get_msg_list?access_token=${this.accessToken}`, {
          cursor: this.lastMsgId || '0',
          limit: 20,
          filter: { msgtype: 'text' },
        });

        if (resp.errcode === 0 && resp.msglist?.length) {
          for (const item of resp.msglist) {
            if (item.msgid === this.lastMsgId) continue;
            this.lastMsgId = item.msgid;

            const msg: IMMessage = {
              id: item.msgid,
              adapterId: this.id,
              adapterType: AdapterType.WeCom,
              direction: MessageDirection.Inbound,
              channelId: item.tolist?.[0] || 'unknown',
              senderId: item.from || 'unknown',
              senderName: item.from,
              content: item.text?.content || item.markdown?.content || '',
              contentType: item.msgtype === 'markdown' ? 'markdown' : 'text',
              timestamp: parseInt(item.send_time) * 1000 || Date.now(),
            };
            this.emit('message', msg);
          }
        }
      } catch (err) {
        if (!this.abortController?.signal.aborted) {
          this.emit('error', err instanceof Error ? err : new Error(String(err)));
        }
      }
    };

    this.pollingTimer = setInterval(poll, interval);
    poll();
  }

  // ─── 通用事件处理 ──────────────────────────────────────────────────

  private processEvent(event: Record<string, unknown>): void {
    const msgType = event.MsgType as string;

    if (msgType === 'text' || msgType === 'markdown') {
      const msg: IMMessage = {
        id: (event.MsgId as string) || `wecom-${Date.now()}`,
        adapterId: this.id,
        adapterType: AdapterType.WeCom,
        direction: MessageDirection.Inbound,
        channelId: (event.FromUserName as string) || 'unknown',
        senderId: (event.FromUserName as string) || 'unknown',
        senderName: event.FromUserName as string,
        content: (event.Content as string) || '',
        contentType: msgType === 'markdown' ? 'markdown' : 'text',
        timestamp: parseInt(event.CreateTime as string) * 1000 || Date.now(),
      };
      this.emit('message', msg);
    }
  }

  // ─── Token 管理 ────────────────────────────────────────────────────

  private async refreshToken(): Promise<void> {
    const resp = await this.http.get<{
      errcode: number;
      errmsg: string;
      access_token: string;
      expires_in: number;
    }>(`/gettoken?corpid=${this.wecomConfig.corpId}&corpsecret=${this.wecomConfig.corpSecret}`);

    if (resp.errcode !== 0) {
      throw new Error(`WeCom auth failed: ${resp.errmsg}`);
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
