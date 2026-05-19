import { BaseAdapter } from './base-adapter';
import {
  AdapterType,
  AdapterConfig,
  IMMessage,
  IMChannel,
  MessageDirection,
} from '../types';
import { HttpClient } from '../utils/http-client';

/**
 * 自定义 IM 适配器 — 适用于公司内部 IM 工具
 *
 * 通过配置 HTTP 端点和认证方式实现与任意 IM 系统的对接：
 * - 发送消息: POST sendMessageUrl
 * - 接收消息: Webhook 回调 / 长轮询
 * - 获取通道: GET getChannelsUrl
 * - 获取历史: GET getHistoryUrl
 */
interface CustomIMConfig {
  /** 基础 URL */
  baseUrl: string;
  /** 发送消息端点 */
  sendMessagePath: string;
  /** 获取通道列表端点 */
  getChannelsPath?: string;
  /** 获取历史消息端点 */
  getHistoryPath?: string;
  /** 认证方式 */
  auth: {
    type: 'bearer' | 'apikey' | 'basic' | 'custom-header';
    /** Bearer token / API Key / Base64(user:pass) */
    credentials: string;
    /** 自定义 header 名称 (type=custom-header 时) */
    headerName?: string;
  };
  /** 消息体字段映射 */
  fieldMapping?: {
    channelIdField?: string;
    contentField?: string;
    senderIdField?: string;
    senderNameField?: string;
    messageIdField?: string;
    timestampField?: string;
  };
  /** 轮询间隔 (ms), 0 表示仅 webhook */
  pollingInterval?: number;
  /** 轮询端点 */
  pollingPath?: string;
}

export class CustomAdapter extends BaseAdapter {
  readonly type = AdapterType.Custom;
  readonly displayName: string;

  private http: HttpClient;
  private pollingTimer: ReturnType<typeof setInterval> | null = null;

  private get customConfig(): CustomIMConfig {
    return this._config.config as unknown as CustomIMConfig;
  }

  constructor(config: AdapterConfig) {
    super(config);
    this.displayName = config.name || '自定义 IM (Custom)';
    const cfg = config.config as unknown as CustomIMConfig;
    this.http = new HttpClient(cfg.baseUrl, {
      timeout: 10000,
      defaultHeaders: this.buildAuthHeaders(cfg.auth),
    });
  }

  protected async doConnect(): Promise<void> {
    if (this.customConfig.getChannelsPath) {
      await this.getChannels();
    }

    if (this.customConfig.pollingInterval && this.customConfig.pollingPath) {
      this.startPolling();
    }
  }

  protected async doDisconnect(): Promise<void> {
    if (this.pollingTimer) {
      clearInterval(this.pollingTimer);
      this.pollingTimer = null;
    }
  }

  async sendMessage(
    channelId: string,
    content: string,
    contentType: IMMessage['contentType'] = 'text'
  ): Promise<IMMessage> {
    const mapping = this.customConfig.fieldMapping || {};
    const body: Record<string, unknown> = {
      [mapping.channelIdField || 'channel_id']: channelId,
      [mapping.contentField || 'content']: content,
      content_type: contentType,
    };

    await this.http.post(this.customConfig.sendMessagePath, body);
    return this.createOutboundMessage(channelId, content, contentType);
  }

  async getChannels(): Promise<IMChannel[]> {
    if (!this.customConfig.getChannelsPath) return [];

    const resp = await this.http.get<{
      data: Array<Record<string, unknown>>;
    }>(this.customConfig.getChannelsPath);

    return (resp.data || []).map((item) => ({
      id: String(item.id || item.channel_id),
      adapterId: this.id,
      name: String(item.name || item.title || 'Unknown'),
      type: 'group' as const,
    }));
  }

  async getHistory(
    channelId: string,
    limit: number = 20,
    _before?: string
  ): Promise<IMMessage[]> {
    if (!this.customConfig.getHistoryPath) return [];

    const mapping = this.customConfig.fieldMapping || {};
    const resp = await this.http.get<{
      data: Array<Record<string, unknown>>;
    }>(`${this.customConfig.getHistoryPath}?channel_id=${channelId}&limit=${limit}`);

    return (resp.data || []).map((item) => ({
      id: String(item[mapping.messageIdField || 'id']),
      adapterId: this.id,
      adapterType: AdapterType.Custom,
      direction: MessageDirection.Inbound,
      channelId,
      senderId: String(item[mapping.senderIdField || 'sender_id'] || 'unknown'),
      senderName: item[mapping.senderNameField || 'sender_name'] as string,
      content: String(item[mapping.contentField || 'content'] || ''),
      contentType: 'text' as const,
      timestamp: Number(item[mapping.timestampField || 'timestamp']) || Date.now(),
    }));
  }

  /** 处理来自 webhook 的事件 */
  handleWebhookEvent(event: Record<string, unknown>): void {
    const mapping = this.customConfig.fieldMapping || {};
    const msg: IMMessage = {
      id: String(event[mapping.messageIdField || 'id'] || `custom-${Date.now()}`),
      adapterId: this.id,
      adapterType: AdapterType.Custom,
      direction: MessageDirection.Inbound,
      channelId: String(event[mapping.channelIdField || 'channel_id'] || 'unknown'),
      senderId: String(event[mapping.senderIdField || 'sender_id'] || 'unknown'),
      senderName: event[mapping.senderNameField || 'sender_name'] as string,
      content: String(event[mapping.contentField || 'content'] || ''),
      contentType: 'text',
      timestamp: Number(event[mapping.timestampField || 'timestamp']) || Date.now(),
    };
    this.emit('message', msg);
  }

  private startPolling(): void {
    const poll = async () => {
      if (this.abortController?.signal.aborted) return;
      try {
        const resp = await this.http.get<{
          data: Array<Record<string, unknown>>;
        }>(this.customConfig.pollingPath!);

        for (const item of resp.data || []) {
          this.handleWebhookEvent(item);
        }
      } catch (err) {
        if (!this.abortController?.signal.aborted) {
          this.emit('error', err instanceof Error ? err : new Error(String(err)));
        }
      }
    };

    this.pollingTimer = setInterval(poll, this.customConfig.pollingInterval || 5000);
  }

  private buildAuthHeaders(auth: CustomIMConfig['auth']): Record<string, string> {
    switch (auth.type) {
      case 'bearer':
        return { Authorization: `Bearer ${auth.credentials}` };
      case 'apikey':
        return { 'X-API-Key': auth.credentials };
      case 'basic':
        return { Authorization: `Basic ${auth.credentials}` };
      case 'custom-header':
        return { [auth.headerName || 'X-Auth']: auth.credentials };
      default:
        return {};
    }
  }
}
