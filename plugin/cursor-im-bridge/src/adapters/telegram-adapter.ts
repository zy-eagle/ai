import { BaseAdapter } from './base-adapter';
import {
  AdapterType,
  AdapterConfig,
  IMMessage,
  IMChannel,
  MessageDirection,
} from '../types';
import { HttpClient } from '../utils/http-client';

interface TelegramConfig {
  botToken: string;
  /** Webhook URL (如设置则使用 webhook 模式，否则使用 long polling) */
  webhookUrl?: string;
  /** 允许的 chat IDs (安全白名单) */
  allowedChatIds?: string[];
}

export class TelegramAdapter extends BaseAdapter {
  readonly type = AdapterType.Telegram;
  readonly displayName = 'Telegram';

  private http: HttpClient;
  private pollingTimer: ReturnType<typeof setInterval> | null = null;
  private lastUpdateId: number = 0;

  private get telegramConfig(): TelegramConfig {
    return this._config.config as unknown as TelegramConfig;
  }

  constructor(config: AdapterConfig) {
    super(config);
    const token = (config.config as Record<string, unknown>).botToken as string;
    this.http = new HttpClient(`https://api.telegram.org/bot${token}`, {
      timeout: 30000,
    });
  }

  protected async doConnect(): Promise<void> {
    const me = await this.http.get<{ ok: boolean; result: { username: string } }>('/getMe');
    if (!me.ok) {
      throw new Error('Telegram bot token is invalid');
    }

    if (this.telegramConfig.webhookUrl) {
      await this.http.post('/setWebhook', {
        url: this.telegramConfig.webhookUrl,
      });
    } else {
      await this.http.post('/deleteWebhook', {});
      this.startLongPolling();
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
    const parseMode = contentType === 'markdown' ? 'MarkdownV2' : undefined;

    await this.http.post('/sendMessage', {
      chat_id: channelId,
      text: content,
      parse_mode: parseMode,
    });

    return this.createOutboundMessage(channelId, content, contentType);
  }

  async getChannels(): Promise<IMChannel[]> {
    // Telegram Bot API 不提供 "列出所有 chat" 的接口
    // 通道列表通过接收消息动态构建，存储在本地
    return [];
  }

  async getHistory(
    _channelId: string,
    _limit?: number,
    _before?: string
  ): Promise<IMMessage[]> {
    // Telegram Bot API 不支持拉取历史消息
    return [];
  }

  /** 处理来自 webhook 的 update */
  handleWebhookEvent(update: Record<string, unknown>): void {
    this.processUpdate(update);
  }

  private startLongPolling(): void {
    const poll = async () => {
      if (this.abortController?.signal.aborted) return;

      try {
        const resp = await this.http.get<{
          ok: boolean;
          result: Array<{
            update_id: number;
            message?: {
              message_id: number;
              from: { id: number; first_name: string; username?: string };
              chat: { id: number; type: string; title?: string };
              text?: string;
              date: number;
            };
          }>;
        }>(`/getUpdates?offset=${this.lastUpdateId + 1}&timeout=25`);

        if (resp.ok && resp.result?.length) {
          for (const update of resp.result) {
            this.lastUpdateId = update.update_id;
            this.processUpdate(update as unknown as Record<string, unknown>);
          }
        }
      } catch (err) {
        if (!this.abortController?.signal.aborted) {
          this.emit('error', err instanceof Error ? err : new Error(String(err)));
        }
      }
    };

    this.pollingTimer = setInterval(poll, 1000);
    poll();
  }

  private processUpdate(update: Record<string, unknown>): void {
    const message = update.message as Record<string, unknown> | undefined;
    if (!message) return;

    const chat = message.chat as Record<string, unknown>;
    const from = message.from as Record<string, unknown>;
    const chatId = String(chat?.id);

    if (this.telegramConfig.allowedChatIds?.length) {
      if (!this.telegramConfig.allowedChatIds.includes(chatId)) {
        return;
      }
    }

    const msg: IMMessage = {
      id: String(message.message_id),
      adapterId: this.id,
      adapterType: AdapterType.Telegram,
      direction: MessageDirection.Inbound,
      channelId: chatId,
      channelName: (chat?.title as string) || (from?.first_name as string),
      senderId: String(from?.id),
      senderName: (from?.first_name as string) || (from?.username as string),
      content: (message.text as string) || '',
      contentType: 'text',
      timestamp: ((message.date as number) || 0) * 1000,
    };

    this.emit('message', msg);
  }
}
