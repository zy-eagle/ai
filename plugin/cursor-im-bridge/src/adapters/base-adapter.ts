import { EventEmitter } from 'events';
import {
  IIMAdapter,
  AdapterType,
  ConnectionStatus,
  MessageDirection,
  IMMessage,
  IMChannel,
  AdapterEvents,
  AdapterConfig,
} from '../types';

export abstract class BaseAdapter extends EventEmitter implements IIMAdapter {
  abstract readonly type: AdapterType;
  abstract readonly displayName: string;

  readonly id: string;
  protected _status: ConnectionStatus = ConnectionStatus.Disconnected;
  protected _config: AdapterConfig;
  protected abortController: AbortController | null = null;

  constructor(config: AdapterConfig) {
    super();
    this.id = `${config.type}-${Date.now()}`;
    this._config = config;
  }

  get status(): ConnectionStatus {
    return this._status;
  }

  protected setStatus(status: ConnectionStatus): void {
    this._status = status;
    this.emit('statusChange', status);
  }

  async connect(): Promise<void> {
    if (this._status === ConnectionStatus.Connected) {
      return;
    }
    this.abortController = new AbortController();
    this.setStatus(ConnectionStatus.Connecting);
    try {
      await this.doConnect();
      this.setStatus(ConnectionStatus.Connected);
    } catch (err) {
      this.setStatus(ConnectionStatus.Error);
      this.emit('error', err instanceof Error ? err : new Error(String(err)));
      throw err;
    }
  }

  async disconnect(): Promise<void> {
    if (this._status === ConnectionStatus.Disconnected) {
      return;
    }
    this.abortController?.abort();
    this.abortController = null;
    try {
      await this.doDisconnect();
    } finally {
      this.setStatus(ConnectionStatus.Disconnected);
    }
  }

  on<K extends keyof AdapterEvents>(event: K, listener: AdapterEvents[K]): this {
    return super.on(event, listener as (...args: unknown[]) => void);
  }

  off<K extends keyof AdapterEvents>(event: K, listener: AdapterEvents[K]): this {
    return super.off(event, listener as (...args: unknown[]) => void);
  }

  abstract sendMessage(
    channelId: string,
    content: string,
    contentType?: IMMessage['contentType']
  ): Promise<IMMessage>;

  abstract getChannels(): Promise<IMChannel[]>;

  abstract getHistory(
    channelId: string,
    limit?: number,
    before?: string
  ): Promise<IMMessage[]>;

  protected abstract doConnect(): Promise<void>;
  protected abstract doDisconnect(): Promise<void>;

  protected createOutboundMessage(
    channelId: string,
    content: string,
    contentType: IMMessage['contentType'] = 'text'
  ): IMMessage {
    return {
      id: `${this.id}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      adapterId: this.id,
      adapterType: this.type,
      direction: MessageDirection.Outbound,
      channelId,
      content,
      contentType,
      senderId: 'self',
      senderName: 'Cursor IM Bridge',
      timestamp: Date.now(),
    };
  }
}
