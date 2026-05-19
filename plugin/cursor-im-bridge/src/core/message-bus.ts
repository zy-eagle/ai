import { EventEmitter } from 'events';
import { IMMessage, IIMAdapter, ConnectionStatus } from '../types';

export interface MessageBusEvents {
  message: (msg: IMMessage) => void;
  adapterStatusChange: (adapterId: string, status: ConnectionStatus) => void;
  adapterError: (adapterId: string, error: Error) => void;
}

/**
 * 消息总线 — 统一收发所有适配器的消息
 * 所有入站消息汇聚至此，出站消息通过此路由到目标适配器
 */
export class MessageBus extends EventEmitter {
  private adapters: Map<string, IIMAdapter> = new Map();
  private messageHistory: IMMessage[] = [];
  private maxHistory: number = 500;

  attach(adapter: IIMAdapter): void {
    this.adapters.set(adapter.id, adapter);

    adapter.on('message', (msg: IMMessage) => {
      this.messageHistory.push(msg);
      if (this.messageHistory.length > this.maxHistory) {
        this.messageHistory.shift();
      }
      this.emit('message', msg);
    });

    adapter.on('statusChange', (status: ConnectionStatus) => {
      this.emit('adapterStatusChange', adapter.id, status);
    });

    adapter.on('error', (error: Error) => {
      this.emit('adapterError', adapter.id, error);
    });
  }

  detach(adapterId: string): void {
    this.adapters.delete(adapterId);
  }

  async send(adapterId: string, channelId: string, content: string, contentType?: IMMessage['contentType']): Promise<IMMessage> {
    const adapter = this.adapters.get(adapterId);
    if (!adapter) {
      throw new Error(`Adapter not found: ${adapterId}`);
    }
    if (adapter.status !== ConnectionStatus.Connected) {
      throw new Error(`Adapter ${adapterId} is not connected (status: ${adapter.status})`);
    }

    const msg = await adapter.sendMessage(channelId, content, contentType);
    this.messageHistory.push(msg);
    if (this.messageHistory.length > this.maxHistory) {
      this.messageHistory.shift();
    }
    return msg;
  }

  async broadcast(channelIds: Array<{ adapterId: string; channelId: string }>, content: string): Promise<IMMessage[]> {
    const results = await Promise.allSettled(
      channelIds.map(({ adapterId, channelId }) => this.send(adapterId, channelId, content))
    );

    return results
      .filter((r): r is PromiseFulfilledResult<IMMessage> => r.status === 'fulfilled')
      .map((r) => r.value);
  }

  getHistory(adapterId?: string, channelId?: string, limit: number = 50): IMMessage[] {
    let filtered = this.messageHistory;
    if (adapterId) {
      filtered = filtered.filter((m) => m.adapterId === adapterId);
    }
    if (channelId) {
      filtered = filtered.filter((m) => m.channelId === channelId);
    }
    return filtered.slice(-limit);
  }

  getConnectedAdapters(): IIMAdapter[] {
    return [...this.adapters.values()].filter(
      (a) => a.status === ConnectionStatus.Connected
    );
  }

  dispose(): void {
    for (const adapterId of this.adapters.keys()) {
      this.detach(adapterId);
    }
    this.removeAllListeners();
    this.messageHistory = [];
  }
}
