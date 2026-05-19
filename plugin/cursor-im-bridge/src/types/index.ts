export enum AdapterType {
  Feishu = 'feishu',
  WeCom = 'wecom',
  Telegram = 'telegram',
  DingTalk = 'dingtalk',
  Custom = 'custom',
}

export enum MessageDirection {
  Inbound = 'inbound',
  Outbound = 'outbound',
}

export enum ConnectionStatus {
  Disconnected = 'disconnected',
  Connecting = 'connecting',
  Connected = 'connected',
  Error = 'error',
}

export interface IMMessage {
  id: string;
  adapterId: string;
  adapterType: AdapterType;
  direction: MessageDirection;
  channelId: string;
  channelName?: string;
  senderId: string;
  senderName?: string;
  content: string;
  contentType: 'text' | 'markdown' | 'image' | 'file' | 'card';
  timestamp: number;
  replyTo?: string;
  metadata?: Record<string, unknown>;
}

export interface IMChannel {
  id: string;
  adapterId: string;
  name: string;
  type: 'private' | 'group' | 'channel';
  members?: number;
}

export interface AdapterConfig {
  type: AdapterType;
  name: string;
  enabled: boolean;
  config: Record<string, unknown>;
}

export interface AdapterEvents {
  message: (msg: IMMessage) => void;
  statusChange: (status: ConnectionStatus) => void;
  error: (error: Error) => void;
  channelUpdate: (channels: IMChannel[]) => void;
}

export interface IIMAdapter {
  readonly id: string;
  readonly type: AdapterType;
  readonly displayName: string;
  readonly status: ConnectionStatus;

  connect(): Promise<void>;
  disconnect(): Promise<void>;
  sendMessage(channelId: string, content: string, contentType?: IMMessage['contentType']): Promise<IMMessage>;
  getChannels(): Promise<IMChannel[]>;
  getHistory(channelId: string, limit?: number, before?: string): Promise<IMMessage[]>;

  on<K extends keyof AdapterEvents>(event: K, listener: AdapterEvents[K]): void;
  off<K extends keyof AdapterEvents>(event: K, listener: AdapterEvents[K]): void;
}

export interface WebhookPayload {
  adapter: AdapterType;
  event: string;
  data: unknown;
  timestamp: number;
  signature?: string;
}
