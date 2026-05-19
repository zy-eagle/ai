import { AdapterConfig, AdapterType, IIMAdapter } from '../types';
import {
  FeishuAdapter,
  WeComAdapter,
  TelegramAdapter,
  DingTalkAdapter,
  CustomAdapter,
} from '../adapters';

type AdapterFactory = (config: AdapterConfig) => IIMAdapter;

const builtinFactories: Record<string, AdapterFactory> = {
  [AdapterType.Feishu]: (config) => new FeishuAdapter(config),
  [AdapterType.WeCom]: (config) => new WeComAdapter(config),
  [AdapterType.Telegram]: (config) => new TelegramAdapter(config),
  [AdapterType.DingTalk]: (config) => new DingTalkAdapter(config),
  [AdapterType.Custom]: (config) => new CustomAdapter(config),
};

export class AdapterRegistry {
  private factories: Map<string, AdapterFactory> = new Map();
  private instances: Map<string, IIMAdapter> = new Map();

  constructor() {
    for (const [type, factory] of Object.entries(builtinFactories)) {
      this.factories.set(type, factory);
    }
  }

  registerFactory(type: string, factory: AdapterFactory): void {
    this.factories.set(type, factory);
  }

  create(config: AdapterConfig): IIMAdapter {
    const factory = this.factories.get(config.type);
    if (!factory) {
      throw new Error(`Unknown adapter type: ${config.type}. Available: ${[...this.factories.keys()].join(', ')}`);
    }

    const adapter = factory(config);
    this.instances.set(adapter.id, adapter);
    return adapter;
  }

  get(id: string): IIMAdapter | undefined {
    return this.instances.get(id);
  }

  getAll(): IIMAdapter[] {
    return [...this.instances.values()];
  }

  remove(id: string): boolean {
    return this.instances.delete(id);
  }

  clear(): void {
    this.instances.clear();
  }

  getAvailableTypes(): string[] {
    return [...this.factories.keys()];
  }
}
