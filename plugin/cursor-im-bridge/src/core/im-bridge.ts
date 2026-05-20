import * as vscode from 'vscode';
import { AdapterConfig, ConnectionStatus, IIMAdapter, IMMessage, WebhookPayload } from '../types';
import { AdapterRegistry } from './adapter-registry';
import { MessageBus } from './message-bus';
import { WebhookServer } from './webhook-server';
import { TaskProcessor, TaskConfig, TaskResult } from '../processor';
import { SecureStore } from '../utils/secure-store';

/**
 * 核心控制器 — 管理适配器生命周期、消息路由、webhook 分发和任务处理
 */
export class IMBridge {
  private registry: AdapterRegistry;
  private messageBus: MessageBus;
  private webhookServer: WebhookServer;
  private taskProcessor: TaskProcessor | null = null;
  private outputChannel: vscode.OutputChannel;
  private statusBarItem: vscode.StatusBarItem;
  private autoReplyEnabled: boolean = false;
  private secureStore: SecureStore;

  constructor(private context: vscode.ExtensionContext) {
    this.registry = new AdapterRegistry();
    this.messageBus = new MessageBus();
    this.secureStore = new SecureStore(context.secrets);
    this.webhookServer = new WebhookServer(
      this.getWebhookPort(),
      this.getWebhookSecrets()
    );
    this.outputChannel = vscode.window.createOutputChannel('IM Bridge');
    this.statusBarItem = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Left,
      100
    );

    this.setupWebhookRouting();
    this.setupStatusBar();
  }

  async initialize(): Promise<void> {
    const configs = this.getAdapterConfigs();
    const autoConnect = vscode.workspace
      .getConfiguration('cursorImBridge')
      .get<boolean>('autoConnect', false);

    for (const config of configs) {
      if (!config.enabled) continue;
      try {
        // 优先从 OS 钥匙链（SecretStorage）加载敏感凭证，覆盖 settings.json 中的明文值
        const secureConfig = await this.mergeSecureCredentials(config);
        const adapter = this.registry.create(secureConfig);
        this.messageBus.attach(adapter);
        this.log(`Adapter registered: ${adapter.displayName} (${adapter.id})`);

        if (autoConnect) {
          await this.connectAdapter(adapter.id);
        }
      } catch (err) {
        this.log(`Failed to register adapter ${config.name}: ${err}`, 'error');
      }
    }

    await this.webhookServer.start();
    this.log(`Webhook server started on port ${this.webhookServer.getPort()}`);

    // 初始化任务处理器 (如果配置了自动回复)
    await this.initTaskProcessor();

    this.updateStatusBar();
  }

  // ─── 任务处理 (自动回复) ─────────────────────────────────────────────

  private async initTaskProcessor(): Promise<void> {
    const taskConfig = this.getTaskConfig();
    if (!taskConfig) {
      this.log('Task Processor skipped: autoReply not enabled in settings');
      return;
    }

    this.taskProcessor = new TaskProcessor(taskConfig);
    this.autoReplyEnabled = true;

    // 检查 Cursor CLI 可用性
    const status = await this.taskProcessor.checkReady();
    if (status.available) {
      this.log(`Task Processor ready. Cursor CLI version: ${status.version || 'unknown'}`);
    } else {
      this.log('Task Processor: Cursor CLI not found. Auto-reply disabled.', 'warn');
      this.autoReplyEnabled = false;
      return;
    }

    // 监听任务事件
    this.taskProcessor.on('taskStarted', ({ taskId, prompt }) => {
      this.log(`Task started [${taskId}]: ${prompt.slice(0, 100)}`);
    });

    this.taskProcessor.on('taskCompleted', (result: TaskResult) => {
      this.log(`Task completed [${result.taskId}] in ${result.cliResult.duration}ms`);
    });

    this.taskProcessor.on('taskFailed', (result: TaskResult) => {
      this.log(`Task failed [${result.taskId}]: ${result.cliResult.error}`, 'error');
    });

    // 将消息处理流接入自动回复
    this.setupAutoReply();
  }

  private setupAutoReply(): void {
    this.messageBus.on('message', async (msg: IMMessage) => {
      if (!this.autoReplyEnabled || !this.taskProcessor) return;

      try {
        // 发送"正在处理"状态（用户有待确认操作时跳过）
        const processingConfig = this.getTaskConfig();
        const hasPending = this.taskProcessor.hasPendingRisk(msg.channelId, msg.senderId);
        if (processingConfig?.sendProcessingStatus && !hasPending) {
          const statusMsg = processingConfig.processingTemplate || '⏳ 正在处理您的请求，请稍候...';
          await this.messageBus.send(msg.adapterId, msg.channelId, statusMsg).catch(() => {});
        }

        const result = await this.taskProcessor.processMessage(msg);
        if (!result) return;

        await this.messageBus.send(msg.adapterId, msg.channelId, result.reply);
        this.log(`Auto-reply sent to ${msg.channelId} via ${msg.adapterType}`);
      } catch (err) {
        this.log(`Auto-reply error: ${err}`, 'error');
      }
    });

    if (this.taskProcessor) {
      this.taskProcessor.on('riskDetected', ({ taskId, assessment }: { taskId: string; assessment: { level: string; category: string } }) => {
        this.log(`Risk detected [${taskId}]: ${assessment.category} (${assessment.level})`);
      });
      this.taskProcessor.on('riskConfirmed', ({ taskId }: { taskId: string }) => {
        this.log(`Risk confirmed [${taskId}], executing...`);
      });
      this.taskProcessor.on('riskCancelled', ({ taskId }: { taskId: string }) => {
        this.log(`Risk cancelled [${taskId}]`);
      });
    }
  }

  /**
   * 手动触发：用指定消息调用 Cursor CLI 并回复
   */
  async processAndReply(adapterId: string, channelId: string, prompt: string): Promise<TaskResult | null> {
    if (!this.taskProcessor) {
      throw new Error('Task Processor not initialized. Enable autoReply in settings.');
    }

    const fakeMsg: IMMessage = {
      id: `manual-${Date.now()}`,
      adapterId,
      adapterType: this.registry.get(adapterId)?.type || 'custom' as never,
      direction: 'inbound' as never,
      channelId,
      senderId: 'manual',
      content: prompt,
      contentType: 'text',
      timestamp: Date.now(),
    };

    const result = await this.taskProcessor.processMessage(fakeMsg);
    if (result) {
      await this.messageBus.send(adapterId, channelId, result.reply);
    }
    return result;
  }

  // ─── 连接管理 ─────────────────────────────────────────────────────

  async connectAll(): Promise<void> {
    const adapters = this.registry.getAll();
    await Promise.allSettled(
      adapters.map((a) => this.connectAdapter(a.id))
    );
  }

  async connectAdapter(adapterId: string): Promise<void> {
    const adapter = this.registry.get(adapterId);
    if (!adapter) {
      throw new Error(`Adapter not found: ${adapterId}`);
    }

    try {
      await adapter.connect();
      this.log(`Connected: ${adapter.displayName}`);
      this.updateStatusBar();
    } catch (err) {
      this.log(`Connection failed for ${adapter.displayName}: ${err}`, 'error');
      throw err;
    }
  }

  async disconnectAll(): Promise<void> {
    const adapters = this.registry.getAll();
    await Promise.allSettled(
      adapters.map((a) => a.disconnect())
    );
    this.updateStatusBar();
  }

  async sendMessage(adapterId: string, channelId: string, content: string): Promise<IMMessage> {
    return this.messageBus.send(adapterId, channelId, content);
  }

  onMessage(handler: (msg: IMMessage) => void): vscode.Disposable {
    this.messageBus.on('message', handler);
    return new vscode.Disposable(() => {
      this.messageBus.off('message', handler);
    });
  }

  getAdapters(): IIMAdapter[] {
    return this.registry.getAll();
  }

  getMessageBus(): MessageBus {
    return this.messageBus;
  }

  isAutoReplyEnabled(): boolean {
    return this.autoReplyEnabled;
  }

  async setAutoReply(enabled: boolean): Promise<void> {
    if (enabled && !this.taskProcessor) {
      await this.initTaskProcessor();
    }
    this.autoReplyEnabled = enabled;
    this.log(`Auto-reply ${enabled ? 'enabled' : 'disabled'}`);
    this.updateStatusBar();
  }

  async dispose(): Promise<void> {
    await this.disconnectAll();
    await this.webhookServer.stop();
    this.messageBus.dispose();
    this.registry.clear();
    this.outputChannel.dispose();
    this.statusBarItem.dispose();
  }

  // ─── Webhook ──────────────────────────────────────────────────────

  private setupWebhookRouting(): void {
    this.webhookServer.on('webhook', (payload: WebhookPayload) => {
      this.routeWebhook(payload);
    });
  }

  private routeWebhook(payload: WebhookPayload): void {
    const adapters = this.registry.getAll().filter(
      (a) => a.type === payload.adapter
    );

    for (const adapter of adapters) {
      const handler = adapter as unknown as { handleWebhookEvent?: (data: unknown) => void };
      if (typeof handler.handleWebhookEvent === 'function') {
        try {
          handler.handleWebhookEvent(payload.data);
        } catch (err) {
          this.log(`Webhook handling error in ${adapter.displayName}: ${err}`, 'error');
        }
      }
    }
  }

  // ─── Status Bar ───────────────────────────────────────────────────

  private setupStatusBar(): void {
    this.statusBarItem.command = 'cursorImBridge.showPanel';
    this.statusBarItem.show();
    this.updateStatusBar();
  }

  private updateStatusBar(): void {
    const adapters = this.registry.getAll();
    const connected = adapters.filter(
      (a) => a.status === ConnectionStatus.Connected
    ).length;
    const total = adapters.length;
    const autoReplyTag = this.autoReplyEnabled ? ' [AI]' : '';

    if (total === 0) {
      this.statusBarItem.text = '$(comment-discussion) IM Bridge: No adapters';
      this.statusBarItem.tooltip = 'Click to configure IM adapters';
    } else if (connected === total) {
      this.statusBarItem.text = `$(check) IM Bridge: ${connected}/${total}${autoReplyTag}`;
      this.statusBarItem.tooltip = `All ${total} adapters connected${autoReplyTag ? ' | Auto-reply ON' : ''}`;
    } else {
      this.statusBarItem.text = `$(warning) IM Bridge: ${connected}/${total}${autoReplyTag}`;
      this.statusBarItem.tooltip = `${connected} of ${total} adapters connected`;
    }
  }

  // ─── Config ───────────────────────────────────────────────────────

  private getWebhookPort(): number {
    return vscode.workspace
      .getConfiguration('cursorImBridge')
      .get<number>('webhookPort', 3927);
  }

  private getWebhookSecrets() {
    const configs = this.getAdapterConfigs();
    const secrets: import('./webhook-server').WebhookSecrets = {};

    for (const cfg of configs) {
      const c = cfg.config as Record<string, unknown>;
      if (cfg.type === 'feishu') {
        secrets.feishu = {
          verificationToken: c.verificationToken as string | undefined,
          encryptKey: c.encryptKey as string | undefined,
        };
      } else if (cfg.type === 'dingtalk') {
        secrets.dingtalk = { robotSecret: c.robotSecret as string | undefined };
      } else if (cfg.type === 'wecom') {
        secrets.wecom = { token: c.token as string | undefined };
      }
    }
    return secrets;
  }

  private getAdapterConfigs(): AdapterConfig[] {
    return vscode.workspace
      .getConfiguration('cursorImBridge')
      .get<AdapterConfig[]>('adapters', []);
  }

  /**
   * 将 SecretStorage 中的凭证合并到适配器配置，覆盖 settings.json 明文值。
   * SecretStorage 优先级高于 settings.json，确保敏感字段不暴露在配置文件中。
   */
  private async mergeSecureCredentials(config: AdapterConfig): Promise<AdapterConfig> {
    const sensitiveFields: Record<string, string[]> = {
      feishu:   ['appSecret', 'encryptKey', 'verificationToken'],
      wecom:    ['corpSecret', 'encodingAESKey'],
      telegram: ['botToken'],
      dingtalk: ['appSecret', 'robotSecret'],
      custom:   ['credentials'],
    };

    const fields = sensitiveFields[config.type] || [];
    if (fields.length === 0) return config;

    const mergedConfig = { ...config.config } as Record<string, unknown>;
    let hasMerged = false;

    for (const field of fields) {
      const stored = await this.secureStore.getAdapterSecret(config.type, field);
      if (stored) {
        mergedConfig[field] = stored;
        hasMerged = true;
      }
    }

    if (!hasMerged) return config;

    return { ...config, config: mergedConfig };
  }

  /**
   * 将凭证保存到 SecretStorage（推荐用户调用，取代直接写 settings.json）
   */
  async storeAdapterSecret(adapterType: string, field: string, value: string): Promise<void> {
    await this.secureStore.setAdapterSecret(adapterType, field, value);
    this.log(`Secret stored for ${adapterType}.${field}`);
  }

  private getTaskConfig(): TaskConfig | null {
    const config = vscode.workspace.getConfiguration('cursorImBridge');
    const enabled = config.get<boolean>('autoReply.enabled', false);
    if (!enabled) return null;

    return {
      cli: {
        cliPath: config.get<string>('autoReply.cliPath') || undefined,
        cwd: config.get<string>('autoReply.cwd') || undefined,
        timeout: config.get<number>('autoReply.timeout', 120000),
      },
      sendProcessingStatus: config.get<boolean>('autoReply.sendProcessingStatus', true),
      processingTemplate: config.get<string>('autoReply.processingTemplate'),
      successTemplate: config.get<string>('autoReply.successTemplate'),
      errorTemplate: config.get<string>('autoReply.errorTemplate'),
      triggerPrefix: config.get<string>('autoReply.triggerPrefix', ''),
      ignoreSenders: config.get<string[]>('autoReply.ignoreSenders', ['self']),
      maxConcurrent: config.get<number>('autoReply.maxConcurrent', 3),
      riskControlEnabled: config.get<boolean>('autoReply.riskControl.enabled', true),
      riskConfirmTimeout: config.get<number>('autoReply.riskControl.confirmTimeout', 300000),
    };
  }

  private log(message: string, level: 'info' | 'error' | 'warn' = 'info'): void {
    const timestamp = new Date().toISOString();
    const line = `[${timestamp}] [${level.toUpperCase()}] ${message}`;
    this.outputChannel.appendLine(line);

    if (level === 'error') {
      vscode.window.showErrorMessage(`IM Bridge: ${message}`);
    }
  }
}
