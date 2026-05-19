import * as vscode from 'vscode';
import { AdapterConfig, ConnectionStatus, IIMAdapter, IMMessage, WebhookPayload } from '../types';
import { AdapterRegistry } from './adapter-registry';
import { MessageBus } from './message-bus';
import { WebhookServer } from './webhook-server';
import { TaskProcessor, TaskConfig, TaskResult } from '../processor';

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

  constructor(private context: vscode.ExtensionContext) {
    this.registry = new AdapterRegistry();
    this.messageBus = new MessageBus();
    this.webhookServer = new WebhookServer(this.getWebhookPort());
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
        const adapter = this.registry.create(config);
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
    if (!taskConfig) return;

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
        // 发送处理中状态
        const processingConfig = this.getTaskConfig();
        if (processingConfig?.sendProcessingStatus) {
          const statusMsg = processingConfig.processingTemplate || '⏳ 正在处理您的请求，请稍候...';
          await this.messageBus.send(msg.adapterId, msg.channelId, statusMsg).catch(() => {});
        }

        // 调用 Cursor CLI 处理
        const result = await this.taskProcessor.processMessage(msg);

        if (result) {
          // 将结果回复到原来的 channel
          await this.messageBus.send(msg.adapterId, msg.channelId, result.reply);
          this.log(`Auto-reply sent to ${msg.channelId} via ${msg.adapterType}`);
        }
      } catch (err) {
        this.log(`Auto-reply error: ${err}`, 'error');
      }
    });
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

  setAutoReply(enabled: boolean): void {
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

  private getAdapterConfigs(): AdapterConfig[] {
    return vscode.workspace
      .getConfiguration('cursorImBridge')
      .get<AdapterConfig[]>('adapters', []);
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
