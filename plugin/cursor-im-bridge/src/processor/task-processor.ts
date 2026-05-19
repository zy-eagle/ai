import { EventEmitter } from 'events';
import { IMMessage } from '../types';
import { CursorCLI, CursorCLIOptions, CursorCLIResult } from './cursor-cli';

export interface TaskConfig {
  /** Cursor CLI 配置 */
  cli: CursorCLIOptions;
  /** 是否发送处理中状态消息 */
  sendProcessingStatus?: boolean;
  /** 处理中状态消息模板 */
  processingTemplate?: string;
  /** 成功响应模板 (支持 {output}, {duration} 占位符) */
  successTemplate?: string;
  /** 失败响应模板 (支持 {error} 占位符) */
  errorTemplate?: string;
  /** 超时响应模板 */
  timeoutTemplate?: string;
  /** 队列满时的回复模板 */
  queueFullTemplate?: string;
  /** 消息过滤：只处理包含指定前缀的消息 */
  triggerPrefix?: string;
  /** 消息过滤：忽略来自这些 senderId 的消息 */
  ignoreSenders?: string[];
  /** 最大并发任务数 */
  maxConcurrent?: number;
  /** 最大队列长度（防止 DoS），超出时直接拒绝新消息 */
  maxQueueSize?: number;
}

export interface TaskResult {
  taskId: string;
  sourceMessage: IMMessage;
  cliResult: CursorCLIResult;
  reply: string;
}

interface TaskQueueItem {
  id: string;
  message: IMMessage;
  prompt: string;
}

/**
 * 任务处理器 — 接收 IM 消息，调用 Cursor CLI 处理，生成回复内容
 *
 * 流程: IM消息 → 过滤 → 提取Prompt → Cursor CLI → 格式化回复
 */
export class TaskProcessor extends EventEmitter {
  private cli: CursorCLI;
  private config: Required<TaskConfig>;
  private queue: TaskQueueItem[] = [];
  private activeCount: number = 0;
  private processing: boolean = false;

  constructor(config: TaskConfig) {
    super();
    this.cli = new CursorCLI(config.cli);
    this.config = {
      cli: config.cli,
      sendProcessingStatus: config.sendProcessingStatus ?? true,
      processingTemplate: config.processingTemplate ?? '⏳ 正在处理您的请求，请稍候...',
      successTemplate: config.successTemplate ?? '{output}',
      errorTemplate: config.errorTemplate ?? '❌ 处理失败: {error}',
      timeoutTemplate: config.timeoutTemplate ?? '⏰ 处理超时，请稍后重试或简化您的请求。',
      queueFullTemplate: config.queueFullTemplate ?? '⚠️ 当前任务队列已满，请稍后再试。',
      triggerPrefix: config.triggerPrefix ?? '',
      ignoreSenders: config.ignoreSenders ?? ['self'],
      maxConcurrent: config.maxConcurrent ?? 3,
      maxQueueSize: config.maxQueueSize ?? 20,
    };
  }

  /**
   * 处理收到的 IM 消息
   * 返回 null 表示消息被过滤（不需要处理）
   */
  async processMessage(message: IMMessage): Promise<TaskResult | null> {
    if (!this.shouldProcess(message)) {
      return null;
    }

    const prompt = this.extractPrompt(message);
    if (!prompt) {
      return null;
    }

    // 队列满则拒绝，防止无限堆积造成 OOM
    if (this.queue.length >= this.config.maxQueueSize) {
      return {
        taskId: `rejected-${Date.now()}`,
        sourceMessage: message,
        cliResult: { success: false, output: '', error: 'Queue full', exitCode: null, duration: 0 },
        reply: this.config.queueFullTemplate,
      };
    }

    const taskId = `task-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;

    const item: TaskQueueItem = { id: taskId, message, prompt };
    this.queue.push(item);
    this.emit('taskQueued', { taskId, message, prompt });

    return this.processNextInQueue(item);
  }

  /**
   * 检查 Cursor CLI 是否就绪
   */
  async checkReady(): Promise<{ available: boolean; version?: string }> {
    const available = await this.cli.isAvailable();
    if (!available) {
      return { available: false };
    }
    const version = await this.cli.getVersion();
    return { available: true, version: version || undefined };
  }

  getQueueLength(): number {
    return this.queue.length;
  }

  getActiveCount(): number {
    return this.activeCount;
  }

  private async processNextInQueue(item: TaskQueueItem): Promise<TaskResult> {
    while (this.activeCount >= this.config.maxConcurrent) {
      await new Promise((resolve) => setTimeout(resolve, 500));
    }

    this.activeCount++;
    this.emit('taskStarted', { taskId: item.id, prompt: item.prompt });

    try {
      const cliResult = await this.cli.execute(item.prompt);
      const reply = this.formatReply(cliResult);

      const result: TaskResult = {
        taskId: item.id,
        sourceMessage: item.message,
        cliResult,
        reply,
      };

      this.emit('taskCompleted', result);
      return result;
    } catch (err) {
      const errorResult: TaskResult = {
        taskId: item.id,
        sourceMessage: item.message,
        cliResult: {
          success: false,
          output: '',
          error: err instanceof Error ? err.message : String(err),
          exitCode: null,
          duration: 0,
        },
        reply: this.config.errorTemplate.replace('{error}', String(err)),
      };

      this.emit('taskFailed', errorResult);
      return errorResult;
    } finally {
      this.activeCount--;
      this.queue = this.queue.filter((q) => q.id !== item.id);
    }
  }

  private shouldProcess(message: IMMessage): boolean {
    if (message.direction === 'outbound') {
      return false;
    }

    if (this.config.ignoreSenders.includes(message.senderId)) {
      return false;
    }

    if (this.config.triggerPrefix) {
      if (!message.content.startsWith(this.config.triggerPrefix)) {
        return false;
      }
    }

    if (!message.content.trim()) {
      return false;
    }

    return true;
  }

  private extractPrompt(message: IMMessage): string {
    let content = message.content.trim();

    if (this.config.triggerPrefix && content.startsWith(this.config.triggerPrefix)) {
      content = content.slice(this.config.triggerPrefix.length).trim();
    }

    return content;
  }

  private formatReply(result: CursorCLIResult): string {
    if (!result.success) {
      if (result.error?.includes('Timeout')) {
        return this.config.timeoutTemplate;
      }
      return this.config.errorTemplate.replace('{error}', result.error || 'Unknown error');
    }

    let reply = this.config.successTemplate
      .replace('{output}', result.output)
      .replace('{duration}', `${(result.duration / 1000).toFixed(1)}s`);

    // 飞书等 IM 对消息长度有限制，截断过长输出
    const maxLength = 4000;
    if (reply.length > maxLength) {
      reply = reply.slice(0, maxLength - 100) + '\n\n... (输出已截断，完整结果请查看 Cursor)';
    }

    return reply;
  }
}
