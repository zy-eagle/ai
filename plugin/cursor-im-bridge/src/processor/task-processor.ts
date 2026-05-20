import { EventEmitter } from 'events';
import { IMMessage } from '../types';
import { CursorCLI, CursorCLIOptions, CursorCLIResult } from './cursor-cli';
import { RiskControl } from './risk-control';

export interface TaskConfig {
  cli: CursorCLIOptions;
  sendProcessingStatus?: boolean;
  processingTemplate?: string;
  successTemplate?: string;
  errorTemplate?: string;
  timeoutTemplate?: string;
  queueFullTemplate?: string;
  triggerPrefix?: string;
  ignoreSenders?: string[];
  maxConcurrent?: number;
  maxQueueSize?: number;
  /** 风控：是否启用 AI 意图分析 */
  riskControlEnabled?: boolean;
  /** 风控：确认超时时间（ms，默认 5 分钟） */
  riskConfirmTimeout?: number;
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
 * 任务处理器
 *
 * 流程: IM消息 → 过滤 → [风控AI分析] → [确认] → Cursor CLI 执行 → 格式化回复
 */
export class TaskProcessor extends EventEmitter {
  private cli: CursorCLI;
  private config: Required<TaskConfig>;
  private riskControl: RiskControl | null = null;
  private queue: TaskQueueItem[] = [];
  private activeCount: number = 0;

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
      riskControlEnabled: config.riskControlEnabled ?? true,
      riskConfirmTimeout: config.riskConfirmTimeout ?? 300000,
    };

    if (this.config.riskControlEnabled) {
      this.riskControl = new RiskControl({
        confirmTimeoutMs: this.config.riskConfirmTimeout,
      });
    }
  }

  /**
   * 处理收到的 IM 消息
   */
  async processMessage(message: IMMessage): Promise<TaskResult | null> {
    if (!this.shouldProcess(message)) {
      return null;
    }

    const prompt = this.extractPrompt(message);
    if (!prompt) {
      return null;
    }

    // 风控：先检查是否是对待确认操作的回复
    if (this.riskControl) {
      const confirmReply = this.riskControl.checkReply(
        message.channelId, message.senderId, prompt
      );

      if (confirmReply) {
        if (confirmReply.action === 'confirm') {
          this.emit('riskConfirmed', {
            taskId: confirmReply.pending.taskId,
            prompt: confirmReply.pending.prompt,
          });
          return this.executeTask(message, confirmReply.pending.prompt);
        } else {
          this.emit('riskCancelled', { taskId: confirmReply.pending.taskId });
          return {
            taskId: confirmReply.pending.taskId,
            sourceMessage: message,
            cliResult: { success: true, output: '', exitCode: null, duration: 0 },
            reply: '✅ 操作已取消。',
          };
        }
      }

      // 如果该用户有待确认项，但回复的不是确认/取消，提醒用户
      if (this.riskControl.hasPending(message.channelId, message.senderId)) {
        return {
          taskId: `reminder-${Date.now()}`,
          sourceMessage: message,
          cliResult: { success: true, output: '', exitCode: null, duration: 0 },
          reply: '⏳ 您有一个待确认的高危操作。请先回复 "确认" 执行或 "取消" 放弃，再提交新指令。',
        };
      }
    }

    // 队列满则拒绝
    if (this.queue.length >= this.config.maxQueueSize) {
      return {
        taskId: `rejected-${Date.now()}`,
        sourceMessage: message,
        cliResult: { success: false, output: '', error: 'Queue full', exitCode: null, duration: 0 },
        reply: this.config.queueFullTemplate,
      };
    }

    // 风控：语义模式检测
    if (this.riskControl) {
      const assessment = this.riskControl.analyze(prompt);
      this.emit('riskAssessed', { prompt, assessment });

      if (this.riskControl.needsConfirmation(assessment)) {
        const taskId = this.riskControl.createPending(
          message.channelId, message.senderId, prompt, assessment
        );
        const confirmPrompt = this.riskControl.buildConfirmPrompt(assessment);

        this.emit('riskDetected', { taskId, prompt, assessment });

        return {
          taskId,
          sourceMessage: message,
          cliResult: { success: true, output: '', exitCode: null, duration: 0 },
          reply: confirmPrompt,
        };
      }
    }

    // 安全或低风险：直接执行
    return this.executeTask(message, prompt);
  }

  /**
   * 预检：是否该用户有待确认的高危操作（用于跳过"正在处理"消息）
   */
  hasPendingRisk(channelId: string, senderId: string): boolean {
    return this.riskControl?.hasPending(channelId, senderId) ?? false;
  }

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

  private async executeTask(message: IMMessage, prompt: string): Promise<TaskResult> {
    const taskId = `task-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    const item: TaskQueueItem = { id: taskId, message, prompt };
    this.queue.push(item);
    this.emit('taskQueued', { taskId, message, prompt });

    return this.processNextInQueue(item);
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
    if (message.direction === 'outbound') return false;
    if (this.config.ignoreSenders.includes(message.senderId)) return false;
    if (this.config.triggerPrefix && !message.content.startsWith(this.config.triggerPrefix)) return false;
    if (!message.content.trim()) return false;
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
      if (result.error?.includes('Timeout')) return this.config.timeoutTemplate;
      return this.config.errorTemplate.replace('{error}', result.error || 'Unknown error');
    }

    let reply = this.config.successTemplate
      .replace('{output}', result.output)
      .replace('{duration}', `${(result.duration / 1000).toFixed(1)}s`);

    const maxLength = 4000;
    if (reply.length > maxLength) {
      reply = reply.slice(0, maxLength - 100) + '\n\n... (输出已截断，完整结果请查看 Cursor)';
    }
    return reply;
  }
}
