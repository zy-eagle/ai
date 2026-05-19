import { IMMessage } from '../types';

export enum TaskStatus {
  Pending = 'pending',
  Processing = 'processing',
  Completed = 'completed',
  Failed = 'failed',
}

export enum ProcessorBackend {
  /** 通过 Cursor SDK 调用 Cursor Agent */
  CursorAgent = 'cursor-agent',
  /** 直接调用大模型 API (OpenAI/Claude/自定义) */
  LLMApi = 'llm-api',
}

export interface Task {
  id: string;
  status: TaskStatus;
  /** 来源消息 */
  sourceMessage: IMMessage;
  /** 解析后的用户意图/指令 */
  instruction: string;
  /** 处理结果 */
  result?: string;
  /** 错误信息 */
  error?: string;
  /** 创建时间 */
  createdAt: number;
  /** 完成时间 */
  completedAt?: number;
  /** 上下文 (多轮对话) */
  conversationId?: string;
}

export interface ProcessorConfig {
  /** 使用的后端 */
  backend: ProcessorBackend;
  /** Cursor Agent 配置 */
  cursorAgent?: {
    /** 工作目录 (Cursor 项目路径) */
    workspaceDir?: string;
  };
  /** LLM API 配置 */
  llmApi?: {
    /** API 端点 */
    baseUrl: string;
    /** API Key */
    apiKey: string;
    /** 模型名称 */
    model: string;
    /** 系统提示词 */
    systemPrompt?: string;
    /** 最大 token */
    maxTokens?: number;
    /** 温度 */
    temperature?: number;
  };
  /** 自动回复开关 */
  autoReply: boolean;
  /** 最大并发任务数 */
  maxConcurrency: number;
  /** 任务超时时间 (ms) */
  taskTimeout: number;
  /** 触发前缀 (只处理以此前缀开头的消息, 为空则处理所有) */
  triggerPrefix?: string;
  /** 忽略的发送者ID */
  ignoreSenders?: string[];
}

export interface IProcessorBackend {
  readonly name: string;
  process(instruction: string, context?: ConversationContext): Promise<string>;
  dispose(): void;
}

export interface ConversationContext {
  conversationId: string;
  history: Array<{ role: 'user' | 'assistant'; content: string }>;
}
