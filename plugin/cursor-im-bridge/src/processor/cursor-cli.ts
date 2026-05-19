import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';
import { getPlatform } from '../utils/platform';

export interface CursorCLIOptions {
  /** Cursor CLI 可执行文件路径 (默认: 自动检测) */
  cliPath?: string;
  /** 工作目录 (默认: 当前工作区) */
  cwd?: string;
  /** 执行超时 (ms, 默认: 120000) */
  timeout?: number;
  /** 环境变量 */
  env?: Record<string, string>;
}

export interface CursorCLIResult {
  success: boolean;
  output: string;
  error?: string;
  exitCode: number | null;
  duration: number;
}

/**
 * Cursor CLI 执行器
 * 通过命令行调用 Cursor Agent 处理任务
 */
export class CursorCLI {
  private cliPath: string;
  private defaultCwd: string;
  private defaultTimeout: number;
  private env: Record<string, string>;

  constructor(options: CursorCLIOptions = {}) {
    this.cliPath = options.cliPath || this.detectCLIPath();
    this.defaultCwd = options.cwd || process.cwd();
    this.defaultTimeout = options.timeout || 120000;
    this.env = options.env || {};
  }

  /**
   * 执行 Cursor CLI 命令 (使用 Agent 模式处理任务)
   */
  async execute(prompt: string, options?: { cwd?: string; timeout?: number }): Promise<CursorCLIResult> {
    const sanitized = this.sanitizePrompt(prompt);
    if (!sanitized.ok) {
      return { success: false, output: '', error: sanitized.reason, exitCode: null, duration: 0 };
    }

    const cwd = options?.cwd || this.defaultCwd;
    const timeout = options?.timeout || this.defaultTimeout;
    const startTime = Date.now();

    return new Promise((resolve) => {
      // shell: false — args 数组直接传给进程，不经过 shell 解释，防止命令注入
      const args = ['agent', '--message', sanitized.value];
      let stdout = '';
      let stderr = '';
      let killed = false;

      const proc: ChildProcess = spawn(this.cliPath, args, {
        cwd,
        env: { ...process.env, ...this.env },
        shell: false,
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      const timer = setTimeout(() => {
        killed = true;
        proc.kill('SIGTERM');
        setTimeout(() => proc.kill('SIGKILL'), 5000);
      }, timeout);

      proc.stdout?.on('data', (data) => {
        stdout += data.toString();
      });

      proc.stderr?.on('data', (data) => {
        stderr += data.toString();
      });

      proc.on('close', (code) => {
        clearTimeout(timer);
        const duration = Date.now() - startTime;

        if (killed) {
          resolve({
            success: false,
            output: stdout,
            error: `Timeout after ${timeout}ms`,
            exitCode: code,
            duration,
          });
          return;
        }

        resolve({
          success: code === 0,
          output: stdout.trim(),
          error: stderr.trim() || undefined,
          exitCode: code,
          duration,
        });
      });

      proc.on('error', (err) => {
        clearTimeout(timer);
        resolve({
          success: false,
          output: '',
          error: `Failed to start cursor CLI: ${err.message}`,
          exitCode: null,
          duration: Date.now() - startTime,
        });
      });
    });
  }

  /**
   * 流式执行，逐步回调输出
   */
  async executeStream(
    prompt: string,
    onChunk: (chunk: string) => void,
    options?: { cwd?: string; timeout?: number }
  ): Promise<CursorCLIResult> {
    const sanitized = this.sanitizePrompt(prompt);
    if (!sanitized.ok) {
      return { success: false, output: '', error: sanitized.reason, exitCode: null, duration: 0 };
    }

    const cwd = options?.cwd || this.defaultCwd;
    const timeout = options?.timeout || this.defaultTimeout;
    const startTime = Date.now();

    return new Promise((resolve) => {
      const args = ['agent', '--message', sanitized.value];
      let stdout = '';
      let stderr = '';
      let killed = false;

      const proc: ChildProcess = spawn(this.cliPath, args, {
        cwd,
        env: { ...process.env, ...this.env },
        shell: false,
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      const timer = setTimeout(() => {
        killed = true;
        proc.kill('SIGTERM');
        setTimeout(() => proc.kill('SIGKILL'), 5000);
      }, timeout);

      proc.stdout?.on('data', (data) => {
        const chunk = data.toString();
        stdout += chunk;
        onChunk(chunk);
      });

      proc.stderr?.on('data', (data) => {
        stderr += data.toString();
      });

      proc.on('close', (code) => {
        clearTimeout(timer);
        const duration = Date.now() - startTime;

        resolve({
          success: !killed && code === 0,
          output: stdout.trim(),
          error: killed ? `Timeout after ${timeout}ms` : (stderr.trim() || undefined),
          exitCode: code,
          duration,
        });
      });

      proc.on('error', (err) => {
        clearTimeout(timer);
        resolve({
          success: false,
          output: '',
          error: `Failed to start cursor CLI: ${err.message}`,
          exitCode: null,
          duration: Date.now() - startTime,
        });
      });
    });
  }

  /**
   * 检查 Cursor CLI 是否可用
   */
  async isAvailable(): Promise<boolean> {
    return new Promise((resolve) => {
      const proc = spawn(this.cliPath, ['--version'], {
        shell: false,
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      proc.on('close', (code) => {
        resolve(code === 0);
      });

      proc.on('error', () => {
        resolve(false);
      });

      setTimeout(() => {
        proc.kill();
        resolve(false);
      }, 5000);
    });
  }

  /**
   * 获取 Cursor CLI 版本
   */
  async getVersion(): Promise<string | null> {
    return new Promise((resolve) => {
      let output = '';
      const proc = spawn(this.cliPath, ['--version'], {
        shell: false,
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      proc.stdout?.on('data', (data) => {
        output += data.toString();
      });

      proc.on('close', (code) => {
        resolve(code === 0 ? output.trim() : null);
      });

      proc.on('error', () => {
        resolve(null);
      });
    });
  }

  private detectCLIPath(): string {
    const platform = getPlatform();
    switch (platform) {
      case 'win32':
        return 'cursor.cmd';
      case 'darwin':
        return '/usr/local/bin/cursor';
      case 'linux':
        return 'cursor';
      default:
        return 'cursor';
    }
  }

  /**
   * 对 prompt 内容进行安全验证
   * - 限制最大长度，防止超大 payload
   * - 拒绝包含 null 字节（防止参数截断攻击）
   */
  private sanitizePrompt(
    prompt: string
  ): { ok: true; value: string } | { ok: false; reason: string } {
    const MAX_LENGTH = 8000;

    if (!prompt || !prompt.trim()) {
      return { ok: false, reason: 'Prompt is empty' };
    }

    if (prompt.length > MAX_LENGTH) {
      return { ok: false, reason: `Prompt exceeds maximum length of ${MAX_LENGTH} characters` };
    }

    if (prompt.includes('\0')) {
      return { ok: false, reason: 'Prompt contains invalid null byte' };
    }

    return { ok: true, value: prompt.trim() };
  }
}
