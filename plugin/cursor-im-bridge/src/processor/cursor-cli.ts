import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

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
 * 通过 `agent -p` 非交互模式调用 Cursor Agent 处理任务
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

  async execute(prompt: string, options?: { cwd?: string; timeout?: number }): Promise<CursorCLIResult> {
    const sanitized = this.sanitizePrompt(prompt);
    if (!sanitized.ok) {
      return { success: false, output: '', error: sanitized.reason, exitCode: null, duration: 0 };
    }

    const cwd = options?.cwd || this.defaultCwd;
    const timeout = options?.timeout || this.defaultTimeout;
    return this.spawnAgent(['-p', '--trust', sanitized.value], cwd, timeout);
  }

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
    return this.spawnAgent(['-p', '--trust', '--stream-partial-output', sanitized.value], cwd, timeout, onChunk);
  }

  async isAvailable(): Promise<boolean> {
    return new Promise((resolve) => {
      const proc = spawn(this.cliPath, ['--version'], {
        shell: this.needsShell(),
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      const timer = setTimeout(() => { proc.kill(); resolve(false); }, 5000);
      proc.on('close', (code) => { clearTimeout(timer); resolve(code === 0); });
      proc.on('error', () => { clearTimeout(timer); resolve(false); });
    });
  }

  async getVersion(): Promise<string | null> {
    return new Promise((resolve) => {
      let output = '';
      const proc = spawn(this.cliPath, ['--version'], {
        shell: this.needsShell(),
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      proc.stdout?.on('data', (data) => { output += data.toString(); });
      proc.on('close', (code) => { resolve(code === 0 ? output.trim() : null); });
      proc.on('error', () => { resolve(null); });
    });
  }

  private spawnAgent(
    args: string[],
    cwd: string,
    timeout: number,
    onChunk?: (chunk: string) => void
  ): Promise<CursorCLIResult> {
    const startTime = Date.now();

    return new Promise((resolve) => {
      let stdout = '';
      let stderr = '';
      let killed = false;

      const proc: ChildProcess = spawn(this.cliPath, args, {
        cwd,
        env: { ...process.env, ...this.env },
        shell: this.needsShell(),
        windowsHide: true,
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
        onChunk?.(chunk);
      });

      proc.stderr?.on('data', (data) => { stderr += data.toString(); });

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
   * Windows .cmd files require shell, direct binaries do not.
   * shell:false is preferred as it prevents shell injection from prompt content.
   */
  private needsShell(): boolean {
    return this.cliPath.endsWith('.cmd') || this.cliPath.endsWith('.bat');
  }

  private detectCLIPath(): string {
    const localAppData = process.env.LOCALAPPDATA || '';
    const home = process.env.HOME || process.env.USERPROFILE || '';

    const candidates = [
      localAppData && path.join(localAppData, 'cursor-agent', 'agent.cmd'),
      localAppData && path.join(localAppData, 'cursor-agent', 'agent'),
      home && path.join(home, '.cursor', 'bin', 'agent'),
      home && path.join(home, '.local', 'bin', 'agent'),
    ].filter(Boolean) as string[];

    for (const candidate of candidates) {
      try {
        if (fs.existsSync(candidate)) return candidate;
      } catch { /* ignore */ }
    }

    return 'agent';
  }

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
