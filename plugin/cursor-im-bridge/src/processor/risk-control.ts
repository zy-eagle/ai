/**
 * 风控模块 — 混合策略：语义关键词预检 + AI 确认
 *
 * 策略：
 * 1. 用宽泛的语义模式检测潜在高危意图（宁可误报，不可漏报）
 * 2. 命中后向用户发送确认提示（含 AI 对操作的解读）
 * 3. 用户确认后才执行
 *
 * 与纯关键词匹配不同，这里用的是多维度语义模式，
 * 覆盖各种中英文表达方式、口语化表达和同义替换。
 */

export type RiskLevel = 'safe' | 'high' | 'critical';

export interface RiskAssessment {
  level: RiskLevel;
  category: string;
  reason: string;
  originalPrompt: string;
}

export interface PendingConfirmation {
  taskId: string;
  channelId: string;
  senderId: string;
  prompt: string;
  assessment: RiskAssessment;
  createdAt: number;
}

const CONFIRM_KEYWORDS = ['确认', '确定', 'confirm', 'yes', 'y', '是', '执行', '好的', 'ok'];
const CANCEL_KEYWORDS = ['取消', '撤销', 'cancel', 'no', 'n', '否', '算了', '不要', '停'];

interface RiskPattern {
  category: string;
  level: RiskLevel;
  reason: string;
  /** 任意一个匹配即命中 */
  matchers: RegExp[];
}

/**
 * 宽泛的语义模式——覆盖多种表达方式
 * 设计原则：宁可多拦截让用户确认一下，不可漏掉高危操作
 */
const RISK_PATTERNS: RiskPattern[] = [
  {
    category: '文件/目录删除',
    level: 'high',
    reason: '该操作将删除文件或目录，可能不可逆',
    matchers: [
      /删[除掉了]/, /移除/, /清[除理空掉].*(?:文件|目录|文件夹|代码|项目)/,
      /(?:文件|目录|文件夹|代码|项目).*(?:删[除掉了]|移除|清[除理空掉])/,
      /rm\b/i, /del\b/i, /remove/i, /unlink/i, /rimraf/i,
      /(?:去掉|干掉|消灭|抹掉|擦除).*(?:文件|目录|文件夹|folder|dir)/i,
    ],
  },
  {
    category: '数据库操作',
    level: 'critical',
    reason: '该操作涉及数据库修改，可能导致数据丢失',
    matchers: [
      /drop\s+(?:table|database|index|collection)/i,
      /truncate/i,
      /delete\s+from/i,
      /删[除掉].*(?:数据|表|记录|库)/, /(?:数据|表|记录|库).*删[除掉]/,
      /清[空除].*(?:数据|表|记录)/, /(?:数据|表|记录).*清[空除]/,
      /迁移.*数据库/, /重建.*(?:索引|表)/,
    ],
  },
  {
    category: '部署/发布',
    level: 'high',
    reason: '该操作涉及部署或发布，将影响线上环境',
    matchers: [
      /部署/, /发布/, /上线/, /发版/,
      /deploy/i, /publish/i, /release/i,
      /push.*(?:prod|production|线上|生产)/i,
      /(?:prod|production|线上|生产).*push/i,
    ],
  },
  {
    category: 'Git 危险操作',
    level: 'high',
    reason: '该操作可能丢失 Git 历史或覆盖远程代码',
    matchers: [
      /force\s*push/i, /push.*-f/i, /push.*--force/i,
      /强制.*推送/, /强推/,
      /reset.*hard/i, /git\s+reset/i,
      /rebase.*(?:main|master)/i,
      /重置.*(?:分支|代码|提交)/, /回滚.*(?:提交|版本)/,
    ],
  },
  {
    category: '系统级操作',
    level: 'critical',
    reason: '该操作涉及系统级变更，执行后果可能极其严重',
    matchers: [
      /rm\s+-rf\s+[\/\\]/i,
      /format\s+[a-z]:/i, /格式化.*[盘驱磁]/,
      /shutdown/i, /关机/, /重启.*(?:服务器|系统)/,
      /kill\s+(?:-9\s+)?(?:\d+|all)/i,
      /(?:停止|关闭|终止).*(?:所有|全部).*(?:服务|进程)/,
    ],
  },
  {
    category: '批量修改',
    level: 'high',
    reason: '该操作涉及批量修改，影响范围较大',
    matchers: [
      /(?:所有|全部|批量).*(?:替换|修改|更新|重命名)/,
      /(?:替换|修改|更新|重命名).*(?:所有|全部|每个)/,
      /sed\s+-i/i, /find.*-exec/i,
      /(?:全局|全量).*(?:替换|修改)/,
    ],
  },
  {
    category: '凭证/密钥操作',
    level: 'high',
    reason: '该操作涉及敏感凭证，可能造成信息泄露',
    matchers: [
      /(?:发送|输出|显示|打印|告诉|给我|分享).*(?:密[码钥]|password|secret|token|api.?key|private.?key)/i,
      /(?:密[码钥]|password|secret|token|api.?key|private.?key).*(?:发[送出]|告诉|给|分享|公开)/i,
      /(?:上传|提交|push).*(?:\.env|secret|credential|密钥)/i,
    ],
  },
];

export class RiskControl {
  private pendingConfirmations: Map<string, PendingConfirmation> = new Map();
  private confirmTimeoutMs: number;

  constructor(options?: { confirmTimeoutMs?: number }) {
    this.confirmTimeoutMs = options?.confirmTimeoutMs || 300000;
  }

  /**
   * 分析用户指令的风险等级
   */
  analyze(prompt: string): RiskAssessment {
    for (const pattern of RISK_PATTERNS) {
      for (const matcher of pattern.matchers) {
        if (matcher.test(prompt)) {
          return {
            level: pattern.level,
            category: pattern.category,
            reason: pattern.reason,
            originalPrompt: prompt,
          };
        }
      }
    }

    return {
      level: 'safe',
      category: '常规操作',
      reason: '未检测到高危操作模式',
      originalPrompt: prompt,
    };
  }

  /**
   * 判断是否需要确认
   */
  needsConfirmation(assessment: RiskAssessment): boolean {
    return assessment.level === 'high' || assessment.level === 'critical';
  }

  /**
   * 生成确认提示消息
   */
  buildConfirmPrompt(assessment: RiskAssessment): string {
    const icon = assessment.level === 'critical' ? '🚨' : '⚠️';
    const levelText = assessment.level === 'critical' ? '极高风险' : '高风险';
    const preview = assessment.originalPrompt.length > 150
      ? assessment.originalPrompt.slice(0, 150) + '...'
      : assessment.originalPrompt;

    return [
      `${icon} **${levelText}操作 — 需要确认**`,
      '',
      `📋 您的指令: "${preview}"`,
      `📂 操作类型: ${assessment.category}`,
      `💡 风险说明: ${assessment.reason}`,
      '',
      '请回复以下内容来确认或取消：',
      '  ✅ "确认" 或 "yes" → 执行操作',
      '  ❌ "取消" 或 "no" → 取消操作',
      '',
      `⏰ 未确认将在 ${Math.round(this.confirmTimeoutMs / 60000)} 分钟后自动取消。`,
    ].join('\n');
  }

  /**
   * 创建待确认记录
   */
  createPending(channelId: string, senderId: string, prompt: string, assessment: RiskAssessment): string {
    this.cleanExpired();

    const taskId = `risk-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    this.pendingConfirmations.set(this.pendingKey(channelId, senderId), {
      taskId,
      channelId,
      senderId,
      prompt,
      assessment,
      createdAt: Date.now(),
    });
    return taskId;
  }

  /**
   * 检查消息是否是对待确认操作的回复
   */
  checkReply(channelId: string, senderId: string, content: string): {
    action: 'confirm' | 'cancel';
    pending: PendingConfirmation;
  } | null {
    const key = this.pendingKey(channelId, senderId);
    const pending = this.pendingConfirmations.get(key);

    if (!pending) return null;

    if (Date.now() - pending.createdAt > this.confirmTimeoutMs) {
      this.pendingConfirmations.delete(key);
      return null;
    }

    const normalized = content.trim().toLowerCase();

    if (CONFIRM_KEYWORDS.some(k => normalized === k || normalized.startsWith(k))) {
      this.pendingConfirmations.delete(key);
      return { action: 'confirm', pending };
    }

    if (CANCEL_KEYWORDS.some(k => normalized === k || normalized.startsWith(k))) {
      this.pendingConfirmations.delete(key);
      return { action: 'cancel', pending };
    }

    return null;
  }

  /**
   * 获取某个 channel+sender 是否有待确认项
   */
  hasPending(channelId: string, senderId: string): boolean {
    const key = this.pendingKey(channelId, senderId);
    const pending = this.pendingConfirmations.get(key);
    if (!pending) return false;

    if (Date.now() - pending.createdAt > this.confirmTimeoutMs) {
      this.pendingConfirmations.delete(key);
      return false;
    }
    return true;
  }

  private pendingKey(channelId: string, senderId: string): string {
    return `${channelId}::${senderId}`;
  }

  private cleanExpired(): void {
    const now = Date.now();
    for (const [key, pending] of this.pendingConfirmations) {
      if (now - pending.createdAt > this.confirmTimeoutMs) {
        this.pendingConfirmations.delete(key);
      }
    }
  }
}
