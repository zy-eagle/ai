import * as crypto from 'crypto';

/**
 * Webhook 签名验证工具
 * 防止伪造 Webhook 请求触发 Cursor CLI 执行
 */

// ─── 飞书 Webhook 验证 ──────────────────────────────────────────────

/**
 * 验证飞书 Webhook 请求签名
 * 参考: https://open.feishu.cn/document/server-docs/webhook/webhook-verification
 */
export function verifyFeishuSignature(
  timestamp: string,
  nonce: string,
  encryptKey: string,
  bodyStr: string,
  signature: string
): boolean {
  if (!timestamp || !nonce || !signature) return false;

  // 飞书签名算法: sha256(timestamp + nonce + encryptKey + body)
  const content = `${timestamp}${nonce}${encryptKey}${bodyStr}`;
  const expected = crypto.createHash('sha256').update(content).digest('hex');

  return timingSafeEqual(expected, signature);
}

/**
 * 验证飞书 Token (simpler verification_token mode)
 */
export function verifyFeishuToken(
  bodyToken: string,
  verificationToken: string
): boolean {
  if (!bodyToken || !verificationToken) return false;
  return timingSafeEqual(bodyToken, verificationToken);
}

// ─── 钉钉 Webhook 验证 ──────────────────────────────────────────────

/**
 * 验证钉钉 Stream 回调签名
 * timestamp 和 secret 来自钉钉请求头
 */
export function verifyDingTalkSignature(
  timestamp: string,
  secret: string,
  signature: string
): boolean {
  if (!timestamp || !secret || !signature) return false;

  const stringToSign = `${timestamp}\n${secret}`;
  const hmac = crypto.createHmac('sha256', secret);
  hmac.update(stringToSign);
  const expected = hmac.digest('base64');

  return timingSafeEqual(expected, signature);
}

// ─── 企业微信 Webhook 验证 ──────────────────────────────────────────

/**
 * 验证企业微信回调 token（URL 验证和消息验证共用）
 */
export function verifyWeComSignature(
  token: string,
  timestamp: string,
  nonce: string,
  echostr: string,
  msgSignature: string
): boolean {
  if (!token || !timestamp || !nonce || !msgSignature) return false;

  const items = [token, timestamp, nonce, echostr].sort();
  const str = items.join('');
  const expected = crypto.createHash('sha1').update(str).digest('hex');

  return timingSafeEqual(expected, msgSignature);
}

// ─── 通用 ──────────────────────────────────────────────────────────

/**
 * 时序安全的字符串比较（防止 timing attack）
 */
function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) {
    // 长度不等时仍执行比较（防止长度信息泄漏）
    crypto.timingSafeEqual(Buffer.alloc(1), Buffer.alloc(1));
    return false;
  }
  return crypto.timingSafeEqual(Buffer.from(a, 'utf8'), Buffer.from(b, 'utf8'));
}
