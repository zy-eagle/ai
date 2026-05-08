# Test Report — MCP Query Server

## 1. Dependency Topology
| Dependency | Type | Category | Test Strategy |
|-----------|------|----------|---------------|
| Remote API (https://mydomain.com) | HTTP API | Third-Party | Mock via `vi.stubGlobal('fetch')` |
| Environment Variables | Config | System | `process.env` override in beforeEach/afterEach |

## 2. Mock Data Registry
| Mock ID | Target | Purpose | Data Shape |
|---------|--------|---------|------------|
| fetchSpy (session) | global fetch | 模拟 Session 鉴权请求 | `{ ok: true, status: 200, json: () => {...} }` |
| fetchSpy (aksk) | global fetch | 模拟 AK/SK 签名请求 | 同上 |
| fetchSpy (error) | global fetch | 模拟网络故障/超时 | `throw Error` / `DOMException` |
| env override | process.env | 模拟不同鉴权配置 | `AUTH_MODE`, `SESSION_TOKEN`, `ACCESS_KEY`, `SECRET_KEY` |

## 3. Scenario Coverage Matrix
| # | Scenario | Input | Expected Output | Mock Used | Result | Duration |
|---|----------|-------|-----------------|-----------|--------|----------|
| 1 | maskSecret 长字符串 | `"abcdefghij"` | `"abc***hij"` | — | PASS | <1ms |
| 2 | maskSecret 短字符串 | `"abc"` | `"***"` | — | PASS | <1ms |
| 3 | HMAC 签名确定性 | 相同输入 | 相同签名 | — | PASS | 1ms |
| 4 | HMAC 不同 body 不同签名 | 不同 body | 不同签名 | — | PASS | 1ms |
| 5 | SK 不出现在 header | AK/SK 签名 | headers 无 SK | — | PASS | <1ms |
| 6 | Session 凭证加载 | env 变量 | 正确 mode/token | env override | PASS | 1ms |
| 7 | 自定义 header name | `SESSION_HEADER_NAME` | 自定义名 | env override | PASS | <1ms |
| 8 | AK/SK 凭证加载 | env 变量 | 正确 AK/SK | env override | PASS | <1ms |
| 9 | 缺少 Session Token | 无 env | 抛异常 | env override | PASS | 1ms |
| 10 | 缺少 AK/SK | 无 env | 抛异常 | env override | PASS | <1ms |
| 11 | 拒绝 HTTP 端点 | `http://...` | 抛异常 | — | PASS | 2ms |
| 12 | 接受 HTTPS 端点 | `https://...` | 正常创建 | — | PASS | <1ms |
| 13 | Session 注入 GET header | GET /data | Authorization 头 | fetchSpy | PASS | 2ms |
| 14 | Token 不泄露到 body | POST 请求 | body 无 token | fetchSpy | PASS | <1ms |
| 15 | AK/SK HMAC 签名注入 | GET /data | X-Access-Key + X-Signature | fetchSpy | PASS | 2ms |
| 16 | SK 不出现在请求头 | AK/SK 请求 | 所有 header 无 SK | fetchSpy | PASS | <1ms |
| 17 | HTTP 4xx/5xx 结构化错误 | 403 响应 | `{ error: true }` | fetchSpy | PASS | <1ms |
| 18 | 网络故障重试 3 次后 502 | 网络错误 | 502 + 3 次调用 | fetchSpy | PASS | 3.7s |
| 19 | 熔断器触发 | 连续失败 | CircuitBreaker OPEN | fetchSpy | PASS | 6.5s |
| 20 | 超时中断 | 慢响应 | 502 | fetchSpy | PASS | 3.5s |
| 21 | 脱敏 flat 对象 | `{authorization: "..."}` | `***` 替换 | — | PASS | 2ms |
| 22 | 脱敏嵌套对象 | `{headers: {cookie: "..."}}` | 嵌套 mask | — | PASS | <1ms |
| 23 | 脱敏数组 | `[{token: "..."}]` | 数组内 mask | — | PASS | <1ms |
| 24 | 原始类型透传 | 字符串/数字/null | 原样返回 | — | PASS | <1ms |
| 25 | 密码字段脱敏 | `{password: "..."}` | mask 替换 | — | PASS | <1ms |

## 4. System-Level Resilience Tests
| # | Fault Type | Target | Expected | Actual | Result |
|---|-----------|--------|----------|--------|--------|
| 1 | Network failure | fetch mock | Retry 3x → 502 | Retry 3x → 502 | PASS |
| 2 | Request timeout | AbortController | Timeout → retry → 502 | Timeout → retry → 502 | PASS |
| 3 | Cascading failure | Circuit breaker | Open after 5 failures | Open after 5+ failures | PASS |
| 4 | HTTPS enforcement | Constructor | Reject http:// | Threw error | PASS |

## 5. Summary
- **Test Files**: 3 passed (auth, client, sanitize/server)
- **Tests**: 25 passed, 0 failed
- **Duration**: 14.2s
- **Security Checks**: SK 不泄露 ✅, Token 不入 body ✅, 返回值脱敏 ✅, HTTPS 强制 ✅, 熔断器 ✅
