import * as vscode from 'vscode';

/**
 * 安全凭证存储 — 使用 VS Code SecretStorage API
 * 凭证加密存储在操作系统钥匙链中：
 * - Windows: Windows Credential Manager
 * - macOS: Keychain
 * - Linux: libsecret (GNOME Keyring / KDE Wallet)
 */
export class SecureStore {
  private static readonly PREFIX = 'cursorImBridge.';

  constructor(private secrets: vscode.SecretStorage) {}

  async get(key: string): Promise<string | undefined> {
    return this.secrets.get(SecureStore.PREFIX + key);
  }

  async set(key: string, value: string): Promise<void> {
    await this.secrets.store(SecureStore.PREFIX + key, value);
  }

  async delete(key: string): Promise<void> {
    await this.secrets.delete(SecureStore.PREFIX + key);
  }

  async getAdapterSecret(adapterType: string, field: string): Promise<string | undefined> {
    return this.get(`${adapterType}.${field}`);
  }

  async setAdapterSecret(adapterType: string, field: string, value: string): Promise<void> {
    await this.set(`${adapterType}.${field}`, value);
  }

  async deleteAdapterSecrets(adapterType: string): Promise<void> {
    const keys = ['appId', 'appSecret', 'token', 'botToken', 'appKey', 'corpId', 'corpSecret', 'credentials'];
    for (const key of keys) {
      await this.delete(`${adapterType}.${key}`);
    }
  }
}
