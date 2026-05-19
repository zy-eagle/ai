import * as os from 'os';
import * as path from 'path';

export type Platform = 'win32' | 'linux' | 'darwin';

export function getPlatform(): Platform {
  return os.platform() as Platform;
}

export function getConfigDir(): string {
  const platform = getPlatform();
  const home = os.homedir();

  switch (platform) {
    case 'win32':
      return path.join(process.env.APPDATA || path.join(home, 'AppData', 'Roaming'), 'cursor-im-bridge');
    case 'darwin':
      return path.join(home, 'Library', 'Application Support', 'cursor-im-bridge');
    case 'linux':
      return path.join(process.env.XDG_CONFIG_HOME || path.join(home, '.config'), 'cursor-im-bridge');
    default:
      return path.join(home, '.cursor-im-bridge');
  }
}

export function getDataDir(): string {
  const platform = getPlatform();
  const home = os.homedir();

  switch (platform) {
    case 'win32':
      return path.join(process.env.LOCALAPPDATA || path.join(home, 'AppData', 'Local'), 'cursor-im-bridge');
    case 'darwin':
      return path.join(home, 'Library', 'Application Support', 'cursor-im-bridge', 'data');
    case 'linux':
      return path.join(process.env.XDG_DATA_HOME || path.join(home, '.local', 'share'), 'cursor-im-bridge');
    default:
      return path.join(home, '.cursor-im-bridge', 'data');
  }
}

export function getLogDir(): string {
  const platform = getPlatform();
  const home = os.homedir();

  switch (platform) {
    case 'win32':
      return path.join(process.env.LOCALAPPDATA || path.join(home, 'AppData', 'Local'), 'cursor-im-bridge', 'logs');
    case 'darwin':
      return path.join(home, 'Library', 'Logs', 'cursor-im-bridge');
    case 'linux':
      return path.join(process.env.XDG_STATE_HOME || path.join(home, '.local', 'state'), 'cursor-im-bridge', 'logs');
    default:
      return path.join(home, '.cursor-im-bridge', 'logs');
  }
}

/** 获取适合当前平台的通知命令 (可选, 用于系统级通知) */
export function getNotifyCommand(title: string, body: string): string[] | null {
  const platform = getPlatform();
  switch (platform) {
    case 'win32':
      return [
        'powershell',
        '-Command',
        `[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms');` +
        `[System.Windows.Forms.MessageBox]::Show('${body.replace(/'/g, "''")}','${title.replace(/'/g, "''")}')`,
      ];
    case 'darwin':
      return ['osascript', '-e', `display notification "${body}" with title "${title}"`];
    case 'linux':
      return ['notify-send', title, body];
    default:
      return null;
  }
}
