import * as vscode from 'vscode';
import { IMBridge } from './core/im-bridge';
import { ConnectionStatus } from './types';

let bridge: IMBridge | undefined;

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  bridge = new IMBridge(context);

  context.subscriptions.push(
    vscode.commands.registerCommand('cursorImBridge.connect', async () => {
      if (!bridge) return;
      const adapters = bridge.getAdapters();
      if (adapters.length === 0) {
        const action = await vscode.window.showInformationMessage(
          'No IM adapters configured. Would you like to configure one?',
          'Configure'
        );
        if (action === 'Configure') {
          vscode.commands.executeCommand('cursorImBridge.configure');
        }
        return;
      }

      const disconnected = adapters.filter(
        (a) => a.status !== ConnectionStatus.Connected
      );

      if (disconnected.length === 0) {
        vscode.window.showInformationMessage('All adapters are already connected.');
        return;
      }

      if (disconnected.length === 1) {
        await bridge.connectAdapter(disconnected[0].id);
        return;
      }

      const items: Array<{ label: string; description: string; adapterId: string }> = disconnected.map((a) => ({
        label: a.displayName,
        description: a.status,
        adapterId: a.id,
      }));
      items.unshift({ label: 'Connect All', description: 'All disconnected adapters', adapterId: '__all__' });

      const selected = await vscode.window.showQuickPick(items, {
        placeHolder: 'Select adapter to connect',
      });

      if (!selected) return;
      if (selected.adapterId === '__all__') {
        await bridge.connectAll();
      } else {
        await bridge.connectAdapter(selected.adapterId);
      }
    }),

    vscode.commands.registerCommand('cursorImBridge.disconnect', async () => {
      if (!bridge) return;
      await bridge.disconnectAll();
      vscode.window.showInformationMessage('All IM adapters disconnected.');
    }),

    vscode.commands.registerCommand('cursorImBridge.sendMessage', async () => {
      if (!bridge) return;
      const bus = bridge.getMessageBus();
      const connected = bus.getConnectedAdapters();

      if (connected.length === 0) {
        vscode.window.showWarningMessage('No connected adapters. Please connect first.');
        return;
      }

      const adapterItems = connected.map((a) => ({
        label: a.displayName,
        adapterId: a.id,
      }));

      const selectedAdapter = await vscode.window.showQuickPick(adapterItems, {
        placeHolder: 'Select target adapter',
      });
      if (!selectedAdapter) return;

      const adapter = connected.find((a) => a.id === selectedAdapter.adapterId);
      if (!adapter) return;

      const channels = await adapter.getChannels();
      let channelId: string;

      if (channels.length > 0) {
        const channelItems = channels.map((c) => ({
          label: c.name,
          description: c.type,
          channelId: c.id,
        }));
        const selectedChannel = await vscode.window.showQuickPick(channelItems, {
          placeHolder: 'Select channel',
        });
        if (!selectedChannel) return;
        channelId = selectedChannel.channelId;
      } else {
        const input = await vscode.window.showInputBox({
          prompt: 'Enter channel/chat ID',
          placeHolder: 'e.g. oc_xxxxx or @user_id',
        });
        if (!input) return;
        channelId = input;
      }

      const content = await vscode.window.showInputBox({
        prompt: 'Enter message content',
        placeHolder: 'Type your message here...',
      });
      if (!content) return;

      try {
        await bridge.sendMessage(selectedAdapter.adapterId, channelId, content);
        vscode.window.showInformationMessage('Message sent successfully!');
      } catch (err) {
        vscode.window.showErrorMessage(`Failed to send: ${err}`);
      }
    }),

    vscode.commands.registerCommand('cursorImBridge.showPanel', () => {
      if (!bridge) return;
      MessagePanel.createOrShow(context.extensionUri, bridge);
    }),

    vscode.commands.registerCommand('cursorImBridge.configure', () => {
      vscode.commands.executeCommand(
        'workbench.action.openSettings',
        'cursorImBridge'
      );
    }),

    vscode.commands.registerCommand('cursorImBridge.toggleAutoReply', () => {
      if (!bridge) return;
      const current = bridge.isAutoReplyEnabled();
      bridge.setAutoReply(!current);
      vscode.window.showInformationMessage(
        `IM Bridge Auto-Reply: ${!current ? 'ON' : 'OFF'}`
      );
    }),

    vscode.commands.registerCommand('cursorImBridge.processPrompt', async () => {
      if (!bridge) return;
      const bus = bridge.getMessageBus();
      const connected = bus.getConnectedAdapters();

      if (connected.length === 0) {
        vscode.window.showWarningMessage('No connected adapters.');
        return;
      }

      const adapterItems = connected.map((a) => ({
        label: a.displayName,
        adapterId: a.id,
      }));

      const selectedAdapter = await vscode.window.showQuickPick(adapterItems, {
        placeHolder: 'Select adapter to send result to',
      });
      if (!selectedAdapter) return;

      const channelInput = await vscode.window.showInputBox({
        prompt: 'Channel ID to send result to',
      });
      if (!channelInput) return;

      const prompt = await vscode.window.showInputBox({
        prompt: 'Enter prompt for Cursor CLI',
        placeHolder: 'e.g. 帮我分析这段代码的性能问题...',
      });
      if (!prompt) return;

      try {
        vscode.window.showInformationMessage('Processing via Cursor CLI...');
        const result = await bridge.processAndReply(
          selectedAdapter.adapterId,
          channelInput,
          prompt
        );
        if (result) {
          vscode.window.showInformationMessage(
            `Done! Result sent (${(result.cliResult.duration / 1000).toFixed(1)}s)`
          );
        }
      } catch (err) {
        vscode.window.showErrorMessage(`Process failed: ${err}`);
      }
    })
  );

  await bridge.initialize();

  bridge.onMessage((msg) => {
    const notification = `[${msg.adapterType}] ${msg.senderName || msg.senderId}: ${msg.content.slice(0, 100)}`;
    vscode.window.showInformationMessage(notification, 'Reply').then(async (action) => {
      if (action === 'Reply' && bridge) {
        const reply = await vscode.window.showInputBox({
          prompt: `Reply to ${msg.senderName || msg.senderId}`,
        });
        if (reply) {
          await bridge.sendMessage(msg.adapterId, msg.channelId, reply);
        }
      }
    });
  });
}

export function deactivate(): void {
  bridge?.dispose();
  bridge = undefined;
}

class MessagePanel {
  public static currentPanel: MessagePanel | undefined;
  private static readonly viewType = 'imBridgeMessages';
  private readonly panel: vscode.WebviewPanel;
  private readonly bridge: IMBridge;
  private disposables: vscode.Disposable[] = [];

  static createOrShow(extensionUri: vscode.Uri, bridge: IMBridge): void {
    const column = vscode.window.activeTextEditor
      ? vscode.window.activeTextEditor.viewColumn
      : undefined;

    if (MessagePanel.currentPanel) {
      MessagePanel.currentPanel.panel.reveal(column);
      return;
    }

    const panel = vscode.window.createWebviewPanel(
      MessagePanel.viewType,
      'IM Bridge Messages',
      column || vscode.ViewColumn.Beside,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
      }
    );

    MessagePanel.currentPanel = new MessagePanel(panel, bridge);
  }

  private constructor(panel: vscode.WebviewPanel, bridge: IMBridge) {
    this.panel = panel;
    this.bridge = bridge;

    this.update();

    this.panel.onDidDispose(() => this.dispose(), null, this.disposables);

    this.panel.webview.onDidReceiveMessage(
      async (message) => {
        if (message.command === 'send') {
          try {
            await this.bridge.sendMessage(
              message.adapterId,
              message.channelId,
              message.content
            );
            this.update();
          } catch (err) {
            this.panel.webview.postMessage({
              command: 'error',
              text: String(err),
            });
          }
        } else if (message.command === 'refresh') {
          this.update();
        }
      },
      null,
      this.disposables
    );

    const msgDisposable = this.bridge.onMessage(() => {
      this.update();
    });
    this.disposables.push(msgDisposable);
  }

  private update(): void {
    const bus = this.bridge.getMessageBus();
    const messages = bus.getHistory(undefined, undefined, 100);
    const adapters = this.bridge.getAdapters().map((a) => ({
      id: a.id,
      type: a.type,
      name: a.displayName,
      status: a.status,
    }));

    this.panel.webview.html = this.getHtml(messages, adapters);
  }

  private getHtml(
    messages: Array<{ adapterId: string; adapterType: string; direction: string; channelId: string; senderName?: string; senderId: string; content: string; timestamp: number }>,
    adapters: Array<{ id: string; type: string; name: string; status: string }>
  ): string {
    const msgHtml = messages
      .slice(-50)
      .map(
        (m) => `
      <div class="message ${m.direction}">
        <span class="badge">${m.adapterType}</span>
        <span class="sender">${this.escapeHtml(m.senderName || m.senderId)}</span>
        <span class="time">${new Date(m.timestamp).toLocaleTimeString()}</span>
        <div class="content">${this.escapeHtml(m.content)}</div>
      </div>`
      )
      .join('');

    const adapterOptions = adapters
      .filter((a) => a.status === 'connected')
      .map((a) => `<option value="${a.id}">${this.escapeHtml(a.name)}</option>`)
      .join('');

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <style>
    body { font-family: var(--vscode-font-family); padding: 10px; }
    .message { padding: 8px; margin: 4px 0; border-radius: 6px; border: 1px solid var(--vscode-panel-border); }
    .message.inbound { background: var(--vscode-editor-background); }
    .message.outbound { background: var(--vscode-badge-background); color: var(--vscode-badge-foreground); }
    .badge { font-size: 10px; padding: 2px 6px; border-radius: 3px; background: var(--vscode-badge-background); color: var(--vscode-badge-foreground); }
    .sender { font-weight: bold; margin-left: 6px; }
    .time { font-size: 11px; color: var(--vscode-descriptionForeground); float: right; }
    .content { margin-top: 4px; white-space: pre-wrap; }
    .compose { position: sticky; bottom: 0; background: var(--vscode-editor-background); padding: 10px 0; display: flex; gap: 8px; }
    .compose select, .compose input, .compose button { padding: 6px 10px; }
    .compose input { flex: 1; }
    .adapters { margin-bottom: 10px; }
    .status { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; }
    .status.connected { background: #4caf50; }
    .status.disconnected { background: #9e9e9e; }
    .status.error { background: #f44336; }
  </style>
</head>
<body>
  <div class="adapters">
    ${adapters.map((a) => `<span class="status ${a.status}"></span>${this.escapeHtml(a.name)} `).join('| ')}
  </div>
  <div id="messages">${msgHtml || '<p style="color:var(--vscode-descriptionForeground)">No messages yet. Connect an adapter to start receiving messages.</p>'}</div>
  <div class="compose">
    <select id="adapter">${adapterOptions || '<option disabled>No connected adapters</option>'}</select>
    <input id="channelId" placeholder="Channel ID" />
    <input id="msgInput" placeholder="Type a message..." />
    <button onclick="sendMsg()">Send</button>
  </div>
  <script>
    const vscode = acquireVsCodeApi();
    function sendMsg() {
      const adapterId = document.getElementById('adapter').value;
      const channelId = document.getElementById('channelId').value;
      const content = document.getElementById('msgInput').value;
      if (!adapterId || !channelId || !content) return;
      vscode.postMessage({ command: 'send', adapterId, channelId, content });
      document.getElementById('msgInput').value = '';
    }
    document.getElementById('msgInput').addEventListener('keypress', (e) => {
      if (e.key === 'Enter') sendMsg();
    });
    window.addEventListener('message', (event) => {
      if (event.data.command === 'error') {
        alert(event.data.text);
      }
    });
  </script>
</body>
</html>`;
  }

  private escapeHtml(text: string): string {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  private dispose(): void {
    MessagePanel.currentPanel = undefined;
    this.panel.dispose();
    this.disposables.forEach((d) => d.dispose());
    this.disposables = [];
  }
}
