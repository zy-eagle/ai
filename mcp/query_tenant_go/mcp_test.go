package main

import (
	"bytes"
	"encoding/json"
	"testing"
)

func TestMCPServer_Initialize(t *testing.T) {
	var buf bytes.Buffer
	srv := NewMCPServer("test-server", "1.0.0", "Test instructions")
	srv.writer = &buf

	req := &JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      float64(1),
		Method:  "initialize",
		Params:  json.RawMessage(`{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}`),
	}
	srv.handleRequest(req)

	var resp JSONRPCResponse
	if err := json.Unmarshal(buf.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}

	if resp.ID != float64(1) {
		t.Errorf("id = %v, want 1", resp.ID)
	}
	if resp.Error != nil {
		t.Fatalf("unexpected error: %v", resp.Error)
	}

	result := resp.Result.(map[string]any)
	if result["protocolVersion"] != protocolVersion {
		t.Errorf("protocolVersion = %v", result["protocolVersion"])
	}
	if result["instructions"] != "Test instructions" {
		t.Errorf("instructions = %v", result["instructions"])
	}

	serverInfo := result["serverInfo"].(map[string]any)
	if serverInfo["name"] != "test-server" {
		t.Errorf("serverInfo.name = %v", serverInfo["name"])
	}
}

func TestMCPServer_ToolsList(t *testing.T) {
	var buf bytes.Buffer
	srv := NewMCPServer("test", "1.0.0", "")
	srv.writer = &buf

	srv.RegisterTool(ToolDefinition{
		Name:        "my_tool",
		Title:       "My Tool",
		Description: "A test tool",
		InputSchema: map[string]any{"type": "object"},
	}, func(args map[string]any) *ToolResult {
		return &ToolResult{Content: []ContentItem{{Type: "text", Text: "ok"}}}
	})

	req := &JSONRPCRequest{JSONRPC: "2.0", ID: float64(2), Method: "tools/list"}
	srv.handleRequest(req)

	var resp JSONRPCResponse
	json.Unmarshal(buf.Bytes(), &resp)

	result := resp.Result.(map[string]any)
	tools := result["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools count = %d, want 1", len(tools))
	}

	tool := tools[0].(map[string]any)
	if tool["name"] != "my_tool" {
		t.Errorf("tool name = %v", tool["name"])
	}
}

func TestMCPServer_ToolsCall(t *testing.T) {
	var buf bytes.Buffer
	srv := NewMCPServer("test", "1.0.0", "")
	srv.writer = &buf

	srv.RegisterTool(ToolDefinition{
		Name:        "echo",
		InputSchema: map[string]any{"type": "object"},
	}, func(args map[string]any) *ToolResult {
		msg, _ := args["message"].(string)
		return &ToolResult{Content: []ContentItem{{Type: "text", Text: "echo: " + msg}}}
	})

	params, _ := json.Marshal(ToolCallParams{Name: "echo", Arguments: map[string]any{"message": "hello"}})
	req := &JSONRPCRequest{JSONRPC: "2.0", ID: float64(3), Method: "tools/call", Params: params}
	srv.handleRequest(req)

	var resp JSONRPCResponse
	json.Unmarshal(buf.Bytes(), &resp)

	result := resp.Result.(map[string]any)
	content := result["content"].([]any)
	item := content[0].(map[string]any)
	if item["text"] != "echo: hello" {
		t.Errorf("text = %v", item["text"])
	}
}

func TestMCPServer_UnknownTool(t *testing.T) {
	var buf bytes.Buffer
	srv := NewMCPServer("test", "1.0.0", "")
	srv.writer = &buf

	params, _ := json.Marshal(ToolCallParams{Name: "nonexistent"})
	req := &JSONRPCRequest{JSONRPC: "2.0", ID: float64(4), Method: "tools/call", Params: params}
	srv.handleRequest(req)

	var resp JSONRPCResponse
	json.Unmarshal(buf.Bytes(), &resp)

	if resp.Error == nil {
		t.Fatal("expected error for unknown tool")
	}
	if resp.Error.Code != -32602 {
		t.Errorf("error code = %d, want -32602", resp.Error.Code)
	}
}

func TestMCPServer_UnknownMethod(t *testing.T) {
	var buf bytes.Buffer
	srv := NewMCPServer("test", "1.0.0", "")
	srv.writer = &buf

	req := &JSONRPCRequest{JSONRPC: "2.0", ID: float64(5), Method: "unknown/method"}
	srv.handleRequest(req)

	var resp JSONRPCResponse
	json.Unmarshal(buf.Bytes(), &resp)

	if resp.Error == nil {
		t.Fatal("expected error for unknown method")
	}
	if resp.Error.Code != -32601 {
		t.Errorf("error code = %d, want -32601", resp.Error.Code)
	}
}

func TestMCPServer_Ping(t *testing.T) {
	var buf bytes.Buffer
	srv := NewMCPServer("test", "1.0.0", "")
	srv.writer = &buf

	req := &JSONRPCRequest{JSONRPC: "2.0", ID: float64(6), Method: "ping"}
	srv.handleRequest(req)

	var resp JSONRPCResponse
	json.Unmarshal(buf.Bytes(), &resp)

	if resp.Error != nil {
		t.Fatalf("unexpected error: %v", resp.Error)
	}
}
