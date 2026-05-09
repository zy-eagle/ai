package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sync"
)

const protocolVersion = "2025-11-25"

type JSONRPCRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      any             `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type JSONRPCResponse struct {
	JSONRPC string     `json:"jsonrpc"`
	ID      any        `json:"id,omitempty"`
	Result  any        `json:"result,omitempty"`
	Error   *RPCError  `json:"error,omitempty"`
}

type RPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    any    `json:"data,omitempty"`
}

type ToolDefinition struct {
	Name        string `json:"name"`
	Title       string `json:"title,omitempty"`
	Description string `json:"description,omitempty"`
	InputSchema any    `json:"inputSchema"`
}

type ToolCallParams struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments,omitempty"`
}

type ToolResult struct {
	Content []ContentItem `json:"content"`
	IsError bool          `json:"isError,omitempty"`
}

type ContentItem struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type ToolHandler func(args map[string]any) *ToolResult

type MCPServer struct {
	name         string
	version      string
	instructions string
	tools        []ToolDefinition
	handlers     map[string]ToolHandler
	writeMu      sync.Mutex
	writer       io.Writer
}

func NewMCPServer(name, version, instructions string) *MCPServer {
	return &MCPServer{
		name:         name,
		version:      version,
		instructions: instructions,
		handlers:     make(map[string]ToolHandler),
		writer:       os.Stdout,
	}
}

func (s *MCPServer) RegisterTool(def ToolDefinition, handler ToolHandler) {
	s.tools = append(s.tools, def)
	s.handlers[def.Name] = handler
}

func (s *MCPServer) Serve() error {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 1024*1024), 10*1024*1024)

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		var req JSONRPCRequest
		if err := json.Unmarshal(line, &req); err != nil {
			s.sendError(nil, -32700, "Parse error", err.Error())
			continue
		}

		s.handleRequest(&req)
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("stdin read error: %w", err)
	}
	return nil
}

func (s *MCPServer) handleRequest(req *JSONRPCRequest) {
	switch req.Method {
	case "initialize":
		s.handleInitialize(req)
	case "notifications/initialized":
		// no-op acknowledgement
	case "tools/list":
		s.handleToolsList(req)
	case "tools/call":
		s.handleToolsCall(req)
	case "ping":
		s.sendResult(req.ID, map[string]any{})
	default:
		if req.ID != nil {
			s.sendError(req.ID, -32601, "Method not found", req.Method)
		}
	}
}

func (s *MCPServer) handleInitialize(req *JSONRPCRequest) {
	result := map[string]any{
		"protocolVersion": protocolVersion,
		"capabilities": map[string]any{
			"tools": map[string]any{},
		},
		"serverInfo": map[string]any{
			"name":    s.name,
			"version": s.version,
		},
	}
	if s.instructions != "" {
		result["instructions"] = s.instructions
	}
	s.sendResult(req.ID, result)
}

func (s *MCPServer) handleToolsList(req *JSONRPCRequest) {
	s.sendResult(req.ID, map[string]any{
		"tools": s.tools,
	})
}

func (s *MCPServer) handleToolsCall(req *JSONRPCRequest) {
	var params ToolCallParams
	if req.Params != nil {
		if err := json.Unmarshal(req.Params, &params); err != nil {
			s.sendError(req.ID, -32602, "Invalid params", err.Error())
			return
		}
	}

	handler, ok := s.handlers[params.Name]
	if !ok {
		s.sendError(req.ID, -32602, "Unknown tool: "+params.Name, nil)
		return
	}

	result := handler(params.Arguments)
	s.sendResult(req.ID, result)
}

func (s *MCPServer) sendResult(id any, result any) {
	s.writeJSON(JSONRPCResponse{
		JSONRPC: "2.0",
		ID:      id,
		Result:  result,
	})
}

func (s *MCPServer) sendError(id any, code int, message string, data any) {
	s.writeJSON(JSONRPCResponse{
		JSONRPC: "2.0",
		ID:      id,
		Error:   &RPCError{Code: code, Message: message, Data: data},
	})
}

func (s *MCPServer) writeJSON(v any) {
	data, err := json.Marshal(v)
	if err != nil {
		fmt.Fprintf(os.Stderr, "marshal error: %v\n", err)
		return
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	s.writer.Write(data)
	s.writer.Write([]byte("\n"))
}
