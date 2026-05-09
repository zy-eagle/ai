package main

import (
	"encoding/json"
	"fmt"
	"os"
)

func CreateServer() (*MCPServer, error) {
	executableDir, err := os.Getwd()
	if err != nil {
		executableDir = "."
	}
	if exePath, err := os.Executable(); err == nil {
		executableDir = filepath_Dir(exePath)
	}

	credentials, err := LoadCredentials(executableDir)
	if err != nil {
		return nil, fmt.Errorf("load credentials: %w", err)
	}

	baseURL := os.Getenv("BASE_URL")
	if baseURL == "" {
		baseURL = "https://mydomain.com"
	}
	allowHTTP := os.Getenv("ALLOW_HTTP") == "true"

	client, err := NewAuthenticatedClient(baseURL, credentials, allowHTTP)
	if err != nil {
		return nil, fmt.Errorf("create client: %w", err)
	}

	authDesc := ""
	switch credentials.Mode {
	case AuthModeSession:
		authDesc = "session (token header)"
	case AuthModeAKSK:
		authDesc = fmt.Sprintf("aksk (AK: %s)", MaskSecret(credentials.AKSK.AccessKey))
	}

	server := NewMCPServer("query-server", "0.1.0",
		fmt.Sprintf(
			"Authenticated data query server. Auth mode: %s. "+
				"Credentials are managed server-side — never ask the user for tokens or keys. "+
				"Use query_data to fetch data from the remote API.", authDesc))

	server.RegisterTool(
		ToolDefinition{
			Name:  "query_data",
			Title: "Query Data",
			Description: "Fetch data from the authenticated API. " +
				"The server handles authentication automatically — do NOT include any auth tokens or credentials in parameters.",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{
						"type":        "string",
						"default":     "/",
						"description": `API endpoint path, e.g. "/users" or "/data/reports"`,
					},
					"method": map[string]any{
						"type":        "string",
						"enum":        []string{"GET", "POST", "PUT", "DELETE", "PATCH"},
						"default":     "GET",
						"description": "HTTP method",
					},
					"query": map[string]any{
						"type":        "object",
						"description": "URL query parameters as key-value pairs",
						"additionalProperties": map[string]any{"type": "string"},
					},
					"body": map[string]any{
						"type":        "object",
						"description": "Request body for POST/PUT/PATCH requests",
					},
				},
			},
		},
		func(args map[string]any) *ToolResult {
			opts := QueryOptions{}
			if v, ok := args["path"].(string); ok {
				opts.Path = v
			}
			if v, ok := args["method"].(string); ok {
				opts.Method = v
			}
			if v, ok := args["query"].(map[string]any); ok {
				opts.Query = make(map[string]string, len(v))
				for k, val := range v {
					opts.Query[k] = fmt.Sprint(val)
				}
			}
			if v, ok := args["body"].(map[string]any); ok {
				opts.Body = v
			}

			resp, err := client.Request(opts)
			if err != nil {
				return &ToolResult{
					Content: []ContentItem{{Type: "text", Text: toJSON(map[string]any{"error": true, "message": err.Error()})}},
					IsError: true,
				}
			}

			sanitized := SanitizeResponse(resp.Data)
			return &ToolResult{
				Content: []ContentItem{{
					Type: "text",
					Text:  toJSON(map[string]any{"status": resp.Status, "data": sanitized}),
				}},
			}
		},
	)

	server.RegisterTool(
		ToolDefinition{
			Name:        "check_health",
			Title:       "Health Check",
			Description: "Check if the remote API is reachable. No authentication details are exposed.",
			InputSchema: map[string]any{
				"type":                 "object",
				"additionalProperties": false,
			},
		},
		func(args map[string]any) *ToolResult {
			resp, err := client.Request(QueryOptions{Method: "GET", Path: "/", TimeoutMs: 5000})
			if err != nil {
				return &ToolResult{
					Content: []ContentItem{{Type: "text", Text: toJSON(map[string]any{"healthy": false, "error": err.Error()})}},
					IsError: true,
				}
			}

			healthy := resp.Status >= 200 && resp.Status < 500
			return &ToolResult{
				Content: []ContentItem{{
					Type: "text",
					Text:  toJSON(map[string]any{"healthy": healthy, "status": resp.Status, "auth_mode": string(credentials.Mode)}),
				}},
			}
		},
	)

	return server, nil
}

func filepath_Dir(path string) string {
	for i := len(path) - 1; i >= 0; i-- {
		if path[i] == '/' || path[i] == '\\' {
			return path[:i]
		}
	}
	return "."
}

func toJSON(v any) string {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return fmt.Sprintf(`{"error":true,"message":"marshal error: %s"}`, err.Error())
	}
	return string(b)
}
