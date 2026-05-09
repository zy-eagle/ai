package main

import "testing"

func TestSanitize_FlatObject(t *testing.T) {
	input := map[string]any{
		"name":          "test",
		"authorization": "Bearer secret-token-123456",
		"count":         float64(5),
	}
	result := SanitizeResponse(input).(map[string]any)

	if result["name"] != "test" {
		t.Errorf("name = %v", result["name"])
	}
	if result["count"] != float64(5) {
		t.Errorf("count = %v", result["count"])
	}
	if s, ok := result["authorization"].(string); !ok || s == "Bearer secret-token-123456" {
		t.Errorf("authorization not masked: %v", result["authorization"])
	}
}

func TestSanitize_NestedObject(t *testing.T) {
	input := map[string]any{
		"headers": map[string]any{
			"cookie": "sid=abc123456789",
		},
		"data": map[string]any{"ok": true},
	}
	result := SanitizeResponse(input).(map[string]any)
	headers := result["headers"].(map[string]any)

	if s, ok := headers["cookie"].(string); !ok || s == "sid=abc123456789" {
		t.Errorf("cookie not masked: %v", headers["cookie"])
	}
}

func TestSanitize_Array(t *testing.T) {
	input := []any{
		map[string]any{"token": "my-long-secret-token-value"},
		map[string]any{"name": "safe"},
	}
	result := SanitizeResponse(input).([]any)

	item0 := result[0].(map[string]any)
	if s, ok := item0["token"].(string); !ok || s == "my-long-secret-token-value" {
		t.Errorf("token not masked: %v", item0["token"])
	}

	item1 := result[1].(map[string]any)
	if item1["name"] != "safe" {
		t.Errorf("name = %v", item1["name"])
	}
}

func TestSanitize_Primitives(t *testing.T) {
	if SanitizeResponse("hello") != "hello" {
		t.Error("string passthrough failed")
	}
	if SanitizeResponse(float64(42)) != float64(42) {
		t.Error("number passthrough failed")
	}
	if SanitizeResponse(nil) != nil {
		t.Error("nil passthrough failed")
	}
}

func TestSanitize_Password(t *testing.T) {
	input := map[string]any{
		"user":     "admin",
		"password": "super-secret-password-123",
	}
	result := SanitizeResponse(input).(map[string]any)

	if result["user"] != "admin" {
		t.Errorf("user = %v", result["user"])
	}
	if s, ok := result["password"].(string); !ok || s == "super-secret-password-123" {
		t.Errorf("password not masked: %v", result["password"])
	}
}
