package main

import "strings"

var sensitiveKeys = map[string]bool{
	"authorization": true, "cookie": true, "set-cookie": true,
	"x-access-key": true, "x-signature": true, "x-timestamp": true,
	"session_token": true, "secret_key": true, "access_key": true,
	"token": true, "api_key": true, "apikey": true, "password": true,
}

func SanitizeResponse(data any) any {
	if data == nil {
		return nil
	}

	switch v := data.(type) {
	case map[string]any:
		result := make(map[string]any, len(v))
		for key, val := range v {
			if sensitiveKeys[strings.ToLower(key)] {
				if s, ok := val.(string); ok {
					result[key] = MaskSecret(s)
				} else {
					result[key] = "***"
				}
			} else {
				result[key] = SanitizeResponse(val)
			}
		}
		return result

	case []any:
		result := make([]any, len(v))
		for i, item := range v {
			result[i] = SanitizeResponse(item)
		}
		return result

	default:
		return data
	}
}
