package main

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type AuthMode string

const (
	AuthModeSession AuthMode = "session"
	AuthModeAKSK    AuthMode = "aksk"
)

type SessionCredentials struct {
	Token      string
	HeaderName string
}

type AKSKCredentials struct {
	AccessKey string
	SecretKey string
}

type Credentials struct {
	Mode    AuthMode
	Session *SessionCredentials
	AKSK    *AKSKCredentials
}

func loadEnvFile(path string) {
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		idx := strings.Index(line, "=")
		if idx == -1 {
			continue
		}
		key := strings.TrimSpace(line[:idx])
		value := strings.TrimSpace(line[idx+1:])
		value = strings.Trim(value, `"'`)
		if _, exists := os.LookupEnv(key); !exists {
			os.Setenv(key, value)
		}
	}
}

func LoadCredentials(executableDir string) (*Credentials, error) {
	defaultEnv := filepath.Join(executableDir, ".env")
	if envFile := os.Getenv("ENV_FILE"); envFile != "" {
		if !filepath.IsAbs(envFile) {
			envFile = filepath.Join(executableDir, envFile)
		}
		loadEnvFile(envFile)
	} else {
		loadEnvFile(defaultEnv)
	}

	mode := AuthMode(strings.TrimSpace(os.Getenv("AUTH_MODE")))
	if mode == "" {
		mode = AuthModeSession
	}

	if mode == AuthModeAKSK {
		ak := strings.TrimSpace(os.Getenv("ACCESS_KEY"))
		sk := strings.TrimSpace(os.Getenv("SECRET_KEY"))
		if ak == "" || sk == "" {
			return nil, fmt.Errorf("AUTH_MODE=aksk requires ACCESS_KEY and SECRET_KEY environment variables")
		}
		return &Credentials{
			Mode: AuthModeAKSK,
			AKSK: &AKSKCredentials{AccessKey: ak, SecretKey: sk},
		}, nil
	}

	token := strings.TrimSpace(os.Getenv("SESSION_TOKEN"))
	headerName := os.Getenv("SESSION_HEADER_NAME")
	if headerName == "" {
		headerName = "Authorization"
	}
	if token == "" {
		return nil, fmt.Errorf("AUTH_MODE=session requires SESSION_TOKEN environment variable")
	}
	return &Credentials{
		Mode:    AuthModeSession,
		Session: &SessionCredentials{Token: token, HeaderName: headerName},
	}, nil
}

func SignRequest(creds *AKSKCredentials, method, url, body, timestamp string) map[string]string {
	stringToSign := strings.Join([]string{
		strings.ToUpper(method), url, timestamp, body,
	}, "\n")

	mac := hmac.New(sha256.New, []byte(creds.SecretKey))
	mac.Write([]byte(stringToSign))
	signature := hex.EncodeToString(mac.Sum(nil))

	return map[string]string{
		"X-Access-Key": creds.AccessKey,
		"X-Timestamp":  timestamp,
		"X-Signature":  signature,
	}
}

func MaskSecret(value string) string {
	if len(value) <= 6 {
		return "***"
	}
	return value[:3] + "***" + value[len(value)-3:]
}
