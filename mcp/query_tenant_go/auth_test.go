package main

import (
	"os"
	"testing"
)

func TestMaskSecret_Long(t *testing.T) {
	got := MaskSecret("abcdefghij")
	want := "abc***hij"
	if got != want {
		t.Errorf("MaskSecret(%q) = %q, want %q", "abcdefghij", got, want)
	}
}

func TestMaskSecret_Short(t *testing.T) {
	for _, s := range []string{"abc", "", "ab"} {
		got := MaskSecret(s)
		if got != "***" {
			t.Errorf("MaskSecret(%q) = %q, want %q", s, got, "***")
		}
	}
}

func TestSignRequest_Deterministic(t *testing.T) {
	creds := &AKSKCredentials{AccessKey: "AK001", SecretKey: "SK_SECRET"}
	ts := "2026-01-01T00:00:00Z"

	h1 := SignRequest(creds, "GET", "https://api.example.com/data", "", ts)
	h2 := SignRequest(creds, "GET", "https://api.example.com/data", "", ts)

	if h1["X-Access-Key"] != "AK001" {
		t.Errorf("X-Access-Key = %q, want %q", h1["X-Access-Key"], "AK001")
	}
	if h1["X-Timestamp"] != ts {
		t.Errorf("X-Timestamp = %q, want %q", h1["X-Timestamp"], ts)
	}
	if len(h1["X-Signature"]) != 64 {
		t.Errorf("X-Signature length = %d, want 64", len(h1["X-Signature"]))
	}
	if h1["X-Signature"] != h2["X-Signature"] {
		t.Error("same input should produce same signature")
	}
}

func TestSignRequest_DifferentBody(t *testing.T) {
	creds := &AKSKCredentials{AccessKey: "AK001", SecretKey: "SK_SECRET"}
	ts := "2026-01-01T00:00:00Z"

	h1 := SignRequest(creds, "POST", "https://api.example.com/data", `{"a":1}`, ts)
	h2 := SignRequest(creds, "POST", "https://api.example.com/data", `{"a":2}`, ts)

	if h1["X-Signature"] == h2["X-Signature"] {
		t.Error("different body should produce different signature")
	}
}

func TestSignRequest_NoSecretInHeaders(t *testing.T) {
	creds := &AKSKCredentials{AccessKey: "AK001", SecretKey: "SK_SECRET"}
	headers := SignRequest(creds, "GET", "https://api.example.com", "", "2026-01-01T00:00:00Z")

	for k, v := range headers {
		if v == "SK_SECRET" {
			t.Errorf("secret key exposed in header %q", k)
		}
	}
}

func TestLoadCredentials_Session(t *testing.T) {
	t.Setenv("AUTH_MODE", "session")
	t.Setenv("SESSION_TOKEN", "Bearer test-token-123")
	t.Setenv("SESSION_HEADER_NAME", "")

	creds, err := LoadCredentials(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if creds.Mode != AuthModeSession {
		t.Errorf("mode = %q, want %q", creds.Mode, AuthModeSession)
	}
	if creds.Session.Token != "Bearer test-token-123" {
		t.Errorf("token = %q", creds.Session.Token)
	}
	if creds.Session.HeaderName != "Authorization" {
		t.Errorf("headerName = %q", creds.Session.HeaderName)
	}
}

func TestLoadCredentials_SessionCustomHeader(t *testing.T) {
	t.Setenv("AUTH_MODE", "session")
	t.Setenv("SESSION_TOKEN", "my-session")
	t.Setenv("SESSION_HEADER_NAME", "X-Session-Token")

	creds, err := LoadCredentials(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if creds.Session.HeaderName != "X-Session-Token" {
		t.Errorf("headerName = %q", creds.Session.HeaderName)
	}
}

func TestLoadCredentials_AKSK(t *testing.T) {
	t.Setenv("AUTH_MODE", "aksk")
	t.Setenv("ACCESS_KEY", "AK_TEST")
	t.Setenv("SECRET_KEY", "SK_TEST")

	creds, err := LoadCredentials(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if creds.Mode != AuthModeAKSK {
		t.Errorf("mode = %q, want %q", creds.Mode, AuthModeAKSK)
	}
	if creds.AKSK.AccessKey != "AK_TEST" {
		t.Errorf("accessKey = %q", creds.AKSK.AccessKey)
	}
}

func TestLoadCredentials_SessionMissingToken(t *testing.T) {
	t.Setenv("AUTH_MODE", "session")
	t.Setenv("SESSION_TOKEN", "")

	_, err := LoadCredentials(t.TempDir())
	if err == nil {
		t.Error("expected error for missing SESSION_TOKEN")
	}
}

func TestLoadCredentials_AKSKMissingKeys(t *testing.T) {
	t.Setenv("AUTH_MODE", "aksk")
	t.Setenv("ACCESS_KEY", "")
	t.Setenv("SECRET_KEY", "")

	_, err := LoadCredentials(t.TempDir())
	if err == nil {
		t.Error("expected error for missing ACCESS_KEY/SECRET_KEY")
	}
}

func TestLoadEnvFile(t *testing.T) {
	dir := t.TempDir()
	envFile := dir + "/.env"
	os.WriteFile(envFile, []byte("TEST_LOAD_ENV_VAR=hello_world\n"), 0644)

	os.Unsetenv("TEST_LOAD_ENV_VAR")
	loadEnvFile(envFile)

	val := os.Getenv("TEST_LOAD_ENV_VAR")
	if val != "hello_world" {
		t.Errorf("TEST_LOAD_ENV_VAR = %q, want %q", val, "hello_world")
	}
	os.Unsetenv("TEST_LOAD_ENV_VAR")
}
