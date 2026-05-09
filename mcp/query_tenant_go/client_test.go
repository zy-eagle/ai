package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func newSessionCreds() *Credentials {
	return &Credentials{
		Mode:    AuthModeSession,
		Session: &SessionCredentials{Token: "Bearer test-session-token", HeaderName: "Authorization"},
	}
}

func newAKSKCreds() *Credentials {
	return &Credentials{
		Mode: AuthModeAKSK,
		AKSK: &AKSKCredentials{AccessKey: "AK_TEST", SecretKey: "SK_TEST"},
	}
}

func TestClient_RejectsHTTP(t *testing.T) {
	_, err := NewAuthenticatedClient("http://insecure.com", newSessionCreds(), false)
	if err == nil {
		t.Error("expected error for non-HTTPS URL")
	}
}

func TestClient_AllowsHTTP(t *testing.T) {
	_, err := NewAuthenticatedClient("http://internal.local", newSessionCreds(), true)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestClient_AcceptsHTTPS(t *testing.T) {
	_, err := NewAuthenticatedClient("https://secure.com", newSessionCreds(), false)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestClient_SessionAuth(t *testing.T) {
	var gotAuth string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"result": "ok"})
	}))
	defer ts.Close()

	client, err := NewAuthenticatedClient(ts.URL, newSessionCreds(), true)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := client.Request(QueryOptions{Path: "/data", Method: "GET"})
	if err != nil {
		t.Fatal(err)
	}

	if resp.Status != 200 {
		t.Errorf("status = %d, want 200", resp.Status)
	}
	if gotAuth != "Bearer test-session-token" {
		t.Errorf("auth header = %q", gotAuth)
	}
}

func TestClient_AKSKAuth(t *testing.T) {
	var gotAK, gotSig, gotTS string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAK = r.Header.Get("X-Access-Key")
		gotSig = r.Header.Get("X-Signature")
		gotTS = r.Header.Get("X-Timestamp")
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"data": []int{1, 2, 3}})
	}))
	defer ts.Close()

	client, err := NewAuthenticatedClient(ts.URL, newAKSKCreds(), true)
	if err != nil {
		t.Fatal(err)
	}

	_, err = client.Request(QueryOptions{Path: "/data"})
	if err != nil {
		t.Fatal(err)
	}

	if gotAK != "AK_TEST" {
		t.Errorf("X-Access-Key = %q", gotAK)
	}
	if gotTS == "" {
		t.Error("X-Timestamp is empty")
	}
	if len(gotSig) != 64 {
		t.Errorf("X-Signature length = %d, want 64", len(gotSig))
	}
}

func TestClient_AKSKNoSecretInHeaders(t *testing.T) {
	var allHeaders http.Header
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		allHeaders = r.Header
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{})
	}))
	defer ts.Close()

	client, err := NewAuthenticatedClient(ts.URL, newAKSKCreds(), true)
	if err != nil {
		t.Fatal(err)
	}

	_, err = client.Request(QueryOptions{Path: "/test"})
	if err != nil {
		t.Fatal(err)
	}

	for k, vals := range allHeaders {
		for _, v := range vals {
			if v == "SK_TEST" {
				t.Errorf("secret key exposed in header %q", k)
			}
		}
	}
}

func TestClient_HTTP4xx(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(403)
		json.NewEncoder(w).Encode(map[string]any{"reason": "forbidden"})
	}))
	defer ts.Close()

	client, err := NewAuthenticatedClient(ts.URL, newSessionCreds(), true)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := client.Request(QueryOptions{Path: "/secret"})
	if err != nil {
		t.Fatal(err)
	}

	if resp.Status != 403 {
		t.Errorf("status = %d, want 403", resp.Status)
	}
	data := resp.Data.(map[string]any)
	if data["error"] != true {
		t.Error("expected error=true")
	}
}

func TestClient_RetryExhaustion(t *testing.T) {
	callCount := 0
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		hj, ok := w.(http.Hijacker)
		if ok {
			conn, _, _ := hj.Hijack()
			conn.Close()
			return
		}
		w.WriteHeader(500)
	}))
	defer ts.Close()

	client, err := NewAuthenticatedClient(ts.URL, newSessionCreds(), true)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := client.Request(QueryOptions{Path: "/fail", TimeoutMs: 1000})
	if err != nil {
		if resp == nil {
			t.Fatalf("expected 502 response, got error: %v", err)
		}
	}

	if resp != nil && resp.Status != 502 {
		t.Errorf("status = %d, want 502", resp.Status)
	}
	if callCount < 2 {
		t.Errorf("expected at least 2 calls, got %d", callCount)
	}
}
