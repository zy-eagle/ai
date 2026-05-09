package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

const (
	defaultTimeoutMs  = 10_000
	maxRetries        = 3
	circuitThreshold  = 5
	circuitResetMs    = 30_000
)

type QueryOptions struct {
	Method    string            `json:"method,omitempty"`
	Path      string            `json:"path,omitempty"`
	Query     map[string]string `json:"query,omitempty"`
	Body      map[string]any    `json:"body,omitempty"`
	TimeoutMs int               `json:"timeoutMs,omitempty"`
}

type APIResponse struct {
	Status int `json:"status"`
	Data   any `json:"data"`
}

type circuitState struct {
	mu          sync.Mutex
	failures    int
	lastFailure time.Time
	isOpen      bool
}

type AuthenticatedClient struct {
	baseUrl     string
	credentials *Credentials
	circuit     circuitState
	httpClient  *http.Client
}

func NewAuthenticatedClient(baseUrl string, credentials *Credentials, allowHTTP bool) (*AuthenticatedClient, error) {
	if !strings.HasPrefix(baseUrl, "https://") && !allowHTTP {
		return nil, fmt.Errorf(
			"refusing to connect to non-HTTPS endpoint: %s. "+
				"Set ALLOW_HTTP=true for internal/dev networks, or use an https:// address", baseUrl)
	}
	baseUrl = strings.TrimRight(baseUrl, "/")

	return &AuthenticatedClient{
		baseUrl:     baseUrl,
		credentials: credentials,
		httpClient:  &http.Client{},
	}, nil
}

func (c *AuthenticatedClient) Request(opts QueryOptions) (*APIResponse, error) {
	if err := c.checkCircuit(); err != nil {
		return nil, err
	}

	method := strings.ToUpper(opts.Method)
	if method == "" {
		method = "GET"
	}
	path := opts.Path
	if path == "" {
		path = "/"
	}
	timeoutMs := opts.TimeoutMs
	if timeoutMs <= 0 {
		timeoutMs = defaultTimeoutMs
	}

	reqURL, err := c.buildURL(path, opts.Query)
	if err != nil {
		return nil, fmt.Errorf("build url: %w", err)
	}

	var bodyStr string
	if opts.Body != nil {
		b, err := json.Marshal(opts.Body)
		if err != nil {
			return nil, fmt.Errorf("marshal body: %w", err)
		}
		bodyStr = string(b)
	}

	var lastErr error

	for attempt := range maxRetries {
		headers := map[string]string{
			"Content-Type": "application/json",
			"Accept":       "application/json",
		}
		c.applyAuth(headers, method, reqURL, bodyStr)

		resp, err := c.doHTTP(method, reqURL, bodyStr, headers, time.Duration(timeoutMs)*time.Millisecond)
		if err != nil {
			lastErr = err
			c.recordFailure()

			if attempt < maxRetries-1 {
				backoff := math.Min(float64(1000*int(1<<attempt)), 8000)
				jitter := rand.Float64() * backoff * 0.3
				time.Sleep(time.Duration(backoff+jitter) * time.Millisecond)
			}
			continue
		}

		c.recordSuccess()

		if resp.Status < 200 || resp.Status >= 300 {
			return &APIResponse{
				Status: resp.Status,
				Data: map[string]any{
					"error":   true,
					"message": fmt.Sprintf("HTTP %d", resp.Status),
					"details": resp.Data,
				},
			}, nil
		}

		return resp, nil
	}

	c.recordFailure()
	errMsg := "unknown error"
	if lastErr != nil {
		errMsg = lastErr.Error()
	}

	return &APIResponse{
		Status: 502,
		Data: map[string]any{
			"error":   true,
			"message": fmt.Sprintf("All %d attempts failed", maxRetries),
			"details": errMsg,
		},
	}, nil
}

func (c *AuthenticatedClient) doHTTP(method, reqURL, body string, headers map[string]string, timeout time.Duration) (*APIResponse, error) {
	var bodyReader io.Reader
	if method != "GET" && body != "" {
		bodyReader = bytes.NewBufferString(body)
	}

	req, err := http.NewRequest(method, reqURL, bodyReader)
	if err != nil {
		return nil, err
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	client := &http.Client{Timeout: timeout}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request timed out or failed: %w", err)
	}
	defer resp.Body.Close()

	text, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	var data any
	if err := json.Unmarshal(text, &data); err != nil {
		if len(text) == 0 {
			data = nil
		} else {
			data = string(text)
		}
	}

	return &APIResponse{Status: resp.StatusCode, Data: data}, nil
}

func (c *AuthenticatedClient) buildURL(path string, query map[string]string) (string, error) {
	path = strings.TrimLeft(path, "/")
	fullPath := "/api/v1/" + path

	u, err := url.Parse(c.baseUrl + fullPath)
	if err != nil {
		return "", err
	}

	if len(query) > 0 {
		q := u.Query()
		for k, v := range query {
			q.Set(k, v)
		}
		u.RawQuery = q.Encode()
	}

	return u.String(), nil
}

func (c *AuthenticatedClient) applyAuth(headers map[string]string, method, reqURL, body string) {
	if c.credentials.Mode == AuthModeSession {
		headers[c.credentials.Session.HeaderName] = c.credentials.Session.Token
	} else {
		timestamp := time.Now().UTC().Format(time.RFC3339)
		authHeaders := SignRequest(c.credentials.AKSK, method, reqURL, body, timestamp)
		for k, v := range authHeaders {
			headers[k] = v
		}
	}
}

func (c *AuthenticatedClient) checkCircuit() error {
	c.circuit.mu.Lock()
	defer c.circuit.mu.Unlock()

	if !c.circuit.isOpen {
		return nil
	}

	elapsed := time.Since(c.circuit.lastFailure)
	if elapsed > circuitResetMs*time.Millisecond {
		c.circuit.failures = 0
		c.circuit.isOpen = false
		c.circuit.lastFailure = time.Time{}
		return nil
	}

	remaining := (circuitResetMs*time.Millisecond - elapsed).Seconds()
	return fmt.Errorf("circuit breaker is OPEN — too many failures. Retry after %.0fs", math.Ceil(remaining))
}

func (c *AuthenticatedClient) recordFailure() {
	c.circuit.mu.Lock()
	defer c.circuit.mu.Unlock()

	c.circuit.failures++
	c.circuit.lastFailure = time.Now()
	if c.circuit.failures >= circuitThreshold {
		c.circuit.isOpen = true
	}
}

func (c *AuthenticatedClient) recordSuccess() {
	c.circuit.mu.Lock()
	defer c.circuit.mu.Unlock()

	c.circuit.failures = 0
	c.circuit.isOpen = false
}
