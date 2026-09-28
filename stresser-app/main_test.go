package main

import (
	"bytes"
	"encoding/json"
	"github.com/marcoarc01/Analisando-LOGS-utilizando-LLMs-para-Enriquecimento-de-Contexto/stresser-app/metrics"
	"github.com/marcoarc01/Analisando-LOGS-utilizando-LLMs-para-Enriquecimento-de-Contexto/stresser-app/stress"
	"github.com/marcoarc01/Analisando-LOGS-utilizando-LLMs-para-Enriquecimento-de-Contexto/stresser-app/telemetry"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
)

type routerEngine struct {
	setCalls int
}

func TestHTTPInstrumentationAndSingleStressEvent(t *testing.T) {
	var output bytes.Buffer
	previous := telemetry.Logger
	telemetry.Logger = telemetry.NewLogger(&output)
	defer func() { telemetry.Logger = previous }()
	router := metrics.MetricsMiddleware(newMux(stress.NewEngine()))
	for _, tc := range []struct {
		method, path, body string
		code               int
	}{
		{"GET", "/api/state", "", 200},
		{"POST", "/api/state", "", 405},
		{"POST", "/api/stress", `{"level":null}`, 400},
		{"POST", "/api/stress", `{"level":0}`, 200},
		{"GET", "/customer/private-user", "", 404},
	} {
		metric := metrics.HTTPRequestsTotal.WithLabelValues(tc.method, metrics.NormalizedPath(tc.path), strconv.Itoa(tc.code))
		before := testutil.ToFloat64(metric)
		response := httptest.NewRecorder()
		router.ServeHTTP(response, httptest.NewRequest(tc.method, tc.path, strings.NewReader(tc.body)))
		if response.Code != tc.code || testutil.ToFloat64(metric)-before != 1 {
			t.Fatal("incorrect request count/status", tc)
		}
	}
	updates := 0
	for _, line := range strings.Split(strings.TrimSpace(output.String()), "\n") {
		var record map[string]any
		if json.Unmarshal([]byte(line), &record) != nil {
			t.Fatal("non-JSON log", line)
		}
		if record["event"] == "stress_updated" {
			updates++
			for _, key := range []string{"timestamp", "level", "service", "event", "message", "stress_level", "cpu_workers", "old_stress_level"} {
				if _, ok := record[key]; !ok {
					t.Fatal("missing field", key)
				}
			}
			for _, key := range []string{"method", "route", "status_code", "degradation_type"} {
				if _, ok := record[key]; ok {
					t.Fatal("inapplicable field", key)
				}
			}
		}
	}
	if updates != 1 {
		t.Fatalf("stress_updated count=%d", updates)
	}
	response := httptest.NewRecorder()
	router.ServeHTTP(response, httptest.NewRequest("GET", "/metrics", nil))
	text := response.Body.String()
	for _, sensitive := range []string{"private-user", "request_id=", "experiment_run_id="} {
		if strings.Contains(text, sensitive) {
			t.Fatal("unbounded metric label", sensitive)
		}
	}
	for _, name := range []string{"stress_level", "stress_cpu_workers", "stress_changes_total", "http_requests_total", "http_request_duration_seconds_count"} {
		if !strings.Contains(text, name) {
			t.Fatal("missing metric", name)
		}
	}
}

func (routerEngine) GetLevel() int {
	return 0
}

func (routerEngine) GetWorkers() int {
	return 0
}

func (e *routerEngine) SetLevel(int) {
	e.setCalls++
}

func TestRouterHandlesIncorrectStressMethods(t *testing.T) {
	tests := []struct {
		name         string
		method       string
		wantStatus   int
		bodyContains string
	}{
		{
			name:         "GET is rejected",
			method:       http.MethodGet,
			wantStatus:   http.StatusMethodNotAllowed,
			bodyContains: "Method Not Allowed",
		},
		{name: "HEAD is rejected", method: http.MethodHead, wantStatus: http.StatusMethodNotAllowed},
		{name: "PATCH is rejected", method: http.MethodPatch, wantStatus: http.StatusMethodNotAllowed},
		{name: "DELETE is rejected", method: http.MethodDelete, wantStatus: http.StatusMethodNotAllowed},
		{name: "OPTIONS is rejected", method: http.MethodOptions, wantStatus: http.StatusMethodNotAllowed},
		{
			name:         "PUT is rejected",
			method:       http.MethodPut,
			wantStatus:   http.StatusMethodNotAllowed,
			bodyContains: "Method Not Allowed",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			engine := &routerEngine{}
			request := httptest.NewRequest(test.method, "/api/stress", nil)
			response := httptest.NewRecorder()

			newMux(engine).ServeHTTP(response, request)

			if response.Code != test.wantStatus {
				t.Fatalf("status = %d, want %d", response.Code, test.wantStatus)
			}
			if !strings.Contains(response.Body.String(), test.bodyContains) {
				t.Fatalf("body does not contain %q", test.bodyContains)
			}
			if engine.setCalls != 0 {
				t.Fatalf("SetLevel calls = %d, want 0", engine.setCalls)
			}
			if allow := response.Header().Get("Allow"); allow != "POST" {
				t.Fatalf("Allow = %q, want POST", allow)
			}
		})
	}
}

func TestRouterKnownRoutes(t *testing.T) {
	tests := []struct {
		method       string
		path         string
		body         string
		contentType  string
		bodyContains string
		setCalls     int
	}{
		{http.MethodGet, "/", "", "text/html", "Stresser Observability", 0},
		{http.MethodGet, "/health", "", "application/json", `"status":"ok"`, 0},
		{http.MethodGet, "/api/state", "", "application/json", `"stress_level":0`, 0},
		{http.MethodPost, "/api/stress", `{"level":0}`, "application/json", `"stress_level":0`, 1},
		{http.MethodGet, "/static/style.css", "", "text/css", "{", 0},
		{http.MethodGet, "/metrics", "", "text/plain", "stress_level", 0},
	}
	for _, test := range tests {
		t.Run(test.method+" "+test.path, func(t *testing.T) {
			engine := &routerEngine{}
			response := httptest.NewRecorder()
			newMux(engine).ServeHTTP(response, httptest.NewRequest(test.method, test.path, strings.NewReader(test.body)))
			if response.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200", response.Code)
			}
			if !strings.HasPrefix(response.Header().Get("Content-Type"), test.contentType) || !strings.Contains(response.Body.String(), test.bodyContains) {
				t.Fatalf("unexpected response: Content-Type=%q body=%q", response.Header().Get("Content-Type"), response.Body.String())
			}
			if engine.setCalls != test.setCalls {
				t.Fatalf("SetLevel calls = %d, want %d", engine.setCalls, test.setCalls)
			}
		})
	}
}

func TestRouterUnknownRoutes(t *testing.T) {
	for _, path := range []string{"/missing", "/api/missing", "/api/stress/", "/health/missing"} {
		for _, method := range []string{http.MethodGet, http.MethodPost} {
			t.Run(method+" "+path, func(t *testing.T) {
				response := httptest.NewRecorder()
				newMux(&routerEngine{}).ServeHTTP(response, httptest.NewRequest(method, path, nil))
				if response.Code != http.StatusNotFound {
					t.Fatalf("status = %d, want 404", response.Code)
				}
			})
		}
	}
}
