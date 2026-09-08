package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

type routerEngine struct {
	setCalls int
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
