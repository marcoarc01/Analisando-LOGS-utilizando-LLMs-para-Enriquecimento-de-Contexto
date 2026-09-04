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
			name:         "GET falls back to UI",
			method:       http.MethodGet,
			wantStatus:   http.StatusOK,
			bodyContains: "Stresser Observability",
		},
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
		})
	}
}
