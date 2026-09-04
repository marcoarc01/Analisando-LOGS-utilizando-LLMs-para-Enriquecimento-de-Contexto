package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

type stubEngine struct {
	level     int
	workers   int
	setLevels []int
}

func (e *stubEngine) GetLevel() int {
	return e.level
}

func (e *stubEngine) GetWorkers() int {
	return e.workers
}

func (e *stubEngine) SetLevel(level int) {
	e.level = level
	e.setLevels = append(e.setLevels, level)
	if level == 0 {
		e.workers = 0
	}
}

func TestStateHandler(t *testing.T) {
	tests := []struct {
		name   string
		engine *stubEngine
		want   StateResponse
	}{
		{
			name:   "initial state",
			engine: &stubEngine{},
			want:   StateResponse{},
		},
		{
			name:   "current state",
			engine: &stubEngine{level: 42, workers: 3},
			want:   StateResponse{StressLevel: 42, CPUWorkers: 3},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodGet, "/api/state", nil)
			response := httptest.NewRecorder()

			StateHandler(test.engine)(response, request)

			if response.Code != http.StatusOK {
				t.Fatalf("status = %d, want %d", response.Code, http.StatusOK)
			}
			if contentType := response.Header().Get("Content-Type"); contentType != "application/json" {
				t.Fatalf("Content-Type = %q, want %q", contentType, "application/json")
			}

			var fields map[string]json.RawMessage
			if err := json.Unmarshal(response.Body.Bytes(), &fields); err != nil {
				t.Fatalf("invalid JSON response: %v", err)
			}
			if len(fields) != 2 || fields["stress_level"] == nil || fields["cpu_workers"] == nil {
				t.Fatalf("response fields = %#v", fields)
			}

			var got StateResponse
			if err := json.Unmarshal(response.Body.Bytes(), &got); err != nil {
				t.Fatalf("invalid state response: %v", err)
			}
			if got != test.want {
				t.Fatalf("response = %#v, want %#v", got, test.want)
			}
		})
	}
}
