package handlers

import (
	"bytes"
	"encoding/json"
	"github.com/marcoarc01/Analisando-LOGS-utilizando-LLMs-para-Enriquecimento-de-Contexto/stresser-app/telemetry"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

type stubEngine struct {
	level     int
	workers   int
	setLevels []int
}

func TestStateHandlerCorrelationLogs(t *testing.T) {
	for _, test := range []struct {
		name        string
		requestID   string
		runID       string
		wantRequest bool
		wantRun     bool
	}{
		{"valid", "request-123", "test-baseline-001", true, true},
		{"missing", "", "", false, false},
		{"invalid", "bad id", "bad\nrun", false, false},
		{"too long", strings.Repeat("a", 129), strings.Repeat("b", 129), false, false},
		{"boundary", strings.Repeat("a", 128), strings.Repeat("b", 128), true, true},
		{"request only", "request-123", "", true, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			var output bytes.Buffer
			logger := telemetry.NewLogger(&output)
			request := httptest.NewRequest(http.MethodGet, "/api/state?secret=private", nil)
			request.Header.Set("x-request-id", test.requestID)
			request.Header.Set("x-experiment-run-id", test.runID)
			request.Header.Set("Authorization", "private-token")
			request.Header.Set("Cookie", "private-cookie")
			response := httptest.NewRecorder()
			stateHandler(&stubEngine{}, logger)(response, request)
			if response.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200", response.Code)
			}
			var record map[string]any
			if err := json.Unmarshal(output.Bytes(), &record); err != nil {
				t.Fatalf("invalid structured log: %v", err)
			}
			if record["service"] != "stresser-app" || record["event"] != "state_queried" || record["route"] != "/api/state" {
				t.Fatalf("unexpected log fields: %#v", record)
			}
			for field, expected := range map[string]struct {
				present bool
				value   string
			}{
				"request_id":        {test.wantRequest, test.requestID},
				"experiment_run_id": {test.wantRun, test.runID},
			} {
				value, present := record[field]
				if present != expected.present || (present && value != expected.value) {
					t.Errorf("%s = %v (present=%v)", field, value, present)
				}
			}
			if strings.Contains(output.String(), "private") {
				t.Fatal("log contains unrelated private request data")
			}
		})
	}
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
