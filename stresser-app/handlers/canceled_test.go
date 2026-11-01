package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"testing"
	"time"
	"unicode"

	"github.com/marcoarc01/Analisando-LOGS-utilizando-LLMs-para-Enriquecimento-de-Contexto/stresser-app/telemetry"
)

func serveState(t *testing.T, ctx context.Context, work int) ([]map[string]any, *httptest.ResponseRecorder) {
	t.Helper()
	var output bytes.Buffer
	request := httptest.NewRequest(http.MethodGet, "/api/state", nil).WithContext(ctx)
	request.Header.Set("X-Request-ID", "request-1")
	request.Header.Set("X-Experiment-Run-ID", "run-a")
	response := httptest.NewRecorder()
	stateHandlerWithWork(&stubEngine{level: 92, workers: 1}, telemetry.NewLogger(&output), work)(response, request)
	var records []map[string]any
	for _, line := range strings.Split(strings.TrimSpace(output.String()), "\n") {
		var record map[string]any
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			t.Fatalf("invalid log %q: %v", line, err)
		}
		records = append(records, record)
	}
	return records, response
}

func TestStateRequestCanceledAfterWork(t *testing.T) {
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	expired, cancelExpired := context.WithTimeout(context.Background(), -time.Second)
	defer cancelExpired()
	for want, ctx := range map[string]context.Context{"context canceled": canceled, "context deadline exceeded": expired} {
		for _, work := range []int{0, 1000} {
			records, response := serveState(t, ctx, work)
			if len(records) != 1 {
				t.Fatalf("%s: %d log lines, want 1", want, len(records))
			}
			record := records[0]
			for key, value := range map[string]string{
				"event": "state_request_canceled", "level": "warn", "message": "State request canceled",
				"service": "stresser-app", "method": "GET", "route": "/api/state", "error": want,
				"request_id": "request-1", "experiment_run_id": "run-a",
			} {
				if record[key] != value {
					t.Fatalf("%s: %s = %v, want %q", want, key, record[key], value)
				}
			}
			keys := make([]string, 0, len(record))
			for key := range record {
				keys = append(keys, key)
			}
			sort.Strings(keys)
			if strings.Join(keys, ",") != "error,event,experiment_run_id,level,message,method,request_id,route,service,timestamp" {
				t.Fatalf("%s: unexpected fields %v", want, keys)
			}
			for key, value := range record {
				if key == "timestamp" || key == "request_id" || key == "experiment_run_id" {
					continue
				}
				if strings.IndexFunc(value.(string), unicode.IsDigit) >= 0 {
					t.Fatalf("%s: number in %s=%v", want, key, value)
				}
			}
			if response.Body.Len() != 0 {
				t.Fatalf("%s: answered a canceled request: %q", want, response.Body.String())
			}
		}
	}
}

func TestStateQueriedWhenTheClientWaits(t *testing.T) {
	records, response := serveState(t, context.Background(), 1000)
	if len(records) != 1 || records[0]["event"] != "state_queried" || response.Code != http.StatusOK {
		t.Fatalf("records = %v, status = %d", records, response.Code)
	}
}

func TestCanceledReason(t *testing.T) {
	for err, want := range map[error]string{nil: "", context.Canceled: "context canceled",
		context.DeadlineExceeded: "context deadline exceeded"} {
		if got := canceledReason(err); got != want {
			t.Fatalf("canceledReason(%v) = %q, want %q", err, got, want)
		}
	}
}
