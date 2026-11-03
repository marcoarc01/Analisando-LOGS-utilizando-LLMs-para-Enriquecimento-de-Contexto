package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/marcoarc01/Analisando-LOGS-utilizando-LLMs-para-Enriquecimento-de-Contexto/stresser-app/telemetry"
)

// About 1 s of hashing on the test machine: far beyond the budgets used below.
const longWork = 20_000_000

func serveWithBudget(t *testing.T, ctx context.Context, work int, budget string) ([]map[string]any, *httptest.ResponseRecorder, time.Duration) {
	t.Helper()
	var output bytes.Buffer
	request := httptest.NewRequest(http.MethodGet, "/api/state", nil).WithContext(ctx)
	request.Header.Set("X-Request-ID", "request-1")
	request.Header.Set("X-Experiment-Run-ID", "run-a")
	if budget != "" {
		request.Header.Set(DeadlineHeader, budget)
	}
	response := httptest.NewRecorder()
	started := time.Now()
	stateHandlerWithWork(&stubEngine{level: 100, workers: 1}, telemetry.NewLogger(&output), work)(response, request)
	elapsed := time.Since(started)
	var records []map[string]any
	for _, line := range strings.Split(strings.TrimSpace(output.String()), "\n") {
		var record map[string]any
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			t.Fatalf("invalid log %q: %v", line, err)
		}
		records = append(records, record)
	}
	return records, response, elapsed
}

func sortedKeys(record map[string]any) string {
	keys := make([]string, 0, len(record))
	for key := range record {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return strings.Join(keys, ",")
}

func TestDeadlineExceededStopsTheWorkAndAnswers504(t *testing.T) {
	records, response, elapsed := serveWithBudget(t, context.Background(), longWork, "20")
	if len(records) != 1 {
		t.Fatalf("%d log lines, want 1", len(records))
	}
	record := records[0]
	for key, value := range map[string]string{
		"event": "state_request_deadline_exceeded", "level": "warn", "message": "State request deadline exceeded",
		"service": "stresser-app", "method": "GET", "route": "/api/state", "error": "context deadline exceeded",
		"request_id": "request-1", "experiment_run_id": "run-a",
	} {
		if record[key] != value {
			t.Fatalf("%s = %v, want %q", key, record[key], value)
		}
	}
	// Same fields as state_request_canceled, nothing else (no level, budget or latency).
	if got := sortedKeys(record); got != "error,event,experiment_run_id,level,message,method,request_id,route,service,timestamp" {
		t.Fatalf("unexpected fields %s", got)
	}
	if response.Code != http.StatusGatewayTimeout || !strings.Contains(response.Body.String(), "deadline exceeded") {
		t.Fatalf("status %d body %q, want 504", response.Code, response.Body.String())
	}
	if elapsed > 500*time.Millisecond {
		t.Fatalf("work was not interrupted: %v for a 20 ms budget", elapsed)
	}
}

func TestDeadlineWithoutWorkStillAnswersInTime(t *testing.T) {
	records, response, _ := serveWithBudget(t, context.Background(), 1000, "60000")
	if len(records) != 1 || records[0]["event"] != "state_queried" || response.Code != http.StatusOK {
		t.Fatalf("records = %v, status = %d", records, response.Code)
	}
}

func TestWithoutBudgetNothingChanges(t *testing.T) {
	_, plain, _ := serveWithBudget(t, context.Background(), 1000, "")
	reference, _, _ := serveWithBudget(t, context.Background(), 1000, "")
	for _, value := range []string{"", "0", "-5", "abc", "1.5", "12345678", " 20"} {
		records, response, _ := serveWithBudget(t, context.Background(), 1000, value)
		if len(records) != 1 || records[0]["event"] != "state_queried" || response.Code != http.StatusOK ||
			response.Body.String() != plain.Body.String() || sortedKeys(records[0]) != sortedKeys(reference[0]) {
			t.Fatalf("budget %q changed the behaviour: %v %d %q", value, records, response.Code, response.Body.String())
		}
	}
	// Without a budget the work is not interrupted: it runs to the end, then the
	// canceled request is logged as before.
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	records, response, _ := serveWithBudget(t, canceled, 1000, "")
	if len(records) != 1 || records[0]["event"] != "state_request_canceled" || response.Body.Len() != 0 {
		t.Fatalf("records = %v, body = %q", records, response.Body.String())
	}
}

func TestCancellationKeepsStateRequestCanceledWithBudget(t *testing.T) {
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	records, response, elapsed := serveWithBudget(t, canceled, longWork, "60000")
	if len(records) != 1 || records[0]["event"] != "state_request_canceled" || records[0]["error"] != "context canceled" {
		t.Fatalf("records = %v", records)
	}
	if response.Body.Len() != 0 || elapsed > 500*time.Millisecond {
		t.Fatalf("body %q after %v, want no answer and an interrupted work", response.Body.String(), elapsed)
	}
}

func TestRequestBudget(t *testing.T) {
	for value, want := range map[string]time.Duration{"1": time.Millisecond, "292": 292 * time.Millisecond,
		"9999999": 9999999 * time.Millisecond} {
		request := httptest.NewRequest(http.MethodGet, "/api/state", nil)
		request.Header.Set(DeadlineHeader, value)
		if got, ok := requestBudget(request); !ok || got != want {
			t.Fatalf("requestBudget(%q) = %v %v, want %v", value, got, ok, want)
		}
	}
	for _, value := range []string{"", "0", "00", "-1", "+5", "1e3", "abc", "12345678"} {
		request := httptest.NewRequest(http.MethodGet, "/api/state", nil)
		request.Header.Set(DeadlineHeader, value)
		if _, ok := requestBudget(request); ok {
			t.Fatalf("requestBudget(%q) accepted", value)
		}
	}
}

func TestChainedSHA256CtxMatchesAndStops(t *testing.T) {
	for _, n := range []int{0, 1, workCheckBlock - 1, workCheckBlock, 3*workCheckBlock + 7} {
		sum, err := chainedSHA256Ctx(context.Background(), n)
		if err != nil || !reflect.DeepEqual(sum, chainedSHA256(n)) {
			t.Fatalf("chainedSHA256Ctx(%d) = %x %v, want %x", n, sum, err, chainedSHA256(n))
		}
	}
	done, cancel := context.WithCancel(context.Background())
	cancel()
	if sum, err := chainedSHA256Ctx(done, longWork); err != context.Canceled || sum != chainedSHA256(0) {
		t.Fatalf("expired context: %x %v, want zero work and context.Canceled", sum, err)
	}
}

func BenchmarkChainedSHA256(b *testing.B) {
	for i := 0; i < b.N; i++ {
		chainedSHA256(100_000)
	}
}

func BenchmarkChainedSHA256Ctx(b *testing.B) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Hour)
	defer cancel()
	for i := 0; i < b.N; i++ {
		chainedSHA256Ctx(ctx, 100_000)
	}
}
