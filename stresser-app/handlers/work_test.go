package handlers

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sort"
	"testing"

	"github.com/marcoarc01/Analisando-LOGS-utilizando-LLMs-para-Enriquecimento-de-Contexto/stresser-app/telemetry"
)

func TestChainedSHA256(t *testing.T) {
	var zero [sha256.Size]byte
	if got := chainedSHA256(0); got != zero {
		t.Fatalf("chainedSHA256(0) = %x, want zeros", got)
	}
	want := zero
	for i := 1; i <= 3; i++ {
		want = sha256.Sum256(want[:])
		if got := chainedSHA256(i); got != want {
			t.Fatalf("chainedSHA256(%d) = %x, want %x", i, got, want)
		}
	}
}

func TestStateWorkKeepsResponseAndLogFields(t *testing.T) {
	serve := func(iterations int) (string, []string) {
		var output bytes.Buffer
		request := httptest.NewRequest(http.MethodGet, "/api/state", nil)
		request.Header.Set("X-Request-ID", "request-1")
		response := httptest.NewRecorder()
		stateHandlerWithWork(&stubEngine{level: 75, workers: 2}, telemetry.NewLogger(&output), iterations)(response, request)
		var record map[string]any
		if err := json.Unmarshal(output.Bytes(), &record); err != nil {
			t.Fatalf("invalid log: %v", err)
		}
		keys := make([]string, 0, len(record))
		for key := range record {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		return response.Body.String(), keys
	}
	body, keys := serve(0)
	for _, iterations := range []int{1, 5000} {
		gotBody, gotKeys := serve(iterations)
		if gotBody != body || !reflect.DeepEqual(gotKeys, keys) {
			t.Fatalf("work=%d changed the response or log fields: %s %v", iterations, gotBody, gotKeys)
		}
	}
}
