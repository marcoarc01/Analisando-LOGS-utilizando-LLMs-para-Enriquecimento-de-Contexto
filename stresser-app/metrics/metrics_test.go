package metrics

import (
	"github.com/prometheus/client_golang/prometheus/testutil"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
)

func TestMetricsMiddleware(t *testing.T) {
	called := false
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.WriteHeader(http.StatusNoContent)
	})
	request := httptest.NewRequest(http.MethodGet, "/test", nil)
	response := httptest.NewRecorder()

	MetricsMiddleware(next).ServeHTTP(response, request)

	if !called {
		t.Fatal("next handler was not called")
	}
	if response.Code != http.StatusNoContent {
		t.Fatalf("status = %d, want %d", response.Code, http.StatusNoContent)
	}
}

func TestCountsStatusAndNormalizesLabels(t *testing.T) {
	for _, tc := range []struct {
		path, method string
		code         int
	}{
		{"/api/state", "GET", 200}, {"/api/stress", "POST", 400},
		{"/api/state", "POST", 405}, {"/private/customer-123", "GET", 404},
		{"/api/state", "GET", 500}, {"/arbitrary", "FREE", 404},
	} {
		path, method := NormalizedPath(tc.path), normalizedMethod(tc.method)
		counter := HTTPRequestsTotal.WithLabelValues(method, path, strconv.Itoa(tc.code))
		before := testutil.ToFloat64(counter)
		handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(tc.code) })
		response := httptest.NewRecorder()
		MetricsMiddleware(handler).ServeHTTP(response, httptest.NewRequest(tc.method, tc.path, nil))
		if got := testutil.ToFloat64(counter) - before; got != 1 {
			t.Fatalf("increment=%v", got)
		}
		if response.Code != tc.code {
			t.Fatalf("status=%d", response.Code)
		}
		if tc.path == "/private/customer-123" && path != "__unmatched__" {
			t.Fatal("unbounded path label")
		}
	}
}
