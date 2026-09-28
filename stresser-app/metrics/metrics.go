package metrics

import (
	"github.com/felixge/httpsnoop"
	"net/http"
	"strconv"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var StressLevel = promauto.NewGauge(prometheus.GaugeOpts{
	Name: "stress_level",
	Help: "Current CPU stress level percentage (0-100)",
})

var StressCPUWorkers = promauto.NewGauge(prometheus.GaugeOpts{
	Name: "stress_cpu_workers",
	Help: "Number of active CPU stress workers",
})

var HTTPRequestsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
	Name: "http_requests_total",
	Help: "Total HTTP requests received",
}, []string{"method", "path", "status"})

var StressChangesTotal = promauto.NewCounter(prometheus.CounterOpts{
	Name: "stress_changes_total",
	Help: "Total number of stress level changes",
})

var HTTPRequestDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
	Name: "http_request_duration_seconds",
	Help: "Duration of HTTP requests in seconds",
}, []string{"path", "method"})

func MetricsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		measurement := httpsnoop.CaptureMetrics(next, w, r)
		path, method := NormalizedPath(r.URL.Path), normalizedMethod(r.Method)
		HTTPRequestsTotal.WithLabelValues(method, path, strconv.Itoa(measurement.Code)).Inc()
		HTTPRequestDuration.WithLabelValues(path, method).Observe(measurement.Duration.Seconds())
	})
}

func NormalizedPath(path string) string {
	switch path {
	case "/", "/health", "/metrics", "/api/state", "/api/stress":
		return path
	default:
		return "__unmatched__"
	}
}

func normalizedMethod(method string) string {
	switch method {
	case "GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "TRACE", "CONNECT":
		return method
	default:
		return "OTHER"
	}
}
