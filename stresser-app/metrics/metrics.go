package metrics

import (
	"net/http"
	"time"

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
		start := time.Now()

		next.ServeHTTP(w, r)

		duration := time.Since(start).Seconds()
		HTTPRequestDuration.WithLabelValues(r.URL.Path, r.Method).Observe(duration)
	})
}
