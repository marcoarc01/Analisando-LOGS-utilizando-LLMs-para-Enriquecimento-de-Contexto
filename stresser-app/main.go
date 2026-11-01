package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"runtime"
	"strconv"

	"github.com/marcoarc01/Analisando-LOGS-utilizando-LLMs-para-Enriquecimento-de-Contexto/stresser-app/handlers"
	"github.com/marcoarc01/Analisando-LOGS-utilizando-LLMs-para-Enriquecimento-de-Contexto/stresser-app/metrics"
	"github.com/marcoarc01/Analisando-LOGS-utilizando-LLMs-para-Enriquecimento-de-Contexto/stresser-app/stress"
	"github.com/marcoarc01/Analisando-LOGS-utilizando-LLMs-para-Enriquecimento-de-Contexto/stresser-app/telemetry"
	"github.com/marcoarc01/Analisando-LOGS-utilizando-LLMs-para-Enriquecimento-de-Contexto/stresser-app/tracing"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
)

func main() {
	log.SetFlags(0)
	log.SetOutput(telemetry.ErrorWriter{})
	shutdown := tracing.InitTracer()
	defer shutdown()

	mode := os.Getenv("STRESS_MODE")
	engine, err := stress.NewEngineForMode(mode)
	if err != nil {
		telemetry.Logger.Error("Invalid stress mode", "event", "config_invalid")
		os.Exit(1)
	}
	work, err := workIterations(os.Getenv("STATE_WORK_ITERATIONS"))
	if err != nil {
		telemetry.Logger.Error("Invalid state work iterations", "event", "config_invalid")
		os.Exit(1)
	}
	mux := newMuxWithWork(engine, work)

	port := os.Getenv("STRESSER_PORT")
	if port == "" {
		port = "8080"
	}

	addr := fmt.Sprintf(":%s", port)
	telemetry.Logger.Info("Service started", startupFields(engine.Mode())...)

	handler := otelhttp.NewHandler(metrics.MetricsMiddleware(mux), "stresser-app")

	if err := http.ListenAndServe(addr, handler); err != nil {
		telemetry.Logger.Error("Server startup failed", "event", "server_start_failed")
		os.Exit(1)
	}
}

func newMux(engine handlers.StressEngine) *http.ServeMux {
	return newMuxWithWork(engine, 0)
}

func newMuxWithWork(engine handlers.StressEngine, stateWork int) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", handlers.HealthHandler)
	mux.HandleFunc("GET /api/state", handlers.StateHandlerWithWork(engine, stateWork))
	mux.HandleFunc("POST /api/stress", handlers.StressHandler(engine))
	mux.Handle("GET /metrics", promhttp.Handler())
	mux.Handle("GET /static/", handlers.StaticHandler())
	mux.HandleFunc("GET /{$}", handlers.UIHandler)
	return mux
}

// The default (workers) start event is unchanged; duty mode records only the
// mode and GOMAXPROCS, never the stress level, workers or busy fraction.
func startupFields(mode string) []any {
	fields := []any{"event", "service_started"}
	if mode == stress.ModeDuty {
		fields = append(fields, "stress_mode", mode, "gomaxprocs", runtime.GOMAXPROCS(0))
	}
	return fields
}

// workIterations parses STATE_WORK_ITERATIONS: empty means 0 (no work).
func workIterations(value string) (int, error) {
	if value == "" {
		return 0, nil
	}
	iterations, err := strconv.Atoi(value)
	if err != nil || iterations < 0 {
		return 0, fmt.Errorf("invalid work iterations %q", value)
	}
	return iterations, nil
}
