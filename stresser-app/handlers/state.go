package handlers

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"regexp"
	"time"

	"github.com/marcoarc01/Analisando-LOGS-utilizando-LLMs-para-Enriquecimento-de-Contexto/stresser-app/telemetry"

	"go.opentelemetry.io/otel/attribute"
)

type StressEngine interface {
	GetLevel() int
	GetWorkers() int
	SetLevel(int)
}

type StateResponse struct {
	StressLevel int `json:"stress_level"`
	CPUWorkers  int `json:"cpu_workers"`
}

var correlationIDPattern = regexp.MustCompile(`^[A-Za-z0-9._:-]{1,128}$`)

func StateHandler(engine StressEngine) http.HandlerFunc {
	return StateHandlerWithWork(engine, 0)
}

// StateHandlerWithWork runs workIterations of fixed CPU work before answering
// (0 = none, the default behaviour).
func StateHandlerWithWork(engine StressEngine, workIterations int) http.HandlerFunc {
	return stateHandlerWithWork(engine, telemetry.Logger, workIterations)
}

func stateHandler(engine StressEngine, logger *slog.Logger) http.HandlerFunc {
	return stateHandlerWithWork(engine, logger, 0)
}

func stateHandlerWithWork(engine StressEngine, logger *slog.Logger, workIterations int) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		started := time.Now()
		ctx := r.Context()
		_, span := tracer.Start(ctx, "GET /api/state")
		defer span.End()

		if workIterations > 0 {
			sum := chainedSHA256(workIterations)
			workSink.Store(uint32(sum[0]))
		}

		level := engine.GetLevel()
		workers := engine.GetWorkers()

		span.SetAttributes(
			attribute.Int("stress.level", level),
			attribute.Int("stress.workers", workers),
		)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(StateResponse{
			StressLevel: level,
			CPUWorkers:  workers,
		})
		fields := []any{
			"event", "state_queried",
			"method", r.Method,
			"route", "/api/state",
			"status_code", http.StatusOK,
			"latency_ms", float64(time.Since(started).Microseconds()) / 1000,
			"stress_level", level,
			"cpu_workers", workers,
		}
		for header, field := range map[string]string{
			"X-Request-ID":        "request_id",
			"X-Experiment-Run-ID": "experiment_run_id",
		} {
			if value := r.Header.Get(header); correlationIDPattern.MatchString(value) {
				fields = append(fields, field, value)
			}
		}
		logger.InfoContext(ctx, "state queried", fields...)
	}
}
