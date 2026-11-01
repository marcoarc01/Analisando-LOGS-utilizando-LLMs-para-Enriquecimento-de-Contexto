package handlers

import (
	"context"
	"encoding/json"
	"errors"
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
		// The client may have given up (for instance on its own timeout) while the
		// work ran: record that instead of answering a request nobody waits for.
		if reason := canceledReason(ctx.Err()); reason != "" {
			fields := append([]any{
				"event", "state_request_canceled",
				"method", r.Method,
				"route", "/api/state",
				"error", reason,
			}, correlationFields(r)...)
			logger.WarnContext(ctx, "State request canceled", fields...)
			return
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
		fields = append(fields, correlationFields(r)...)
		logger.InfoContext(ctx, "state queried", fields...)
	}
}

func correlationFields(r *http.Request) []any {
	var fields []any
	for _, pair := range [][2]string{{"X-Request-ID", "request_id"}, {"X-Experiment-Run-ID", "experiment_run_id"}} {
		if value := r.Header.Get(pair[0]); correlationIDPattern.MatchString(value) {
			fields = append(fields, pair[1], value)
		}
	}
	return fields
}

// canceledReason maps a request context error to a fixed text, never its details.
func canceledReason(err error) string {
	switch {
	case errors.Is(err, context.DeadlineExceeded):
		return "context deadline exceeded"
	case errors.Is(err, context.Canceled):
		return "context canceled"
	}
	return ""
}
