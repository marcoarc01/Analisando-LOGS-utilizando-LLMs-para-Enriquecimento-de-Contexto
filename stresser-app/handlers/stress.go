package handlers

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/marcoarc01/Analisando-LOGS-utilizando-LLMs-para-Enriquecimento-de-Contexto/stresser-app/metrics"
	"github.com/marcoarc01/Analisando-LOGS-utilizando-LLMs-para-Enriquecimento-de-Contexto/stresser-app/stress"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
)

var tracer = otel.Tracer("stresser-app")

type StressRequest struct {
	Level int `json:"level"`
}

type StressResponse struct {
	Message     string `json:"message"`
	StressLevel int    `json:"stress_level"`
	CPUWorkers  int    `json:"cpu_workers"`
}

func StressHandler(engine *stress.Engine) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()
		ctx, span := tracer.Start(ctx, "POST /api/stress")
		defer span.End()

		var req StressRequest

		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			span.RecordError(err)
			span.SetStatus(codes.Error, "JSON inválido")
			span.SetAttributes(attribute.String("error.type", "invalid_json"))

			metrics.HTTPRequestsTotal.WithLabelValues("POST", "/api/stress", "400").Inc()
			http.Error(w, `{"error": "JSON inválido"}`, http.StatusBadRequest)
			return
		}

		span.SetAttributes(attribute.Int("stress.level.requested", req.Level))

		if req.Level < 0 || req.Level > 100 {
			span.SetStatus(codes.Error, "level fora do range")
			span.SetAttributes(attribute.String("error.type", "invalid_range"))

			metrics.HTTPRequestsTotal.WithLabelValues("POST", "/api/stress", "400").Inc()
			http.Error(w, `{"error": "level deve ser entre 0 e 100"}`, http.StatusBadRequest)
			return
		}

		engine.SetLevel(req.Level)

		workers := engine.GetWorkers()
		span.SetAttributes(
			attribute.Int("stress.level.applied", req.Level),
			attribute.Int("stress.workers", workers),
		)
		span.SetStatus(codes.Ok, "Stress level atualizado")

		metrics.HTTPRequestsTotal.WithLabelValues("POST", "/api/stress", "200").Inc()

		log.Printf("Stress alterado para %d%% (%d workers)", req.Level, workers)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(StressResponse{
			Message:     "Stress level atualizado",
			StressLevel: engine.GetLevel(),
			CPUWorkers:  workers,
		})
	}
}
