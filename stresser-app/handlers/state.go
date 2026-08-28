package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/marcoarc01/Analisando-LOGS-utilizando-LLMs-para-Enriquecimento-de-Contexto/stresser-app/stress"
	"go.opentelemetry.io/otel/attribute"
)

type StateResponse struct {
	StressLevel int `json:"stress_level"`
	CPUWorkers  int `json:"cpu_workers"`
}

func StateHandler(engine *stress.Engine) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()
		_, span := tracer.Start(ctx, "GET /api/state")
		defer span.End()

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
	}
}
