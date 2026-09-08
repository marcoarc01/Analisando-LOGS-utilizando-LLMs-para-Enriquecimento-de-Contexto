package handlers

import (
	"encoding/json"
	"errors"
	"io"
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
	Level *int `json:"level"`
}

type StressResponse struct {
	Message     string `json:"message"`
	StressLevel int    `json:"stress_level"`
	CPUWorkers  int    `json:"cpu_workers"`
}

func StressHandler(engine StressEngine) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()
		ctx, span := tracer.Start(ctx, "POST /api/stress")
		defer span.End()

		var req StressRequest

		decoder := json.NewDecoder(r.Body)
		decoder.DisallowUnknownFields()
		err := decoder.Decode(&req)
		if err == nil {
			if trailingErr := decoder.Decode(new(any)); trailingErr != io.EOF {
				err = errors.New("conteúdo adicional depois do JSON")
			}
		}
		if err == nil && req.Level == nil {
			err = errors.New("level é obrigatório e deve ser inteiro")
		}
		if err != nil {
			span.RecordError(err)
			span.SetStatus(codes.Error, "JSON inválido")
			span.SetAttributes(attribute.String("error.type", "invalid_json"))

			metrics.HTTPRequestsTotal.WithLabelValues("POST", "/api/stress", "400").Inc()
			http.Error(w, `{"error": "JSON inválido"}`, http.StatusBadRequest)
			return
		}

		level := *req.Level
		span.SetAttributes(attribute.Int("stress.level.requested", level))

		if !stress.ValidLevel(level) {
			span.SetStatus(codes.Error, "level fora do range")
			span.SetAttributes(attribute.String("error.type", "invalid_range"))

			metrics.HTTPRequestsTotal.WithLabelValues("POST", "/api/stress", "400").Inc()
			http.Error(w, `{"error": "level deve ser entre 0 e 100"}`, http.StatusBadRequest)
			return
		}

		engine.SetLevel(level)

		workers := engine.GetWorkers()
		span.SetAttributes(
			attribute.Int("stress.level.applied", level),
			attribute.Int("stress.workers", workers),
		)
		span.SetStatus(codes.Ok, "Stress level atualizado")

		metrics.HTTPRequestsTotal.WithLabelValues("POST", "/api/stress", "200").Inc()

		log.Printf("Stress alterado para %d%% (%d workers)", level, workers)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(StressResponse{
			Message:     "Stress level atualizado",
			StressLevel: engine.GetLevel(),
			CPUWorkers:  workers,
		})
	}
}
