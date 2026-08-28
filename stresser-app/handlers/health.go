package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/marcoarc01/Analisando-LOGS-utilizando-LLMs-para-Enriquecimento-de-Contexto/stresser-app/metrics"
)

func HealthHandler(w http.ResponseWriter, r *http.Request) {
	metrics.HTTPRequestsTotal.WithLabelValues("GET", "/health", "200").Inc()

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{
		"status": "ok",
	})
}
