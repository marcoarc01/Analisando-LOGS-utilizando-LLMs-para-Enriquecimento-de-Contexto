package main

import (
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/marcoarc01/Analisando-LOGS-utilizando-LLMs-para-Enriquecimento-de-Contexto/stresser-app/handlers"
	"github.com/marcoarc01/Analisando-LOGS-utilizando-LLMs-para-Enriquecimento-de-Contexto/stresser-app/metrics"
	"github.com/marcoarc01/Analisando-LOGS-utilizando-LLMs-para-Enriquecimento-de-Contexto/stresser-app/stress"
	"github.com/marcoarc01/Analisando-LOGS-utilizando-LLMs-para-Enriquecimento-de-Contexto/stresser-app/tracing"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
)

func main() {
	shutdown := tracing.InitTracer()
	defer shutdown()

	engine := stress.NewEngine()
	mux := newMux(engine)

	port := os.Getenv("STRESSER_PORT")
	if port == "" {
		port = "8080"
	}

	addr := fmt.Sprintf(":%s", port)
	log.Printf("Stresser App rodando em http://localhost%s", addr)
	log.Printf("Metricas em http://localhost%s/metrics", addr)

	handler := otelhttp.NewHandler(metrics.MetricsMiddleware(mux), "stresser-app")

	if err := http.ListenAndServe(addr, handler); err != nil {
		log.Fatalf("Erro ao iniciar servidor: %v", err)
	}
}

func newMux(engine handlers.StressEngine) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", handlers.HealthHandler)
	mux.HandleFunc("GET /api/state", handlers.StateHandler(engine))
	mux.HandleFunc("POST /api/stress", handlers.StressHandler(engine))
	mux.Handle("GET /metrics", promhttp.Handler())
	mux.Handle("GET /static/", handlers.StaticHandler())
	mux.HandleFunc("GET /", handlers.UIHandler)
	return mux
}
