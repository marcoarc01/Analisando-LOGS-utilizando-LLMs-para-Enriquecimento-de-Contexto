package stress

import (
	"context"
	"runtime"
	"sync"

	"github.com/marcoarc01/Analisando-LOGS-utilizando-LLMs-para-Enriquecimento-de-Contexto/stresser-app/metrics"
	"github.com/marcoarc01/Analisando-LOGS-utilizando-LLMs-para-Enriquecimento-de-Contexto/stresser-app/telemetry"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
)

var tracer = otel.Tracer("stresser-app")

type Engine struct {
	mu         sync.Mutex
	level      int
	workers    int
	cancel     []chan struct{}
	maxWorkers int
	worker     func(<-chan struct{})
}

func NewEngine() *Engine {
	return newEngine(runtime.NumCPU(), cpuWorker)
}

func newEngine(maxWorkers int, worker func(<-chan struct{})) *Engine {
	return &Engine{
		level:      0,
		workers:    0,
		cancel:     make([]chan struct{}, 0),
		maxWorkers: maxWorkers,
		worker:     worker,
	}
}

func ValidLevel(level int) bool {
	return level >= 0 && level <= 100
}

func (e *Engine) GetLevel() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.level
}

func (e *Engine) GetWorkers() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.workers
}

func (e *Engine) SetLevel(level int) {
	if !ValidLevel(level) {
		return
	}

	ctx := context.Background()
	_, span := tracer.Start(ctx, "engine.SetLevel")
	defer span.End()

	e.mu.Lock()
	defer e.mu.Unlock()

	oldLevel := e.level
	e.level = level

	span.SetAttributes(
		attribute.Int("stress.level.old", oldLevel),
		attribute.Int("stress.level.new", level),
	)

	for _, ch := range e.cancel {
		close(ch)
	}
	e.cancel = make([]chan struct{}, 0)

	desiredWorkers := (level * e.maxWorkers) / 100

	for i := 0; i < desiredWorkers; i++ {
		ch := make(chan struct{})
		e.cancel = append(e.cancel, ch)
		go e.worker(ch)
	}

	e.workers = desiredWorkers

	span.SetAttributes(
		attribute.Int("stress.workers.count", desiredWorkers),
		attribute.Int("stress.cpu.max", e.maxWorkers),
	)

	metrics.StressLevel.Set(float64(level))
	metrics.StressCPUWorkers.Set(float64(desiredWorkers))
	metrics.StressChangesTotal.Inc()

	telemetry.Logger.Info("Stress setting updated", "event", "stress_updated",
		"stress_level", level, "cpu_workers", desiredWorkers, "old_stress_level", oldLevel)
}

func cpuWorker(stop <-chan struct{}) {
	for {
		select {
		case <-stop:
			return
		default:
			_ = 1 + 1
		}
	}
}
