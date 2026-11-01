package stress

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"time"

	"github.com/marcoarc01/Analisando-LOGS-utilizando-LLMs-para-Enriquecimento-de-Contexto/stresser-app/metrics"
	"github.com/marcoarc01/Analisando-LOGS-utilizando-LLMs-para-Enriquecimento-de-Contexto/stresser-app/telemetry"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
)

var tracer = otel.Tracer("stresser-app")

const (
	ModeWorkers = "workers"
	ModeDuty    = "duty"
	DutyPeriod  = 100 * time.Millisecond
)

type Engine struct {
	mu         sync.Mutex
	level      int
	workers    int
	cancel     []chan struct{}
	maxWorkers int
	mode       string
	worker     func(<-chan struct{})
	dutyWorker func(<-chan struct{}, time.Duration, time.Duration)
}

func NewEngine() *Engine {
	return newEngine(runtime.NumCPU(), cpuWorker)
}

// NewEngineForMode returns the default workers engine for "" or "workers".
// "duty" runs one goroutine per P (runtime.GOMAXPROCS), each busy for level%
// of every DutyPeriod, so the stress fills the CPU the service may use.
func NewEngineForMode(mode string) (*Engine, error) {
	switch mode {
	case "", ModeWorkers:
		return NewEngine(), nil
	case ModeDuty:
		return newDutyEngine(runtime.GOMAXPROCS(0), dutyWorker), nil
	}
	return nil, fmt.Errorf("unknown stress mode %q", mode)
}

func newDutyEngine(maxWorkers int, worker func(<-chan struct{}, time.Duration, time.Duration)) *Engine {
	engine := newEngine(maxWorkers, nil)
	engine.mode = ModeDuty
	engine.dutyWorker = worker
	return engine
}

func newEngine(maxWorkers int, worker func(<-chan struct{})) *Engine {
	return &Engine{
		level:      0,
		workers:    0,
		cancel:     make([]chan struct{}, 0),
		maxWorkers: maxWorkers,
		mode:       ModeWorkers,
		worker:     worker,
	}
}

func (e *Engine) Mode() string {
	return e.mode
}

func DesiredWorkers(mode string, level, maxWorkers int) int {
	if mode == ModeDuty {
		if level > 0 {
			return maxWorkers
		}
		return 0
	}
	return (level * maxWorkers) / 100
}

func DutyCycle(level int, period time.Duration) (busy, idle time.Duration) {
	busy = period * time.Duration(level) / 100
	return busy, period - busy
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

	desiredWorkers := DesiredWorkers(e.mode, level, e.maxWorkers)
	busy, idle := DutyCycle(level, DutyPeriod)

	for i := 0; i < desiredWorkers; i++ {
		ch := make(chan struct{})
		e.cancel = append(e.cancel, ch)
		if e.mode == ModeDuty {
			go e.dutyWorker(ch, busy, idle)
		} else {
			go e.worker(ch)
		}
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

func dutyWorker(stop <-chan struct{}, busy, idle time.Duration) {
	timer := time.NewTimer(0)
	defer timer.Stop()
	for {
		deadline := time.Now().Add(busy)
		for time.Now().Before(deadline) {
			select {
			case <-stop:
				return
			default:
			}
		}
		if idle <= 0 {
			select {
			case <-stop:
				return
			default:
			}
			continue
		}
		if !timer.Stop() {
			select {
			case <-timer.C:
			default:
			}
		}
		timer.Reset(idle)
		select {
		case <-stop:
			return
		case <-timer.C:
		}
	}
}
