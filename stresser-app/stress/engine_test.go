package stress

import (
	"io"
	"log"
	"sync"
	"testing"
	"time"
)

func silenceLogs(t *testing.T) {
	writer := log.Writer()
	log.SetOutput(io.Discard)
	t.Cleanup(func() {
		log.SetOutput(writer)
	})
}

func TestEngineInitialState(t *testing.T) {
	engine := NewEngine()

	if level := engine.GetLevel(); level != 0 {
		t.Fatalf("level = %d, want 0", level)
	}
	if workers := engine.GetWorkers(); workers != 0 {
		t.Fatalf("workers = %d, want 0", workers)
	}
}

func TestEngineTransitions(t *testing.T) {
	silenceLogs(t)
	engine := newEngine(0, cpuWorker)

	for _, level := range []int{25, 75, 0} {
		engine.SetLevel(level)
		if got := engine.GetLevel(); got != level {
			t.Fatalf("level = %d, want %d", got, level)
		}
		if workers := engine.GetWorkers(); workers != 0 {
			t.Fatalf("workers = %d, want 0", workers)
		}
	}
}

func TestEngineValidatesLimits(t *testing.T) {
	silenceLogs(t)
	engine := newEngine(0, cpuWorker)
	engine.SetLevel(40)

	for _, level := range []int{-1, 101} {
		engine.SetLevel(level)
		if got := engine.GetLevel(); got != 40 {
			t.Fatalf("level after SetLevel(%d) = %d, want 40", level, got)
		}
	}

	for _, level := range []int{0, 100} {
		engine.SetLevel(level)
		if got := engine.GetLevel(); got != level {
			t.Fatalf("level after SetLevel(%d) = %d, want %d", level, got, level)
		}
	}

	engine.SetLevel(0)
}

func TestValidLevel(t *testing.T) {
	tests := []struct {
		level int
		want  bool
	}{
		{level: -1, want: false},
		{level: 0, want: true},
		{level: 100, want: true},
		{level: 101, want: false},
	}

	for _, test := range tests {
		if got := ValidLevel(test.level); got != test.want {
			t.Fatalf("ValidLevel(%d) = %t, want %t", test.level, got, test.want)
		}
	}
}

func TestEngineStopsControlledWorker(t *testing.T) {
	silenceLogs(t)
	started := make(chan struct{}, 1)
	stopped := make(chan struct{}, 1)
	worker := func(stop <-chan struct{}) {
		started <- struct{}{}
		<-stop
		stopped <- struct{}{}
	}
	engine := newEngine(1, worker)
	t.Cleanup(func() {
		engine.SetLevel(0)
	})

	engine.SetLevel(100)
	waitForSignal(t, started)
	if workers := engine.GetWorkers(); workers != 1 {
		t.Fatalf("workers = %d, want 1", workers)
	}

	engine.SetLevel(0)
	waitForSignal(t, stopped)
	if workers := engine.GetWorkers(); workers != 0 {
		t.Fatalf("workers = %d, want 0", workers)
	}
}

func TestEngineConcurrentAccess(t *testing.T) {
	silenceLogs(t)
	engine := newEngine(0, cpuWorker)
	var waitGroup sync.WaitGroup

	for worker := 0; worker < 8; worker++ {
		waitGroup.Add(1)
		go func(offset int) {
			defer waitGroup.Done()
			for iteration := 0; iteration < 20; iteration++ {
				engine.SetLevel((offset + iteration) % 101)
				engine.GetLevel()
				engine.GetWorkers()
			}
		}(worker)
	}

	waitGroup.Wait()
	if !ValidLevel(engine.GetLevel()) {
		t.Fatalf("invalid final level: %d", engine.GetLevel())
	}
	if workers := engine.GetWorkers(); workers != 0 {
		t.Fatalf("workers = %d, want 0", workers)
	}
	engine.SetLevel(0)
}

func TestCPUWorkerStopsImmediately(t *testing.T) {
	stop := make(chan struct{})
	done := make(chan struct{})
	close(stop)

	go func() {
		cpuWorker(stop)
		close(done)
	}()

	waitForSignal(t, done)
}

func waitForSignal(t *testing.T, signal <-chan struct{}) {
	t.Helper()
	select {
	case <-signal:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for worker")
	}
}
