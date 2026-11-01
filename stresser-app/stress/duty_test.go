package stress

import (
	"runtime"
	"testing"
	"time"
)

func TestDutyCycle(t *testing.T) {
	for _, test := range []struct {
		level      int
		busy, idle time.Duration
	}{
		{0, 0, 100 * time.Millisecond},
		{75, 75 * time.Millisecond, 25 * time.Millisecond},
		{80, 80 * time.Millisecond, 20 * time.Millisecond},
		{92, 92 * time.Millisecond, 8 * time.Millisecond},
		{100, 100 * time.Millisecond, 0},
	} {
		busy, idle := DutyCycle(test.level, DutyPeriod)
		if busy != test.busy || idle != test.idle {
			t.Fatalf("DutyCycle(%d) = %v/%v, want %v/%v", test.level, busy, idle, test.busy, test.idle)
		}
	}
}

func TestDesiredWorkersByMode(t *testing.T) {
	for _, test := range []struct {
		mode             string
		level, max, want int
	}{
		{ModeWorkers, 0, 12, 0},
		{ModeWorkers, 75, 12, 9},
		{ModeWorkers, 80, 12, 9},
		{ModeWorkers, 92, 12, 11},
		{ModeWorkers, 100, 2, 2},
		{ModeDuty, 0, 2, 0},
		{ModeDuty, 1, 2, 2},
		{ModeDuty, 75, 2, 2},
		{ModeDuty, 92, 12, 12},
	} {
		if got := DesiredWorkers(test.mode, test.level, test.max); got != test.want {
			t.Fatalf("DesiredWorkers(%s, %d, %d) = %d, want %d", test.mode, test.level, test.max, got, test.want)
		}
	}
}

func TestNewEngineForMode(t *testing.T) {
	for mode, want := range map[string]string{"": ModeWorkers, "workers": ModeWorkers, "duty": ModeDuty} {
		engine, err := NewEngineForMode(mode)
		if err != nil || engine.Mode() != want {
			t.Fatalf("NewEngineForMode(%q) = %v, %v; want mode %s", mode, engine, err, want)
		}
	}
	if _, err := NewEngineForMode("fraction"); err == nil {
		t.Fatal("unknown mode accepted")
	}
}

func TestDutyEngineStartsOneWorkerPerCPU(t *testing.T) {
	silenceLogs(t)
	type cycle struct{ busy, idle time.Duration }
	started := make(chan cycle, 4)
	stopped := make(chan struct{}, 4)
	engine := newDutyEngine(2, func(stop <-chan struct{}, busy, idle time.Duration) {
		started <- cycle{busy, idle}
		<-stop
		stopped <- struct{}{}
	})
	t.Cleanup(func() { engine.SetLevel(0) })

	engine.SetLevel(80)
	for i := 0; i < 2; i++ {
		select {
		case got := <-started:
			if got != (cycle{80 * time.Millisecond, 20 * time.Millisecond}) {
				t.Fatalf("worker cycle = %v", got)
			}
		case <-time.After(time.Second):
			t.Fatal("timed out waiting for worker")
		}
	}
	if workers := engine.GetWorkers(); workers != 2 {
		t.Fatalf("workers = %d, want 2", workers)
	}

	engine.SetLevel(0)
	for i := 0; i < 2; i++ {
		waitForSignal(t, stopped)
	}
	if workers := engine.GetWorkers(); workers != 0 {
		t.Fatalf("workers = %d, want 0", workers)
	}
}

func TestDutyWorkerStops(t *testing.T) {
	for _, cycle := range [][2]time.Duration{{5 * time.Millisecond, 5 * time.Millisecond}, {10 * time.Millisecond, 0}} {
		stop := make(chan struct{})
		done := make(chan struct{})
		go func() {
			dutyWorker(stop, cycle[0], cycle[1])
			close(done)
		}()
		time.Sleep(25 * time.Millisecond)
		close(stop)
		waitForSignal(t, done)
	}
}

func TestDutyEngineFollowsGOMAXPROCS(t *testing.T) {
	silenceLogs(t)
	previous := runtime.GOMAXPROCS(0)
	t.Cleanup(func() { runtime.GOMAXPROCS(previous) })
	for _, procs := range []int{1, 3} {
		runtime.GOMAXPROCS(procs)
		duty, _ := NewEngineForMode(ModeDuty)
		workers, _ := NewEngineForMode(ModeWorkers)
		if duty.maxWorkers != procs {
			t.Fatalf("duty engine with GOMAXPROCS=%d has %d workers", procs, duty.maxWorkers)
		}
		if workers.maxWorkers != runtime.NumCPU() {
			t.Fatalf("workers engine changed with GOMAXPROCS=%d: %d", procs, workers.maxWorkers)
		}
	}
}
