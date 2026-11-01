package main

import (
	"reflect"
	"runtime"
	"testing"

	"github.com/marcoarc01/Analisando-LOGS-utilizando-LLMs-para-Enriquecimento-de-Contexto/stresser-app/stress"
)

func TestStartupFields(t *testing.T) {
	for _, mode := range []string{"", stress.ModeWorkers} {
		if got := startupFields(mode); !reflect.DeepEqual(got, []any{"event", "service_started"}) {
			t.Fatalf("startupFields(%q) = %v, want the v1 start event", mode, got)
		}
	}
	want := []any{"event", "service_started", "stress_mode", "duty", "gomaxprocs", runtime.GOMAXPROCS(0)}
	if got := startupFields(stress.ModeDuty); !reflect.DeepEqual(got, want) {
		t.Fatalf("startupFields(duty) = %v, want %v", got, want)
	}
}

func TestWorkIterations(t *testing.T) {
	for value, want := range map[string]int{"": 0, "0": 0, "12000": 12000} {
		if got, err := workIterations(value); err != nil || got != want {
			t.Fatalf("workIterations(%q) = %d, %v; want %d", value, got, err, want)
		}
	}
	for _, value := range []string{"-1", "abc", "1.5"} {
		if _, err := workIterations(value); err == nil {
			t.Fatalf("workIterations(%q) accepted", value)
		}
	}
}
