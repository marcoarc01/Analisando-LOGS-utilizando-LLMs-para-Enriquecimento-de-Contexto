package telemetry

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestSchema(t *testing.T) {
	var out bytes.Buffer
	logger := NewLogger(&out)
	for _, event := range []string{"state_queried", "stress_updated", "service_started", "server_error"} {
		logger.Info("Service operation completed", "event", event)
	}
	for _, line := range strings.Split(strings.TrimSpace(out.String()), "\n") {
		var record map[string]any
		if !strings.HasPrefix(line, "{") || json.Unmarshal([]byte(line), &record) != nil {
			t.Fatal("invalid JSON")
		}
		stamp, err := time.Parse(time.RFC3339Nano, record["timestamp"].(string))
		_, offset := stamp.Zone()
		if err != nil || offset != 0 {
			t.Fatal("invalid UTC timestamp")
		}
		if record["level"] != "info" || record["service"] != "stresser-app" || record["message"] == nil {
			t.Fatal(record)
		}
		for _, key := range []string{"time", "msg", "status", "duration_ms", "degradation_type"} {
			if _, ok := record[key]; ok {
				t.Fatal("unexpected field", key)
			}
		}
	}
}

func TestStandardErrorsDoNotLeakSensitiveText(t *testing.T) {
	var out bytes.Buffer
	previous := Logger
	Logger = NewLogger(&out)
	defer func() { Logger = previous }()
	input := []byte("Authorization: secret Cookie: secret body: private")
	n, err := (ErrorWriter{}).Write(input)
	if n != len(input) || err != nil || strings.Contains(out.String(), "secret") || strings.Contains(out.String(), "private") {
		t.Fatal(out.String())
	}
	var record map[string]any
	if json.Unmarshal(out.Bytes(), &record) != nil || record["event"] != "server_error" {
		t.Fatal(out.String())
	}
}
