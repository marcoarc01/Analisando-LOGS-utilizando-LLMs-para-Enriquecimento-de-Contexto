package telemetry

import (
	"io"
	"log/slog"
	"os"
	"strings"
)

// Raw control fields are audit metadata, not future model inputs:
// status_code, stress_level, degradation_type, experiment_run_id, request_id,
// scenario, testid. Event names revealing interventions must also be excluded.
func NewLogger(output io.Writer) *slog.Logger {
	return slog.New(slog.NewJSONHandler(output, &slog.HandlerOptions{
		ReplaceAttr: func(groups []string, a slog.Attr) slog.Attr {
			switch a.Key {
			case slog.TimeKey:
				return slog.Time("timestamp", a.Value.Time().UTC())
			case slog.LevelKey:
				return slog.String("level", strings.ToLower(a.Value.String()))
			case slog.MessageKey:
				a.Key = "message"
			}
			return a
		},
	})).With("service", "stresser-app")
}

var Logger = NewLogger(os.Stdout)

// Standard-library server errors can contain URLs or panic values. Retain a
// stable diagnostic event without copying potentially sensitive free text.
type ErrorWriter struct{}

func (ErrorWriter) Write(p []byte) (int, error) {
	Logger.Error("Server operation failed", "event", "server_error")
	return len(p), nil
}
