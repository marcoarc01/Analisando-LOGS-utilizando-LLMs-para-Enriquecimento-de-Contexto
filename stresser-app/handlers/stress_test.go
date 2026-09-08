package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestStressHandler(t *testing.T) {
	tests := []struct {
		name           string
		body           string
		initialLevel   int
		initialWorkers int
		wantStatus     int
		wantResponse   StressResponse
		wantError      string
		wantSetLevels  []int
		wantLevel      int
		wantWorkers    int
	}{
		{
			name:          "valid level",
			body:          `{"level":37}`,
			wantStatus:    http.StatusOK,
			wantResponse:  StressResponse{Message: "Stress level atualizado", StressLevel: 37, CPUWorkers: 0},
			wantSetLevels: []int{37},
			wantLevel:     37,
		},
		{
			name:           "zero level",
			body:           `{"level":0}`,
			initialLevel:   37,
			initialWorkers: 2,
			wantStatus:     http.StatusOK,
			wantResponse:   StressResponse{Message: "Stress level atualizado", StressLevel: 0, CPUWorkers: 0},
			wantSetLevels:  []int{0},
		},
		{
			name:         "negative level",
			body:         `{"level":-1}`,
			initialLevel: 20,
			wantStatus:   http.StatusBadRequest,
			wantError:    `{"error": "level deve ser entre 0 e 100"}` + "\n",
			wantLevel:    20,
		},
		{
			name:         "level above maximum",
			body:         `{"level":101}`,
			initialLevel: 20,
			wantStatus:   http.StatusBadRequest,
			wantError:    `{"error": "level deve ser entre 0 e 100"}` + "\n",
			wantLevel:    20,
		},
		{
			name:         "invalid JSON",
			body:         `{"level":`,
			initialLevel: 20,
			wantStatus:   http.StatusBadRequest,
			wantError:    `{"error": "JSON inválido"}` + "\n",
			wantLevel:    20,
		},
		{
			name:         "empty body",
			initialLevel: 20,
			wantStatus:   http.StatusBadRequest,
			wantError:    `{"error": "JSON inválido"}` + "\n",
			wantLevel:    20,
		},
		{
			name:         "missing level field",
			body:         `{}`,
			initialLevel: 20,
			wantStatus:   http.StatusBadRequest,
			wantError:    `{"error": "JSON inválido"}` + "\n",
			wantLevel:    20,
		},
		{
			name:          "maximum level",
			body:          `{"level":100}`,
			wantStatus:    http.StatusOK,
			wantResponse:  StressResponse{Message: "Stress level atualizado", StressLevel: 100, CPUWorkers: 0},
			wantSetLevels: []int{100},
			wantLevel:     100,
		},
		{
			name:          "trailing whitespace",
			body:          "{\"level\":37}\n \t",
			wantStatus:    http.StatusOK,
			wantResponse:  StressResponse{Message: "Stress level atualizado", StressLevel: 37, CPUWorkers: 0},
			wantSetLevels: []int{37},
			wantLevel:     37,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			engine := &stubEngine{level: test.initialLevel, workers: test.initialWorkers}
			request := httptest.NewRequest(http.MethodPost, "/api/stress", strings.NewReader(test.body))
			response := httptest.NewRecorder()

			StressHandler(engine)(response, request)

			if response.Code != test.wantStatus {
				t.Fatalf("status = %d, want %d", response.Code, test.wantStatus)
			}
			if len(engine.setLevels) != len(test.wantSetLevels) {
				t.Fatalf("SetLevel calls = %v, want %v", engine.setLevels, test.wantSetLevels)
			}
			for index := range test.wantSetLevels {
				if engine.setLevels[index] != test.wantSetLevels[index] {
					t.Fatalf("SetLevel calls = %v, want %v", engine.setLevels, test.wantSetLevels)
				}
			}
			if engine.level != test.wantLevel || engine.workers != test.wantWorkers {
				t.Fatalf("engine state = (%d, %d), want (%d, %d)", engine.level, engine.workers, test.wantLevel, test.wantWorkers)
			}

			if test.wantError != "" {
				if body := response.Body.String(); body != test.wantError {
					t.Fatalf("body = %q, want %q", body, test.wantError)
				}
				return
			}

			if contentType := response.Header().Get("Content-Type"); contentType != "application/json" {
				t.Fatalf("Content-Type = %q, want %q", contentType, "application/json")
			}
			var got StressResponse
			if err := json.Unmarshal(response.Body.Bytes(), &got); err != nil {
				t.Fatalf("invalid JSON response: %v", err)
			}
			if got != test.wantResponse {
				t.Fatalf("response = %#v, want %#v", got, test.wantResponse)
			}
		})
	}
}

func TestStressHandlerRejectsInvalidBodiesWithoutChangingState(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{"null level", `{"level":null}`},
		{"null body", `null`},
		{"string level", `{"level":"20"}`},
		{"fractional level", `{"level":20.5}`},
		{"float level", `{"level":20.0}`},
		{"boolean level", `{"level":true}`},
		{"array body", `[20]`},
		{"whitespace body", " \n\t"},
		{"unknown field", `{"level":20,"extra":1}`},
		{"second object", `{"level":20}{"level":0}`},
		{"trailing null", `{"level":20} null`},
		{"trailing text", `{"level":20} trailing`},
		{"trailing invalid JSON", `{"level":20} {`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			engine := &stubEngine{level: 25, workers: 2}
			response := httptest.NewRecorder()
			request := httptest.NewRequest(http.MethodPost, "/api/stress", strings.NewReader(test.body))
			StressHandler(engine)(response, request)
			if response.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400", response.Code)
			}
			if engine.level != 25 || engine.workers != 2 || len(engine.setLevels) != 0 {
				t.Fatalf("invalid request changed engine state: %#v", engine)
			}
		})
	}
}
