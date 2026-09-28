# Validação manual dos quatro cenários

Execute na raiz do projeto, com Docker ativo, .env configurado e os oito serviços
normais funcionando. Não execute dois testes ao mesmo tempo.

```bash
docker compose config --quiet
docker compose ps
curl --fail-with-body -sS http://localhost:8000/live
curl --fail-with-body -sS http://localhost:8080/health
```

Para cada linha da tabela, configure o nível e gere um UUID novo. Espere o teste
terminar antes da linha seguinte.

| Ordem | level | scenario |
| --- | --- | --- |
| 1 | 0 | baseline |
| 2 | 80 | stress_cpu |
| 3 | 100 | stress_cpu |
| 4 | 0 | recovery |

Exemplo baseline (troque somente LEVEL e SCENARIO para as demais linhas):

```bash
LEVEL=0
SCENARIO=baseline
curl --fail-with-body -sS -X POST http://localhost:8080/api/stress \
  -H 'Content-Type: application/json' -d "{\"level\":$LEVEL}"
curl --fail-with-body -sS http://localhost:8080/api/state
K6_RUN_ID="$(uuidgen)"
date -u
python3 load-tests/run.py --testid "$K6_RUN_ID" --scenario "$SCENARIO"
date -u
printf 'testid / X-Experiment-Run-ID / experiment_run_id: %s\n' "$K6_RUN_ID"
```

O launcher exige UUID v4 e impede duas execuções gerenciadas concorrentes.
O teste dura 30s e mantém 5/2/1 req/s para products/orders/health.
O mesmo K6_RUN_ID identifica métricas, headers e logs; copie esse UUID nas
consultas documentadas em [README.md](README.md).

Monitore os oito containers durante stress. Se houver instabilidade, interrompa
com Ctrl+C e retorne level para 0 imediatamente. Antes da recuperação, confirme
zero workers e /live e Go /health com HTTP 200.

## Conferências

Em cada execução: aproximadamente 240 requisições (até uma extra por executor),
requests_per_scheduled_second próximo de 8 e dropped_iterations zero.
Verifique avg, p(50), p(95), p(99), max no resumo e as cinco séries Prometheus,
sempre filtradas pelo testid. Verifique os thresholds e o código de saída.

Baseline e recuperação devem retornar 200 em /products e /health, 201 em /orders,
sem falhas. Stress 80 pode retornar 500; stress 100 retorna 500/503 de acordo com
a política existente. Checks de status falham nesses casos por definição.

Capture counters FastAPI e Go antes/depois da carga. Evite consultas auxiliares
a /api/state durante esse intervalo. O counter Go http_requests_total inclui
/api/state, com path normalizado e status. Logs finais FastAPI, state_queried Go
e IDs distintos devem fechar com o resumo k6. latency_injected é um evento
adicional, não uma segunda requisição.

No Loki, service deve ter somente application-api e stresser-app. Execute:

```logql
{service="application-api"} | json | __error__ != ""
{service="stresser-app"} | json | __error__ != ""
{service="application-api"} | json | experiment_run_id="UUID" | event!="latency_injected"
{service="stresser-app"} | json | experiment_run_id="UUID" | event="state_queried"
```

Use um intervalo posterior à recriação dos serviços corrigidos: logs históricos
já persistidos conservam o formato anterior. Nenhuma linha atual da aplicação
deve ter erro de parsing. Nunca envie dados pessoais ou credenciais nos IDs.

## Pedido inválido

```bash
REQUEST_ID="$(uuidgen)"
K6_RUN_ID="$(uuidgen)"
curl -i -sS -X POST http://localhost:8000/orders \
  -H 'Content-Type: application/json' \
  -H "X-Request-ID: $REQUEST_ID" \
  -H "X-Experiment-Run-ID: $K6_RUN_ID" -d '{}'
```

Espere 422, um request_rejected na FastAPI e nenhum state_queried com REQUEST_ID
no Go. Os counters de degradação não devem aumentar.

## Encerramento obrigatório

```bash
curl --fail-with-body -sS -X POST http://localhost:8080/api/stress \
  -H 'Content-Type: application/json' -d '{"level":0}'
curl --fail-with-body -sS http://localhost:8080/api/state
docker compose ps
docker ps -a --filter label=com.docker.compose.service=k6 --format '{{.Names}} {{.Status}}'
```

Confirme 0/0, oito serviços ativos, ambos os alvos up=1 e ausência de container
k6. O launcher usa --rm e não remove volumes. Os quatro testes constituem
validação funcional preliminar; não são coleta definitiva de dataset.
