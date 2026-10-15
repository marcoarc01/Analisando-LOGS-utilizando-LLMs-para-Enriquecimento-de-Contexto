# Validação funcional com k6

A imagem grafana/k6:1.8.1 envia 8,53 requisições por segundo para a FastAPI.
Cada execução dura 30 segundos por padrão e deve ser executada sequencialmente.
O teste não altera o stress; use o endpoint de controle antes de cada execução.

## Executar

Na raiz do projeto, com os oito serviços ativos:

```bash
K6_RUN_ID="$(uuidgen)"
python3 load-tests/run.py --testid "$K6_RUN_ID" --scenario baseline
```

Use um UUID v4 novo por execução. O launcher define K6_EXPERIMENT_RUN_ID;
esse valor é idêntico à tag testid, ao header X-Experiment-Run-ID e ao campo
experiment_run_id nos logs. Nunca use nomes que revelem a condição experimental.

O launcher rejeita outro k6 ativo e usa um nome fixo de container como trava
atômica. Executa Compose com --rm, --no-deps e sem TTY, verificando a remoção
ao terminar. Invocações diretas de Docker podem contornar essa proteção; use
sempre o launcher. Ctrl+C encerra o container pertencente à execução.

## Configuração

| Variável | Uso |
| --- | --- |
| K6_EXPERIMENT_RUN_ID | UUID v4 obrigatório, fornecido pelo launcher via --testid |
| K6_SCENARIO | baseline, stress_cpu, recovery, network_latency, network_timeout ou combined |
| K6_TEST_DURATION | 30s por padrão; inteiro positivo seguido de s, m ou h |
| K6_BASE_URL | Origem HTTP sem caminho, query ou credenciais |
| K6_MAX_EXPERIMENT_DELAY_MS | Recebe MAX_EXPERIMENT_DELAY_MS do Compose, padrão 2000 |

| Executor | Taxa | preAllocatedVUs | maxVUs | Esperado em 30s |
| --- | --- | --- | --- | --- |
| products | 53/10s (5,3/s) | 27 | 38 | 159 |
| orders | 19/10s (1,9/s) | 10 | 14 | 57 |
| health | 9/10s (0,9/s) | 5 | 7 | 27 |
| invalid | 43/100s (0,43/s) | 3 | 4 | 13 |

Os intervalos entre chegadas (188,7, 526,3, 1.111,1 e 2.325,6 ms) não são múltiplos
de 100 ms, para não travar na fase do ciclo de cota de CPU do stresser-app.
O fluxo invalid (~5% do total) alterna POST /orders com produto inexistente (404)
e quantidade 0 (422); ambos são respostas esperadas e não consultam o Go.

Capacidade: ceil(taxa por segundo × (atraso máximo em segundos + 3)) pré-alocados e
ceil(taxa × (atraso máximo em segundos + 5)) máximos. A margem cobre a consulta
à dependência e o agendamento. Timeout HTTP: 5s; gracefulStop: 5s.
As taxas de chegada não mudam com o stress.

Thresholds exigem zero dropped_iterations, preservação de headers em 100% das
requisições, 100% dos checks do fluxo invalid e pelo menos 98% da quantidade
prevista em cada endpoint.
O resumo calcula requests_per_scheduled_second usando os 30s de carga.
A duração total do processo inclui inicialização e drenagem; dividir requisições
por esse tempo total subestima a taxa de chegada. Uma iteração na borda pode
produzir até uma requisição extra por executor.

Os checks de status continuam esperando 200/201 e falham deliberadamente para
500/503 experimentais. Esses checks não abortam os testes de stress.
Os pedidos válidos usam apenas {"product_id":1,"quantity":2}; o fluxo invalid usa
{"product_id":999,"quantity":2} e {"product_id":1,"quantity":0}.

## Métricas

Remote Write: K6_PROMETHEUS_RW_TREND_STATS=avg,p(50),p(95),p(99),max.
A imagem exporta k6_http_req_duration_avg, _p50, _p95, _p99 e _max em segundos.
No resumo, os mesmos valores estão em milissegundos. São snapshots de tendências:
não use increase/rate nesses valores, não some percentis e não misture execuções.

Labels controladas: testid, scenario, endpoint, method, status, expected_response
e check, conforme a métrica. scenario identifica o experimento (baseline,
stress_cpu, recovery); endpoint identifica cada fluxo. request_id, url e name
não são labels. A cardinalidade de testid cresce com o número controlado de
experimentos: não reutilize UUIDs nem gere IDs por requisição nessa tag.

Exemplo de consulta; substitua UUID pelo mesmo K6_RUN_ID do comando:

```promql
sum by (endpoint, status) (last_over_time(k6_http_reqs_total{testid="UUID"}[5m]))
k6_http_req_duration_p50{testid="UUID"}
k6_http_req_duration_p95{testid="UUID"}
k6_http_req_duration_p99{testid="UUID"}
k6_http_req_duration_max{testid="UUID"}
k6_checks_rate{testid="UUID"}
k6_vus{testid="UUID"}
k6_dropped_iterations_total{testid="UUID"}
up{job=~"application-api|stresser-app"}
```

O total final k6 pertence a uma única execução, identificada por UUID e iniciada
em zero. Para counters persistentes FastAPI/Go, compare snapshots antes/depois
ou use increase com atenção às bordas de scrape. /metrics também é contado.
Se dropped_iterations não produzir série, confirme zero pelo resumo e pelo
threshold; ausência de série isoladamente não é prova de zero.

## Logs e correlação

Promtail remove o envelope Docker, usa seu timestamp e mantém o JSON interno.
Somente service (application-api ou stresser-app) é promovido a label de aplicação.
IDs, event, message e demais campos permanecem no conteúdo:

```logql
{service="application-api"} | json | experiment_run_id="UUID"
{service="stresser-app"} | json | experiment_run_id="UUID"
{service=~"application-api|stresser-app"} | json | request_id="REQUEST_UUID"
{service="application-api"} | json | __error__ != ""
{service="stresser-app"} | json | __error__ != ""
```

O schema comum possui timestamp UTC RFC3339, level minúsculo, service, event
snake_case e message estável. Campos HTTP e de correlação são opcionais.
FastAPI emite um evento final por request, mais latency_injected quando há atraso;
conte eventos finais ou request_ids distintos. Go emite state_queried e um único
stress_updated por alteração. /live, /ready e /metrics não têm logs de acesso
experimentais da FastAPI.

Campos status_code, stress_level, degradation_type, experiment_run_id, request_id,
scenario e testid ficam apenas na auditoria bruta, excluídos de futuros modelos.
Nomes de eventos que revelem intervenções também exigem exclusão. Esta etapa
não implementa normalização, dataset ou modelos.

## Testes

```bash
node --test load-tests/baseline.test.cjs
docker compose --profile loadtest config --quiet
K6_RUN_ID="$(uuidgen)"
docker compose --profile loadtest run --rm --no-deps -T \
  -e K6_EXPERIMENT_RUN_ID="$K6_RUN_ID" \
  k6 inspect --include-system-env-vars /scripts/baseline.js
```

Sem K6_EXPERIMENT_RUN_ID válido, inspect e run devem falhar.
Veja [TESTE-MANUAL.md](TESTE-MANUAL.md) para a sequência dos quatro cenários.
