# Analisando Logs utilizando LLMs para Enriquecimento de Contexto

O baseline local com Grafana k6, seus comandos de execução, métricas e correlação
entre FastAPI e Go estão documentados em [load-tests/README.md](load-tests/README.md).

```bash
K6_RUN_ID="$(uuidgen)"
python3 load-tests/run.py --testid "$K6_RUN_ID" --scenario baseline
```
