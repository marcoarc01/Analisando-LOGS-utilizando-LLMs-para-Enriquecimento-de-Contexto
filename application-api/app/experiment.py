from dataclasses import dataclass

from starlette.datastructures import State
from starlette.responses import JSONResponse

from app.stresser_client import StresserUnavailable


@dataclass
class ExperimentContext:
    request_id: str | None = None
    experiment_run_id: str | None = None
    stress_level: int | None = None
    delay_ms: float = 0
    degradation_type: str = "none"
    event: str = "request_rejected"


async def apply_experiment(
    state: State, context: ExperimentContext, *, degrade: bool = True
) -> JSONResponse | None:
    metrics = state.metrics
    try:
        context.stress_level = await state.stresser.get_level(
            request_id=context.request_id,
            experiment_run_id=context.experiment_run_id,
        )
        metrics.stress.set(context.stress_level)
    except StresserUnavailable:
        metrics.lookup_failures.inc()
        metrics.stress.set(float("nan"))
        context.degradation_type = "dependency_unavailable"
        context.event = "stresser_lookup_failed"
        return JSONResponse({"detail": "Dependência indisponível"}, status_code=503)

    if not degrade:
        return None

    decision = state.policy.decide(context.stress_level)
    context.delay_ms = decision.delay_ms
    context.degradation_type = decision.degradation_type
    if context.delay_ms > 0:
        await state.sleep(context.delay_ms / 1000)
    if context.degradation_type != "none":
        metrics.degradations.labels(context.degradation_type).inc()
    if decision.status_code is not None:
        status = decision.status_code
        context.event = f"experimental_{status}"
        if status == 500:
            metrics.errors_500.inc()
        else:
            metrics.errors_503.inc()
        return JSONResponse(
            {
                "detail": "Falha experimental"
                if status == 500
                else "Indisponibilidade experimental"
            },
            status_code=status,
        )
    return None
