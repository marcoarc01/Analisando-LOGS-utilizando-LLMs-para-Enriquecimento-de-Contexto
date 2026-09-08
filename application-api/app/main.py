import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config import Settings
from app.degradation import DegradationPolicy
from app.middleware import ExperimentMiddleware
from app.routes import health, orders, products
from app.stresser_client import StresserClient
from app.telemetry import Metrics, create_logger


def create_app(settings=None, stresser_factory=StresserClient, sleep=asyncio.sleep):
    @asynccontextmanager
    async def lifespan(application):
        config = settings or Settings.from_env()
        application.state.stresser = stresser_factory(config)
        application.state.policy = DegradationPolicy(
            config.experiment_seed, config.max_experiment_delay_ms
        )
        application.state.sleep = sleep
        application.state.logger = create_logger()
        try:
            yield
        finally:
            await application.state.stresser.aclose()

    application = FastAPI(title="Application API", version="1.0.0", lifespan=lifespan)
    application.state.metrics = Metrics()
    application.add_middleware(ExperimentMiddleware)
    application.include_router(health.router)
    application.include_router(products.router)
    application.include_router(orders.router)

    @application.get("/metrics", include_in_schema=False)
    async def metrics(request: Request) -> Response:
        return Response(
            generate_latest(request.app.state.metrics.registry),
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )

    return application


app = create_app()
