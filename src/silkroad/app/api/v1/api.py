from fastapi import APIRouter
from silkroad.app.api.v1.endpoints import health, database, estimate_cov

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(database.router, tags=["database"])
api_router.include_router(estimate_cov.router, tags=["estimation"])
