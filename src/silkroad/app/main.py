import uvicorn
from fastapi import FastAPI
from silkroad.app.api.v1.api import api_router
from silkroad.logging.logger import logger
import typing as tp

app = FastAPI(title="Silkroad API", version="1.0.0")

app.include_router(api_router, prefix="/api/v1")

# Include health check at root level for convenience or keep it under /api/v1
# Let's keep it under /api/v1/health as per router, but maybe also a root health?
# The original server had /health at root. Let's add a redirect or just include health router at root too?
# Or just keep it clean under /api/v1.
# The original server had /health, /symbols, /data/{symbol}.
# To maintain backward compatibility with any existing clients (if any), we might want to mount at root too,
# but the plan was to move to /api/v1. I'll stick to /api/v1 but maybe alias /health.


@app.get("/health")
async def health_check() -> tp.Dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def root() -> tp.Dict[str, str]:
    return {"message": "Welcome to Silkroad API"}


def main():
    """Main function to run the FastAPI server."""
    logger.info("Starting Silkroad API Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
