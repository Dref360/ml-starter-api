import os
from typing import Optional

from fastapi import FastAPI, APIRouter

from ml_starter_api.config import Config
from ml_starter_api.ml.model_runner import ModelRunner

_model_runner: Optional[ModelRunner] = None


def get_model_runner() -> Optional[ModelRunner]:
    return _model_runner


def create_app() -> FastAPI:
    global _model_runner
    if "CFG_PATH" not in os.environ:
        raise EnvironmentError("Can't find the config in CFG_PATH")
    app = FastAPI()
    cfg = Config.parse_file(os.environ["CFG_PATH"])
    _model_runner = ModelRunner(cfg)

    from ml_starter_api.routers.predictions import router as pred_router

    api_router = APIRouter()
    api_router.include_router(pred_router, prefix="/predictions")

    app.include_router(api_router)
    return app
