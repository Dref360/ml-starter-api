import json
import os
from typing import Optional

import pydantic.json
from fastapi import FastAPI, APIRouter
from sqlmodel import SQLModel, create_engine

from ml_starter_api.config import Config
from ml_starter_api.database.manager import DatabaseManager
from ml_starter_api.ml.model_runner import ModelRunner

_model_runner: Optional[ModelRunner] = None
_config: Optional[Config] = None
_database_manager: Optional[DatabaseManager] = None


def get_model_runner() -> Optional[ModelRunner]:
    return _model_runner


def get_config() -> Optional[Config]:
    return _config


def get_database_manager() -> Optional[DatabaseManager]:
    return _database_manager


def create_app() -> FastAPI:
    global _model_runner, _config, _database_manager
    if "CFG_PATH" not in os.environ:
        raise EnvironmentError("Can't find the config in CFG_PATH")
    app = FastAPI()
    _config = Config.parse_file(os.environ["CFG_PATH"])
    engine = get_engine(_config)
    _database_manager = DatabaseManager(_config, engine)
    _model_runner = ModelRunner(_config, _database_manager)

    from ml_starter_api.routers.predictions import router as pred_router
    from ml_starter_api.routers.evaluation import router as evaluation_router

    api_router = APIRouter()
    api_router.include_router(pred_router, prefix="/predictions")
    api_router.include_router(evaluation_router, prefix="/evaluation")

    app.include_router(api_router)

    # At the end probably?
    def create_db_and_tables():
        global _config
        SQLModel.metadata.create_all(engine)
        # Store config
        _config = _database_manager.get_or_insert(_config)

    @app.on_event("startup")
    def on_startup():
        create_db_and_tables()

    return app


def _custom_json_serializer(*args, **kwargs) -> str:
    """
    Encodes json in the same way that pydantic does.
    """
    return json.dumps(*args, default=pydantic.json.pydantic_encoder, **kwargs)


def get_engine(cfg, echo=True):
    sqlite_url = f"sqlite:///{cfg.db}"
    connect_args = {"check_same_thread": False}
    engine = create_engine(
        sqlite_url, echo=echo, connect_args=connect_args, json_serializer=_custom_json_serializer
    )
    return engine
