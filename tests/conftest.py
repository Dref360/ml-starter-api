import pytest
from sqlmodel import SQLModel

from ml_starter_api.app import get_engine
from ml_starter_api.config import Config
from ml_starter_api.database.manager import DatabaseManager
from ml_starter_api.models.predictions import PredictionInput


@pytest.fixture
def simple_config(tmpdir):
    db_path = str(tmpdir / "database.db")
    return Config(db=db_path, model_name="distilbert-base-uncased-finetuned-sst-2-english",
                  cache_file='/tmp')


@pytest.fixture
def simple_request(db_manager):
    inp = PredictionInput(sentence="hello, this is me")
    inp = db_manager.get_or_insert(inp)
    return inp


@pytest.fixture
def simple_request_with_label(db_manager):
    inp = PredictionInput(sentence="This is a pizza shop!", label=1)
    inp = db_manager.get_or_insert(inp)
    return inp


@pytest.fixture
def engine(simple_config):
    # We import all routers here, they have the models!
    # TODO: Is that a good thing?
    from ml_starter_api.routers.predictions import router as pred_router
    engine = get_engine(simple_config, echo=False)
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_manager(simple_config, engine):
    dm = DatabaseManager(cfg=simple_config, engine=engine)
    dm.get_or_insert(simple_config)
    return dm
