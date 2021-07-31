import pytest
from pymongo import MongoClient

from ml_starter_api.config import Config
from ml_starter_api.models.predictions import PredictionInput


@pytest.fixture(autouse=True)
def cleanup_db(simple_config):
    client = MongoClient(host=simple_config.db)
    client.drop_database('db')
    client.close()


@pytest.fixture
def simple_config():
    return Config(db="mongodb://0.0.0.0:27017", model_name="distilbert-base-uncased-finetuned-sst-2-english",
                  cache_file='/tmp')


@pytest.fixture
def simple_request():
    return PredictionInput(sentence="hello, this is me")
