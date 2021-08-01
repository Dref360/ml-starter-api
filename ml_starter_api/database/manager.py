from typing import Type, TypeVar

from pydantic import BaseModel
from pymongo import MongoClient

from ml_starter_api.config import Config
from ml_starter_api.models.predictions import NamedModel

T = TypeVar("T", bound=BaseModel)


class DatabaseManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _get_database(self):
        return MongoClient(host=self.cfg.db)["db"]

    def in_cache(self, request: NamedModel) -> bool:
        db = self._get_database()
        collection = db[request.name]
        out = collection.find_one({"request": request.dict(), "config": self.cfg.dict()})
        return out is not None

    def get_cache(self, request: NamedModel, to: Type[T]) -> T:
        db = self._get_database()
        collection = db[request.name]
        out = collection.find_one({"request": request.dict(), "config": self.cfg.dict()})
        return to(**out["response"])

    def store(self, request: NamedModel, output: BaseModel):
        db = self._get_database()
        collection = db[request.name]
        out = collection.insert_one(
            {
                "request": request.dict(),
                "config": self.cfg.dict(),
                "response": output.dict(),
            }
        )
        return out is None
