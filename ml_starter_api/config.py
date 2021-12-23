from typing import Optional

from sqlmodel import Field

from ml_starter_api.models.predictions import SQLModelWithId


class Config(SQLModelWithId, table=True):  # type: ignore
    id: Optional[int] = Field(default=None, primary_key=True)
    db: str
    model_name: str
    cache_file: str = "/tmp"
