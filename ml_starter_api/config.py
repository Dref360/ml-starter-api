from pydantic import BaseModel


class Config(BaseModel):
    db: str
    model_name: str
    cache_file: str = "/tmp"
