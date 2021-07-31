from typing import Optional, List

from pydantic import BaseModel


class NamedModel(BaseModel):
    name: str
    # TODO add fields to include/exclude for the cache.


class PredictionInput(NamedModel):
    name: str = "Prediction"
    sentence: str
    label: Optional[int]


class PredictionOutput(BaseModel):
    distribution: List[float]
    loss: Optional[float]
    prediction: int
