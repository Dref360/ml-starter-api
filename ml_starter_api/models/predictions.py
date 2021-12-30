from typing import Optional, List

from sqlalchemy import Column, JSON
from sqlmodel import Field

from ml_starter_api.models.common import ValidInput, ValidOutput


class PredictionInput(ValidInput, table=True):  # type: ignore
    sentence: str
    label: Optional[int]


class PredictionOutput(ValidOutput, table=True):  # type: ignore
    input_key_id: Optional[int] = Field(default=None, foreign_key="predictioninput.id")
    distribution: List[float] = Field(sa_column=Column(JSON))
    loss: Optional[float]
    prediction: int

    # Needed for Column(JSON)
    class Config:
        arbitrary_types_allowed = True
