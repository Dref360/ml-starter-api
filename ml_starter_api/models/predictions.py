from typing import Optional, List

from sqlalchemy import Column, JSON
from sqlmodel import SQLModel, Field


class SQLModelWithId(SQLModel):
    id: Optional[int] = Field(default=None, primary_key=True)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        own_dict = self.dict(exclude={"id": True})
        other_dict = other.dict()
        for key, v in own_dict.items():
            if v != other_dict[key]:
                return False
        return True


class ValidInput(SQLModelWithId):
    pass


class ValidOutput(SQLModelWithId):
    config_id: Optional[int] = Field(default=None, foreign_key="config.id")


class PredictionInput(ValidInput, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    sentence: str
    label: Optional[int]


class PredictionOutput(ValidOutput, table=True):
    input_key_id: Optional[int] = Field(default=None, foreign_key="predictioninput.id")
    distribution: List[float] = Field(sa_column=Column(JSON))
    loss: Optional[float]
    prediction: int

    # Needed for Column(JSON)
    class Config:
        arbitrary_types_allowed = True
