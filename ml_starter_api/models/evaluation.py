from typing import Any, Dict, Optional

from sqlalchemy import Column, JSON
from sqlmodel import Field

from ml_starter_api.models.common import ValidInput, ValidOutput, DatasetDefinition


class EvaluationInput(ValidInput, table=True):  # type: ignore
    metric_name: str
    dataset: DatasetDefinition = Field(sa_column=Column(JSON))

    # Needed for Column(JSON)
    class Config:
        arbitrary_types_allowed = True


class EvaluationOutput(ValidOutput, table=True):  # type: ignore
    input_key_id: Optional[int] = Field(default=None, foreign_key="evaluationinput.id")
    extras: Dict[str, Any] = Field(sa_column=Column(JSON))
    value: Dict[str, Any] = Field(sa_column=Column(JSON))

    # Needed for Column(JSON)
    class Config:
        arbitrary_types_allowed = True
