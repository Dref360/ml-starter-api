from typing import Optional

from pydantic import BaseModel
from sqlmodel import Field, SQLModel


class DatasetDefinition(SQLModel):
    name: str
    split: str
    text_column: str
    label_column: str


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
