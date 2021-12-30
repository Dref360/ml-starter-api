from typing import Type, TypeVar, Optional

from sqlmodel import Session, select

from ml_starter_api.config import Config
from ml_starter_api.models.common import (
    ValidOutput,
    ValidInput,
    SQLModelWithId,
)

T = TypeVar("T", bound=SQLModelWithId)


class DatabaseManager:
    def __init__(self, cfg: Config, engine):
        self.cfg = cfg
        self.engine = engine

    def get_cache(self, request: ValidInput, to: Type[T]) -> Optional[T]:
        with Session(self.engine) as session:
            statement = (
                select(to).where(to.input_key_id == request.id).where(to.config_id == self.cfg.id)
            )
            results = session.exec(statement)
            out = results.first()

        return out

    def get_or_insert(self, item: T) -> T:
        with Session(self.engine) as session:
            all_similar = session.exec(select(type(item)))
            found = [similar for similar in all_similar if item == similar]
            if found:
                item = found[0]
            else:
                session.add(item)
                session.commit()
                session.refresh(item)
        # Nested JSON Pydantic items are not formatted correctly.
        reformatted: T = type(item)(**item.dict())
        return reformatted

    def store(self, output: ValidOutput) -> Optional[int]:
        with Session(self.engine) as session:
            session.add(output)
            session.commit()
            session.refresh(output)
            output_id: Optional[int] = output.id
            return output_id
