from copy import deepcopy

import pytest

from ml_starter_api.config import Config
from ml_starter_api.models.predictions import PredictionOutput


def test_database_manager(simple_config, simple_request, db_manager):
    assert db_manager.get_cache(simple_request, PredictionOutput) is None

    # Store data
    output = PredictionOutput(input_key_id=simple_request.id,
                              config_id=simple_config.id,
                              distribution=[0.6, 0.7], loss=None, prediction=0)
    db_manager.store(output)
    assert db_manager.get_cache(simple_request, PredictionOutput) is not None

    # Retrieving the data get the same data
    retrieved = db_manager.get_cache(simple_request, PredictionOutput)
    assert retrieved == output

    # Another request is not in cache
    request2 = deepcopy(simple_request)
    request2.label = 0
    assert not db_manager.get_cache(request2, PredictionOutput) is None

    # Modifying the config, it's not in cache
    cfg = Config(**{**simple_config.dict(exclude={"id":True}), **{"model_name": "test_model"}})
    cfg = db_manager.get_or_insert(cfg)
    db_manager.cfg = cfg
    assert db_manager.get_cache(simple_request, PredictionOutput) is None


if __name__ == '__main__':
    pytest.main()
