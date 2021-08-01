from copy import deepcopy

import pytest

from ml_starter_api.database.manager import DatabaseManager
from ml_starter_api.models.predictions import PredictionOutput


def test_database_manager(simple_config, simple_request):
    manager = DatabaseManager(simple_config)
    assert not manager.in_cache(simple_request)

    # Store data
    output = PredictionOutput(distribution=[0.6, 0.7], loss=None, prediction=0)
    manager.store(simple_request, output)
    assert manager.in_cache(simple_request)

    # Retrieving the data get the same data
    retrieved = manager.get_cache(simple_request, PredictionOutput)
    assert retrieved == output

    # Another request is not in cache
    request2 = deepcopy(simple_request)
    request2.label = 0
    assert not manager.in_cache(request2)

    # Modifying the config, it's not in cache
    cfg = deepcopy(simple_config)
    cfg.model_name = 'test_model'
    manager.cfg = cfg
    assert not manager.in_cache(simple_request)


if __name__ == '__main__':
    pytest.main()
