from ml_starter_api.ml.model_runner import ModelRunner


def test_ml(simple_config, simple_request, simple_request_with_label, db_manager):
    ml = ModelRunner(simple_config, db_manager=db_manager)
    mod = ml.get_model(simple_config.model_name)
    assert simple_config.model_name in ml._loaded_model

    output = ml.run_task(simple_request)
    assert output.loss is None

    output = ml.run_task(simple_request_with_label)
    assert output.loss is not None
