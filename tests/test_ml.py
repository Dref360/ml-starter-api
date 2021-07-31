from ml_starter_api.ml.model_runner import ModelRunner


def test_ml(simple_config, simple_request):
    ml = ModelRunner(simple_config)
    mod = ml.get_model(simple_config.model_name)
    assert simple_config.model_name in ml._loaded_model

    output = ml.run_prediction(simple_request)
    assert output.loss is None

    simple_request.label = 1
    output = ml.run_prediction(simple_request)
    assert output.loss is not None
