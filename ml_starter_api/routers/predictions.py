from typing import Optional

from fastapi import APIRouter, Query, Depends

from ml_starter_api.app import get_model_runner
from ml_starter_api.ml.model_runner import ModelRunner
from ml_starter_api.models.predictions import PredictionOutput, PredictionInput

router = APIRouter()

TAGS = ["Prediction v1"]


@router.get(
    "",
    summary="Get prediction on an input.",
    description="Get the prediction for an input, if cached, will get from cache.",
    tags=TAGS,
)
def get_predictions(
    sentence: str = Query(...),
    label: Optional[int] = Query(None),
    model_runner: ModelRunner = Depends(get_model_runner),
) -> PredictionOutput:
    inp = PredictionInput(sentence=sentence, label=label)
    return model_runner.run_prediction(inp)
