from fastapi import APIRouter, Query, Depends, Body

from ml_starter_api.app import get_model_runner
from ml_starter_api.ml.model_runner import ModelRunner
from ml_starter_api.models.common import DatasetDefinition
from ml_starter_api.models.evaluation import EvaluationInput, EvaluationOutput

router = APIRouter()

TAGS = ["Evaluation v1"]


@router.post(
    "",
    summary="Get Evaluation on a dataset.",
    description="Get the evaluation for a metric, if cached, will get from cache.",
    tags=TAGS,
)
def post_evaluation(
    metric: str = Query(...),
    dataset: DatasetDefinition = Body(...),
    model_runner: ModelRunner = Depends(get_model_runner),
) -> EvaluationOutput:
    inp = EvaluationInput(metric_name=metric, dataset=dataset)
    inp = model_runner.db_manager.get_or_insert(inp)
    return model_runner.run_task(inp)
