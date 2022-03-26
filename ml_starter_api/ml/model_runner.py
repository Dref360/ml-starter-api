from dataclasses import dataclass
from typing import Optional, Dict, Callable, Type, overload

import numpy as np
import structlog
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_metric
from transformers import (
    Pipeline,
)

from ml_starter_api.config import Config
from ml_starter_api.database.manager import DatabaseManager
from ml_starter_api.ml.loading import load_text_classif_pipeline
from ml_starter_api.models.common import ValidOutput, ValidInput
from ml_starter_api.models.evaluation import EvaluationInput, EvaluationOutput
from ml_starter_api.models.predictions import PredictionOutput, PredictionInput

log = structlog.get_logger("ModelRunner")


@dataclass
class TypedFunctor:
    function: Callable
    output_type: Type[ValidOutput]


class ModelRunner:
    def __init__(self, cfg: Config, db_manager: DatabaseManager):
        """Runner for machine learning operations.

        Args:
            cfg: Configuration object
            db_manager: DatabaseManager to get/set ML ops.
        """
        self.cfg = cfg
        self._loaded_model: Dict[str, Pipeline] = {}
        self.db_manager = db_manager
        self.task_mapping: Dict[Type[ValidInput], TypedFunctor] = {
            PredictionInput: TypedFunctor(
                function=self._get_prediction, output_type=PredictionOutput
            ),
            EvaluationInput: TypedFunctor(
                function=self._get_evaluate, output_type=EvaluationOutput
            ),
        }

    def get_model(self, name) -> Pipeline:
        """Get Pipeline from cache or load it.

        Args:
            name: Name of the model to load.

        Returns:
            TextClassificationPipeline
        """
        if name not in self._loaded_model:
            log.info("Loading model.")
            self._loaded_model[name] = load_text_classif_pipeline(name)
        return self._loaded_model[name]

    @overload
    def run_task(self, request: EvaluationInput) -> EvaluationOutput:
        ...

    @overload
    def run_task(self, request: PredictionInput) -> PredictionOutput:  # type: ignore
        ...

    def run_task(self, request):
        """Run a task or get it from database.

        Args:
            request: Input request.

        Returns:
            Request output
        """
        task_data = self.task_mapping[type(request)]
        cached = self.db_manager.get_cache(request, to=task_data.output_type)
        if cached:
            # Was already in the database.
            return cached
        else:
            log.info(f"Run {task_data.function} on model.")
            output = task_data.function(request)
            self.db_manager.store(output)
            return output

    def _get_evaluate(self, request: EvaluationInput) -> EvaluationOutput:
        """Runner function for evaluation.

        Args:
            request: Input request.

        Returns:
            Metrics for input.
        """
        ds = load_dataset(request.dataset.name)[request.dataset.split]  # type: ignore
        metric = load_metric(request.metric_name)
        text, labels = ds[request.dataset.text_column], ds[request.dataset.label_column]
        predictions = [
            self.run_task(self.db_manager.get_or_insert(PredictionInput(sentence=t))).prediction
            for t in text
        ]
        res = metric.compute(predictions=predictions, references=labels)
        return EvaluationOutput(value=res, extras={}, input_key_id=request.id)

    def _get_prediction(self, request) -> PredictionOutput:
        """Get prediction on a request.

        Args:
            request: Input data (sentence, label)

        Returns:
            Prediction from request.
        """
        model = self.get_model(self.cfg.model_name)
        out = [sc["score"] for sc in model(request.sentence)[0]]
        loss: Optional[float] = None
        if request.label:
            # If the user gives the label, we gives the loss to.
            _loss = F.nll_loss(
                torch.FloatTensor(out).unsqueeze(0),
                torch.LongTensor([request.label]),
            )
            loss = _loss.squeeze().item()
        return PredictionOutput(
            input_key_id=request.id,
            config_id=self.cfg.id,
            distribution=out,
            prediction=np.argmax(out),
            loss=loss,
        )
