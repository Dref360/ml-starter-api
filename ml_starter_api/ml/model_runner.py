from typing import Optional, Dict

import numpy as np
import structlog
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    Pipeline,
    AutoTokenizer,
    TextClassificationPipeline,
)

from ml_starter_api.config import Config
from ml_starter_api.database.manager import DatabaseManager
from ml_starter_api.models.predictions import PredictionOutput, PredictionInput

log = structlog.get_logger("ModelRunner")


class ModelRunner:
    def __init__(self, cfg: Config, db_manager: DatabaseManager):
        self.cfg = cfg
        self._loaded_model: Dict[str, Pipeline] = {}
        self.db_manager = db_manager

    def get_model(self, name):
        if name not in self._loaded_model:
            log.info("Loading model.")
            self._loaded_model[name] = self.load_text_classif_pipeline(name)
        return self._loaded_model[name]

    def load_text_classif_pipeline(self, checkpoint_path: str) -> Pipeline:
        use_cuda = torch.cuda.is_available()
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=False)
        device = 0 if use_cuda else -1

        return TextClassificationPipeline(
            model=model, tokenizer=tokenizer, device=device, return_all_scores=True
        )

    def run_prediction(self, request: PredictionInput) -> PredictionOutput:
        if cached := self.db_manager.get_cache(request, to=PredictionOutput):
            return cached
        else:
            log.info("Run prediction on model.")
            output = self._get_prediction(request)
            self.db_manager.store(output)
            return output

    def _get_prediction(self, request) -> PredictionOutput:
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
