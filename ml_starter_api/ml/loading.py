import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline,
    Pipeline,
)


def load_text_classif_pipeline(checkpoint_path: str) -> Pipeline:
    use_cuda = torch.cuda.is_available()
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=False)
    device = 0 if use_cuda else -1

    return TextClassificationPipeline(
        model=model, tokenizer=tokenizer, device=device, return_all_scores=True
    )
