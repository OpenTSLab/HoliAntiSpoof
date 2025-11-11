from dataclasses import dataclass

import numpy as np

from transformers import AutoTokenizer
from transformers.trainer_utils import EvalPrediction

from qwenvl.data.data_qwen import IGNORE_INDEX
from evaluation.parsing_utils import SpoofingParser


@dataclass
class SpoofingAccuracy:

    data_format: str = "json"

    def __post_init__(self, ):
        self.text_parser = SpoofingParser(self.data_format)

    @property
    def name(self):
        return "real_fake_acc"

    def __call__(self, eval_pred: EvalPrediction, processing_class: AutoTokenizer):

        predictions, label_ids = eval_pred.predictions, eval_pred.label_ids
        correct = 0

        for prediction, label_id in zip(predictions, label_ids):
            mask_idx = np.where(label_id != IGNORE_INDEX)[0]
            label_text = processing_class.decode(label_id[mask_idx], skip_special_tokens=True).strip()
            prediction_text = processing_class.decode(prediction[mask_idx - 1], skip_special_tokens=True).strip()

            gt = self.text_parser(label_text)
            pred = self.text_parser(prediction_text)

            if gt["real_or_fake"] == pred["real_or_fake"]:
                correct += 1

        return {self.name: correct / len(predictions)}


class RewardWrapper:

    @property
    def name(self):
        return "reward"

    def __call__(self, eval_pred: EvalPrediction, processing_class: AutoTokenizer):
        return {"reward": eval_pred.losses.mean().item()}
