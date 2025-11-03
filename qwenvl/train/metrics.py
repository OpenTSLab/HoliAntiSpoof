import json
from dataclasses import dataclass

import numpy as np

from transformers import AutoTokenizer
from transformers.trainer_utils import EvalPrediction

from qwenvl.data.data_qwen import IGNORE_INDEX


@dataclass
class SpoofingAccuracy:

    data_format: str = "json"

    def __call__(self, eval_pred: EvalPrediction, processing_class: AutoTokenizer):

        predictions, label_ids = eval_pred.predictions, eval_pred.label_ids
        correct = 0

        for prediction, label_id in zip(predictions, label_ids):
            mask_idx = np.where(label_id != IGNORE_INDEX)[0]
            label_text = processing_class.decode(label_id[mask_idx], skip_special_tokens=True).strip()
            prediction_text = processing_class.decode(prediction[mask_idx - 1], skip_special_tokens=True).strip()

            if self.data_format == "json":
                label_json = json.loads(label_text)
                try:
                    prediction_json = json.loads(prediction_text)
                    if label_json["real_or_fake"] == prediction_json["real_or_fake"]:
                        correct += 1
                except (json.JSONDecodeError, KeyError):
                    pass
            else:
                real_fake_label = label_text.split(".")[0]
                real_fake_prediction = prediction_text.split(".")[0]
                if real_fake_label == real_fake_prediction:
                    correct += 1

        return {"real_fake_acc": correct / len(predictions)}


class RewardWrapper:
    def __call__(self, eval_pred: EvalPrediction, processing_class: AutoTokenizer):
        return {"reward": eval_pred.losses.mean().item()}
