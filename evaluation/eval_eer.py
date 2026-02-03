import json
from typing import Any
from pathlib import Path

import numpy as np
import torch
import hydra

from evaluation.parsing_utils import init_parser


def compute_det_curve(target_scores: np.ndarray, nontarget_scores: np.ndarray):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - \
        (np.arange(1, n_scores + 1) - tar_trial_sums)

    # false rejection rates
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    # Thresholds are the sorted scores
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds


def compute_eer(target_scores: np.ndarray, nontarget_scores: np.ndarray):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def calc_score(item: dict[str, Any], score_name: str):
    if score_name == "logit":
        return item["real_logit"]
    elif score_name == "binary_prob":
        logits = torch.tensor([item["real_logit"], item["fake_logit"]])
        binary_probs = torch.softmax(logits, dim=-1)
        return binary_probs[0].item()
    elif score_name == "vocab_prob_orig":
        return item["real_prob"]
    elif score_name == "vocab_prob_normed":
        return item["real_prob"] / (item["real_prob"] + item["fake_prob"])


@hydra.main(version_base=None, config_path="../configs/eval", config_name="eval_composite")
def main(config):
    # infer_result: str, score_name: str = "binary_prob"
    score_name = config.spoof_score_name
    assert score_name in ("logit", "binary_prob", "vocab_prob_orig", "vocab_prob_normed")
    output = []
    for dataset in config.pred_durations:
        output.extend(json.load(open(dataset.pred, 'r')))

    text_parser = init_parser(output)

    files = set()
    target_scores: list[float] = []
    nontarget_scores: list[float] = []
    for idx, item in enumerate(output):
        if "audio" in item:
            file_id = Path(item["audio"]).name
            if file_id in files:
                continue
            files.add(file_id)

        gt = text_parser(item["ref"])
        score = calc_score(item, score_name)
        if gt["real_or_fake"] == "real":
            target_scores.append(score)
        else:
            nontarget_scores.append(score)

    target_scores = np.array(target_scores)
    nontarget_scores = np.array(nontarget_scores)

    eer, threshold = compute_eer(target_scores, nontarget_scores)

    Path(config.output_file).parent.mkdir(parents=True, exist_ok=True)
    writer = open(config.output_file, 'a')
    print(f"EER: {eer:.4f}", file=writer, flush=True)
    print("", file=writer, flush=True)
    writer.close()

    print(f"EER: {eer:.4f}")
    print(f"Threshold: {threshold:.4f}")


if __name__ == "__main__":
    main()
