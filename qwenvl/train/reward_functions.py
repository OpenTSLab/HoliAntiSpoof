import json
from typing import Any

import numpy as np

from evaluation.parsing_utils import SpoofingParser


def calculate_overlap_ratio(segments_a, segments_b):
    """
    Compute overlap ratio (IoU) between two lists of time segments in a fully vectorized way.
    Each segment list is a list of [start, end] pairs.

    Args:
        segments_a (list[list[float]]): First list of segments.
        segments_b (list[list[float]]): Second list of segments.

    Returns:
        float: Intersection over Union (IoU) of the two segment sets.
    """
    a = np.array(segments_a, dtype=np.float32)
    b = np.array(segments_b, dtype=np.float32)

    total_len_a = np.sum(np.maximum(0, a[:, 1] - a[:, 0]))
    total_len_b = np.sum(np.maximum(0, b[:, 1] - b[:, 0]))

    start_max = np.maximum(a[:, None, 0], b[None, :, 0])
    end_min = np.minimum(a[:, None, 1], b[None, :, 1])

    overlaps = np.maximum(0, end_min - start_max)
    total_overlap = np.sum(overlaps)

    union_len = total_len_a + total_len_b - total_overlap
    return (total_overlap / union_len).item() if union_len > 0 else 0.0


class SpoofReward:
    """
    Callable reward function for JSON-formatted spoofing data.
    """
    def __init__(
        self,
        data_format: str = "json",
        weights: dict[str, float] = {
            "format": 0.2,
            "real_fake": 0.5,
            "spoof_method": 0.05,
            "fake_region": 0.05,
            "keywords": 0.2
        }
    ) -> None:
        self.weights = weights
        self.__name__ = type(self).__name__
        self.parser = SpoofingParser(data_format=data_format)

    def keywords_reward(self, completion: dict, keywords: str | None) -> float:
        if keywords is None:
            return 1.0
        if completion["semantic_influence"] is None:
            return 0.0
        reward = 0.0
        words = keywords.split()
        num = len(words)
        for w in words:
            if w in completion["semantic_influence"]:
                reward += 1.0 / num
        return reward

    def fake_region_reward(self, completion: dict, label: dict) -> float:
        if label["fake_region"] is None:  # no ground truth fake regions
            return completion["fake_region"] is None
        if completion["fake_region"] is None:  # ground truth exists, no prediction
            return 0.0
        if label["fake_region"] == "all":
            return 1.0 if completion["fake_region"] == "all" else 0.0
        try:
            return calculate_overlap_ratio(label["fake_region"], completion["fake_region"])
        except Exception:
            return 0.0

    def real_fake_reward(self, completion: dict, label: dict) -> float:
        if completion["real_or_fake"] == label["real_or_fake"]:
            return 1.0
        return 0.0

    def spoof_method_reward(self, completion: dict, label: dict) -> float:
        if completion["spoof_method"] == label["spoof_method"]:
            return 1.0
        return 0.0

    def __call__(
        self,
        completions: list[str],
        labels_text: list[str],
        keywords: list[str | None],
        **kwargs: Any,
    ) -> list[float]:
        rewards: list[float] = []
        for completion_text, label_text, keywords_item in zip(completions, labels_text, keywords):
            reward = 0.0
            label = self.parser(label_text)
            completion = self.parser(completion_text)

            if not completion["format_success"]:
                reward = 0.0
            else:
                format_reward = 1.0
                real_fake_reward = self.real_fake_reward(completion, label)
                spoof_method_reward = self.spoof_method_reward(completion, label)
                fake_region_reward = self.fake_region_reward(completion, label)
                keywords_reward = self.keywords_reward(completion, keywords_item)
                reward = (
                    format_reward * self.weights["format"] + \
                    real_fake_reward * self.weights["real_fake"] + \
                    spoof_method_reward * self.weights["spoof_method"] +
                    fake_region_reward * self.weights["fake_region"] + \
                    keywords_reward * self.weights["keywords"]
                )

            rewards.append(float(reward))

        # print(rewards)

        return rewards
