#!/usr/bin/env python3
"""
Convert data_json files to in-context learning format.
Extracts conversations and keywords (if present) from original data.
"""

import json
from pathlib import Path
from typing import Any

import fire
import numpy as np
from tqdm import tqdm

from evaluation.parsing_utils import SpoofingParser


def sample_few_shot_examples(
    data: list[dict[str, Any]],
    text_parser: SpoofingParser,
    num_real_samples: int,
    num_fake_samples: int,
    exclude_idx: int | None = None
) -> list[dict[str, Any]]:
    idxs = np.arange(len(data))
    np.random.shuffle(idxs)

    samples = []
    num_sampled_real, num_sampled_fake = 0, 0
    for idx in idxs:
        if exclude_idx is not None and idx == exclude_idx:
            continue
        item = data[idx]
        gt = text_parser(item["conversations"][1]["value"])
        if gt["real_or_fake"] == "real":
            if num_sampled_real < num_real_samples:
                samples.append(item)
                num_sampled_real += 1
            else:
                continue
        else:
            if num_sampled_fake < num_fake_samples:
                samples.append(item)
                num_sampled_fake += 1
            else:
                continue

        if num_sampled_real == num_real_samples and num_sampled_fake == num_fake_samples:
            break

    return samples


def convert_item(
    item: dict[str, Any],
    sampled_prompts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert a single data item to in-context learning format."""
    # {
    #     "conversations": [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "audio", "audio": "<example1>"},
    #                 {"type": "text", "text": "<answer1>"},
    #                 {"type": "audio", "audio": "<example2>"},
    #                 {"type": "text", "text": "<answer2>"},
    #                 // ...
    #                 {"type": "text", "text": "<question>"},
    #             ],
    #         },
    #         {
    #             "role": "assistant",
    #             "content": [
    #                 {"type": "text", "text": "answer"},
    #             ]
    #         }
    #     ]
    # }
    result = {"conversations": [{"role": "user", "content": []}]}
    user_content = result["conversations"][0]["content"]
    for prompt in sampled_prompts:
        answer = prompt["conversations"][1]["value"]
        json_answer = json.loads(answer)
        json_answer.pop("semantic_influence", None)
        answer = json.dumps(json_answer, ensure_ascii=False)
        user_content.append({
            "type": "audio",
            "audio": prompt["audio"],
        })
        user_content.append({
            "type": "text",
            "text": answer,
        })
    user_content.append({"type": "text", "text": item["conversations"][0]["value"].replace("<image>\n", "")})
    user_content.append({"type": "audio", "audio": item["audio"]})

    result["conversations"].append({
        "role": "assistant",
        "content": [{
            "type": "text",
            "text": item["conversations"][1]["value"]
        }]
    })

    # Include keywords if present in original data
    if "keywords" in item:
        result["keywords"] = item["keywords"]

    return result


def convert_file(
    input_path: Path,
    output_path: Path,
    num_samples: int,
    prompt_path: str = None,
    few_shot_path: str = None,
    is_test: bool = False
) -> None:
    """Convert a single JSON file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # If few_shot_path is provided, use it directly without sampling
    if few_shot_path:
        with open(few_shot_path, 'r', encoding='utf-8') as f:
            fixed_sampled_prompts = json.load(f)
        print(f"Using fixed few-shot prompts from {few_shot_path} ({len(fixed_sampled_prompts)} examples)")

        converted_data = []
        for idx, item in enumerate(tqdm(data)):
            converted_item = convert_item(item, fixed_sampled_prompts)
            converted_data.append(converted_item)
    else:
        # Original sampling logic
        if not prompt_path:
            prompt_path = input_path
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_data = json.load(f)

        data_format = "json"
        try:
            json.loads(prompt_data[0]["conversations"][1]["value"])
        except json.JSONDecodeError:
            data_format = "cot"

        text_parser = SpoofingParser(data_format)

        num_total_real, num_total_fake = 0, 0
        for item in prompt_data:
            gt = text_parser(item["conversations"][1]["value"])
            if gt["real_or_fake"] == "real":
                num_total_real += 1
            else:
                num_total_fake += 1

        if num_total_fake > 0 and num_total_real > 0:
            num_real_samples = num_samples // 2
            num_fake_samples = num_samples - num_real_samples
        elif num_total_fake > 0:
            num_real_samples, num_fake_samples = 0, num_samples
        elif num_total_real > 0:
            num_real_samples, num_fake_samples = num_samples, 0

        # For test data, get fixed few-shot examples once and reuse for all items
        if is_test and prompt_path != input_path:
            # Use external prompt data, get fixed few-shot examples once
            fixed_sampled_prompts = sample_few_shot_examples(
                prompt_data, text_parser, num_real_samples, num_fake_samples, exclude_idx=None
            )
        else:
            fixed_sampled_prompts = None

        converted_data = []
        for idx, item in enumerate(tqdm(data)):
            if fixed_sampled_prompts is not None:
                # Test mode with external prompt: use fixed few-shot examples
                sampled_prompts = fixed_sampled_prompts
            elif prompt_path == input_path:
                # Same file: need to exclude current item
                sampled_prompts = sample_few_shot_examples(
                    prompt_data, text_parser, num_real_samples, num_fake_samples, exclude_idx=idx
                )
            else:
                # Training mode with external prompt: random sample each time
                sampled_prompts = sample_few_shot_examples(
                    prompt_data, text_parser, num_real_samples, num_fake_samples, exclude_idx=None
                )

            converted_item = convert_item(item, sampled_prompts)
            converted_data.append(converted_item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(converted_data)} items: {input_path} -> {output_path}")


def main(
    input_path: str,
    num_samples: int,
    output_path: str = None,
    prompt_path: str = None,
    few_shot_path: str = None,
    recursive: bool = False,
    is_test: bool = False
):
    input_path = Path(input_path)

    if input_path.is_file():
        # Single file conversion
        if output_path:
            output_path = Path(output_path)
        else:
            output_path = input_path.parent / f"{input_path.stem}_icl{input_path.suffix}"
        convert_file(input_path, output_path, num_samples, prompt_path, few_shot_path=few_shot_path, is_test=is_test)

    elif input_path.is_dir():
        # Directory conversion
        pattern = "**/*.json" if recursive else "*.json"
        json_files = list(input_path.glob(pattern))
        assert output_path is None

        for json_file in json_files:
            output_json = json_file.parent / f"{json_file.stem}_icl{json_file.suffix}"
            convert_file(json_file, output_json, num_samples, few_shot_path=few_shot_path, is_test=is_test)

    else:
        print(f"Error: {input_path} is not a valid file or directory")


if __name__ == "__main__":
    fire.Fire(main)
