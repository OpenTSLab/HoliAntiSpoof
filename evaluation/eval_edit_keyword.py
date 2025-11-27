import json
import re
from pathlib import Path
from json import JSONDecodeError

from pprint import pprint
import hydra
from tqdm import tqdm

from evaluation.parsing_utils import SpoofingParser


@hydra.main(version_base=None, config_path="../configs/eval", config_name="eval_composite")
def main(config):
    files = set()
    output = []
    for dataset in config.pred_durations:
        output.extend(json.load(open(dataset.pred, 'r')))

    file_to_keywords = {}
    for editing_keywords_file in config.editing_keywords_files:
        with open(editing_keywords_file, "r") as f:
            for item in tqdm(json.load(f)):
                if "keywords" not in item:
                    continue
                file_to_keywords[item["audio"]] = item["keywords"].split()

    keyword_matched = 0
    total_num = 0

    data_format = "json"
    try:
        json.loads(output[0]['ref'])
    except JSONDecodeError:
        data_format = "cot"

    text_parser = SpoofingParser(data_format)

    for idx, item in enumerate(output):
        if "audio" in item:
            file_id = Path(item["audio"]).name
            if file_id in files:
                continue
            files.add(file_id)

        if item["audio"] not in file_to_keywords:
            continue

        gt_data = text_parser(item["ref"])

        if gt_data["semantic_influence"] is None:
            continue

        pred_data = text_parser(item["pred"])
        pred_semantic_influence = pred_data["semantic_influence"]
        if pred_semantic_influence is None:
            pred_semantic_influence = ""

        keywords = file_to_keywords[item["audio"]]
        total_num += len(keywords)
        item_matched = 0
        for keyword in keywords:
            if keyword in pred_semantic_influence:
                item_matched += 1
        keyword_matched += item_matched

    acc = keyword_matched / total_num

    Path(config.output_file).parent.mkdir(parents=True, exist_ok=True)
    writer = open(config.output_file, 'a')
    print(f"Keyword covering rate: {acc:.4f}", file=writer, flush=True)
    print(f"Keyword covering rate: {acc:.4f}")
    print("", file=writer, flush=True)


if __name__ == '__main__':
    main()
