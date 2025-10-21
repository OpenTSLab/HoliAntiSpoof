import json
import re
from pathlib import Path
from json import JSONDecodeError

from pprint import pprint
import hydra
from tqdm import tqdm


@hydra.main(version_base=None, config_path="../configs", config_name="eval_composite")
def main(config):
    files = set()
    output = []
    for pred_file in config.pred_files:
        output.extend(json.load(open(pred_file, 'r')))

    file_to_keywords = {}
    with open("data/partial_edit/test.json", "r") as f:
        for item in tqdm(json.load(f)):
            if "keywords" not in item:
                continue
            file_to_keywords[item["audio"]] = item["keywords"].split()

    keyword_matched = 0
    total_num = 0

    is_json = True
    try:
        json.loads(output[0]['ref'])
    except JSONDecodeError:
        is_json = False

    for idx, item in enumerate(output):
        if "audio" in item:
            file_id = Path(item["audio"]).name
            if file_id in files:
                continue
            files.add(file_id)

        if item["audio"] not in file_to_keywords:
            continue

        if is_json:
            if "semantic_influence" not in json.loads(item["ref"]):
                continue
        else:
            if "The influence is: " not in item["ref"]:
                continue

        keywords = file_to_keywords[item["audio"]]
        total_num += len(keywords)
        if is_json:
            try:
                pred = json.loads(item['pred'])
                assert "semantic_influence" in pred
                pred_semantic_influence = pred["semantic_influence"]
            except (JSONDecodeError, AssertionError):
                pred_semantic_influence = ""
        else:
            match = re.search(r"The influence is: (.+)", item["pred"])
            if match:
                pred_semantic_influence = match.group(1)
            else:
                pred_semantic_influence = ""

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
