import json
from pathlib import Path
from json import JSONDecodeError

import hydra
from sklearn.metrics import accuracy_score, f1_score


@hydra.main(version_base=None, config_path="../configs/eval", config_name="eval_composite")
def main(config):
    files = set()
    output = []
    for pred_file in config.pred_files:
        output.extend(json.load(open(pred_file, 'r')))

    fake_real_mapping = {
        "real": 0,
        "fake": 1,
        "failure": 2,
    }

    is_json = True
    try:
        json.loads(output[0]['ref'])
    except JSONDecodeError:
        is_json = False

    gts, preds = [], []
    for idx, item in enumerate(output):
        if "audio" in item:
            file_id = Path(item["audio"]).name
            if file_id in files:
                continue
            files.add(file_id)
        if is_json:
            ref = json.loads(item['ref'])
            gts.append(ref["real_or_fake"])

            try:
                pred = json.loads(item['pred'])
                assert "real_or_fake" in pred
                assert pred["real_or_fake"] in ["real", "fake"]
                preds.append(pred["real_or_fake"])
            except (JSONDecodeError, AssertionError):
                if '"real_or_fake": "fake"' in item["pred"]:
                    preds.append("fake")
                elif '"real_or_fake": "real"' in item["pred"]:
                    preds.append("real")
                else:
                    print(item["pred"])
                    preds.append("failure")
        else:
            gts.append(item["ref"].split(".")[0].lower())
            preds.append(item['pred'].split(".")[0].lower())
    gt_labels = list(map(fake_real_mapping.get, gts))
    pred_labels = []
    for pred in preds:
        if pred in fake_real_mapping:
            pred_labels.append(fake_real_mapping[pred])
        else:
            pred_labels.append(fake_real_mapping["failure"])

    acc = accuracy_score(gt_labels, pred_labels)

    Path(config.output_file).parent.mkdir(parents=True, exist_ok=True)
    writer = open(config.output_file, 'a')
    print("Real or fake: ", file=writer, flush=True)
    print(f"Accuracy: {acc:.4f}", file=writer, flush=True)
    print(f"Accuracy: {acc:.4f}")

    f1 = f1_score(gt_labels, pred_labels, average='macro', labels=[0, 1])
    print(f"F1 Score (macro): {f1:.4f}", file=writer, flush=True)
    print(f"F1 Score (macro): {f1:.4f}")

    print("", file=writer, flush=True)


if __name__ == '__main__':
    main()
