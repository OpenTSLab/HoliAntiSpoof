import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from qwenvl.train.utils import load_config_from_cli
from evaluation.parsing_utils import init_parser


def main():

    config = load_config_from_cli()
    files = set()
    output = []
    for dataset in config.pred_durations:
        output.extend(json.load(open(dataset.pred, 'r')))

    fake_real_mapping = {
        "real": 0,
        "fake": 1,
        "failure": 2,
    }

    text_parser = init_parser(output)

    gts, preds = [], []
    for idx, item in enumerate(output):
        if "audio" in item:
            file_id = Path(item["audio"]).name
            if file_id in files:
                continue
            files.add(file_id)

        gt = text_parser(item["ref"])
        pred = text_parser(item["pred"])
        gts.append(gt["real_or_fake"])
        preds.append(pred["real_or_fake"])

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
