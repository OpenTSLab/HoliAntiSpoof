import json
import re
from collections import Counter
from pathlib import Path
from json import JSONDecodeError

import hydra
from sklearn.metrics import accuracy_score, f1_score

# spoof_method mapping
detailed_spoof_method_mapping = {
    "text to speech synthesis": 0,
    "voice conversion": 1,
    "speech editing": 2,
    "cut and paste": 3,
    "codec resynthesis": 4,
    "vocoder resynthesis": 5,
    "failure": 6,
}

coarse_spoof_method_mapping = {
    "text to speech synthesis": 0,
    "voice conversion": 0,
    "speech editing": 1,
    "cut and paste": 2,
    "codec resynthesis": 3,
    "vocoder resynthesis": 4,
    "failure": 5,
}


@hydra.main(version_base=None, config_path="../configs/eval", config_name="eval_composite")
def main(config):

    is_coarse = config.get("is_coarse", False)

    output = []
    for pred_file in config.pred_files:
        output.extend(json.load(open(pred_file, 'r')))

    files = set()

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
            # Only process spoof_method for fake audio
            if ref["real_or_fake"] == "fake":
                gts.append(ref["spoof_method"])

                try:
                    pred = json.loads(item['pred'])
                    # Check if prediction is fake and contains spoof_method
                    preds.append(pred["spoof_method"])
                except (JSONDecodeError, KeyError):
                    # print(f"Error processing item {idx}: {item['pred']}")
                    preds.append("failure")
        else:
            ref = item["ref"]
            if ref.split(".")[0].lower() == "fake":
                gt = re.search(r"The spoofing method is (.+?)\.", ref).group(1)
                gts.append(gt)
                match = re.search(r"The spoofing method is (.+?)\.", item['pred'])
                if match:
                    pred = match.group(1)
                else:
                    pred = "failure"
                preds.append(pred)

    # Convert to labels
    gt_labels = []
    pred_labels = []

    spoof_method_mapping = coarse_spoof_method_mapping if is_coarse else detailed_spoof_method_mapping

    for gt, pred in zip(gts, preds):
        if gt in spoof_method_mapping:
            gt_labels.append(spoof_method_mapping[gt])
        else:
            print(f"{gt} not found in {spoof_method_mapping}")
            gt_labels.append(spoof_method_mapping["failure"])

        if pred in spoof_method_mapping:
            pred_labels.append(spoof_method_mapping[pred])
        else:
            pred_labels.append(spoof_method_mapping["failure"])

    # Calculate accuracy
    acc = accuracy_score(gt_labels, pred_labels)

    # Create output directory
    Path(config.output_file).parent.mkdir(parents=True, exist_ok=True)

    # Write results
    with open(config.output_file, 'a') as writer:
        print(f"Fake method (coarse: {is_coarse}): ", file=writer, flush=True)
        print(f"Accuracy: {acc:.4f}", file=writer, flush=True)
        print(f"Accuracy: {acc:.4f}")

        # Calculate F1 score (excluding failure class)
        valid_labels = [v for v in spoof_method_mapping.values() if v != spoof_method_mapping["failure"]]
        f1 = f1_score(gt_labels, pred_labels, average='macro', labels=valid_labels)
        print(f"F1 Score (macro): {f1:.4f}", file=writer, flush=True)
        print(f"F1 Score (macro): {f1:.4f}")

        # Calculate F1 score for each class
        for label_idx in spoof_method_mapping.values():
            if label_idx in gt_labels:
                methods = "; ".join([k for k, v in spoof_method_mapping.items() if v == label_idx])
                f1_per_class = f1_score(gt_labels, pred_labels, average=None, labels=[label_idx])[0]
                print(f"F1 Score ({methods}): {f1_per_class:.4f}", file=writer, flush=True)
                print(f"F1 Score ({methods}): {f1_per_class:.4f}")

        # # Count sample numbers
        # print(f"Total fake samples: {len(gt_labels)}", file=writer, flush=True)
        # print(f"Total fake samples: {len(gt_labels)}")

        # # Count numbers for each class
        # gt_counter = Counter(gts)
        # pred_counter = Counter(preds)

        # print("\nGround truth distribution:", file=writer, flush=True)
        # print("Ground truth distribution:")
        # for method, count in gt_counter.items():
        #     print(f"  {method}: {count}", file=writer, flush=True)
        #     print(f"  {method}: {count}")

        # print("\nPrediction distribution:", file=writer, flush=True)
        # print("Prediction distribution:")
        # for method, count in pred_counter.items():
        #     print(f"  {method}: {count}", file=writer, flush=True)
        #     print(f"  {method}: {count}")

        print("", file=writer, flush=True)


if __name__ == '__main__':
    main()
