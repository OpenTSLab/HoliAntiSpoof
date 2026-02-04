import json
from pathlib import Path

import pandas as pd

from evaluation.metrics import calculate_segment_f1
from evaluation.parsing_utils import init_parser
from qwenvl.train.utils import load_config_from_cli


def main():
    config = load_config_from_cli()

    is_partial_only = config.get("is_partial_only", True)

    output = []
    pred_files = []
    durations: dict[str, dict[str, float]] = {}
    for dataset in config.pred_durations:
        dataset_data = json.load(open(dataset.pred, 'r'))
        output.extend(dataset_data)
        if "duration" in dataset:
            df = pd.read_csv(dataset.duration, sep="\t")
            durations[dataset.pred] = dict(zip(df['audio_id'], df['duration']))

        pred_files.extend([dataset.pred] * len(dataset_data))
    files = set()

    # Store ground truth and predicted fake regions
    gt_segments = {}
    pred_segments = {}

    text_parser = init_parser(output)

    for idx, item in enumerate(output):
        if "audio" in item:
            file_id = Path(item["audio"]).name
            if file_id in files:
                continue
            files.add(file_id)

        gt_data = text_parser(item["ref"])

        if is_partial_only:  # skip real and partial spoof data
            if gt_data["real_or_fake"] == "real":
                continue
            else:
                if gt_data["fake_region"] == "all":
                    continue

        # Process ground truth labels
        gt_fake_region = []
        if gt_data["real_or_fake"] == "fake":
            gt_fake_region = gt_data["fake_region"]
            if gt_fake_region == "all":
                gt_fake_region = [[0, durations[pred_files[idx]][file_id]]]

        gt_segments[file_id] = gt_fake_region

        pred_data = text_parser(item["pred"])
        pred_fake_region = pred_data["fake_region"]
        if pred_fake_region == "all":
            pred_fake_region = [[0, durations[pred_files[idx]][file_id]]]
        elif pred_fake_region is None:
            pred_fake_region = []

        pred_segments[file_id] = pred_fake_region

    # Ensure both dictionaries have the same keys
    common_files = set(gt_segments.keys()) & set(pred_segments.keys())
    gt_segments_filtered = {k: gt_segments[k] for k in common_files}
    pred_segments_filtered = {k: pred_segments[k] for k in common_files}

    if not gt_segments_filtered:
        print("No valid data found!")
        return

    print(f"Processing {len(gt_segments_filtered)} files...")

    # Count different types of data
    real_count = sum(1 for regions in gt_segments_filtered.values() if not regions)
    fake_count = len(gt_segments_filtered) - real_count
    print(f"Real audio files: {real_count}")
    print(f"Fake audio files: {fake_count}")

    # Calculate segment F1 metrics

    metrics = calculate_segment_f1(gt_segments_filtered, pred_segments_filtered)

    # Create output directory
    Path(config.output_file).parent.mkdir(parents=True, exist_ok=True)

    # Write results
    with open(config.output_file, 'a') as writer:
        print(f"Segment F1 Score: {metrics['f_measure']['f_measure']:.4f}", file=writer, flush=True)
        print(f"Segment Precision: {metrics['f_measure']['precision']:.4f}", file=writer, flush=True)
        print(f"Segment Recall: {metrics['f_measure']['recall']:.4f}", file=writer, flush=True)

        # Print to console
        print(f"Segment F1 Score: {metrics['f_measure']['f_measure']:.4f}")
        print(f"Segment Precision: {metrics['f_measure']['precision']:.4f}")
        print(f"Segment Recall: {metrics['f_measure']['recall']:.4f}")


if __name__ == '__main__':
    main()
