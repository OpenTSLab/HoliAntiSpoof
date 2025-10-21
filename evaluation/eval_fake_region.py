import json
import re
from pathlib import Path
from json import JSONDecodeError

import hydra
import pandas as pd
from evaluation.metrics import calculate_segment_f1


@hydra.main(version_base=None, config_path="../configs", config_name="eval_fake_region")
def main(config):

    is_partial_only = config.get("is_partial_only", True)

    output = []
    for pred_file in config.pred_files:
        output.extend(json.load(open(pred_file, 'r')))

    files = set()

    # Read audio duration information
    dfs = []
    for duration_file in config.duration_files:
        df = pd.read_csv(duration_file, sep='\t')
        dfs.append(df)
    df = pd.concat(dfs)
    duration_dict = dict(zip(df['audio_id'], df['duration']))

    # Store ground truth and predicted fake regions
    gt_segments = {}
    pred_segments = {}

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

        if is_json:
            ref = json.loads(item['ref'])
            gt_fake_region_raw = ref["fake_region"] if "fake_region" in ref else None
            gt_real_or_fake = ref["real_or_fake"]
        else:
            ref = item['ref']
            gt_real_or_fake = ref.split(".")[0].lower()
            gt_fake_region_raw = re.search(r"The fake region is (.+?)(?=\.\s|\.$)",
                                           ref).group(1) if gt_real_or_fake == "fake" else None

        # Process ground truth labels
        gt_fake_region = []
        if gt_real_or_fake == "fake":
            if gt_fake_region_raw == "all" or gt_fake_region_raw == "the whole clip":
                # If the entire audio is fake, use the full duration
                gt_fake_region = [[0, duration_dict[file_id]]]
            else:
                if is_json:
                    gt_fake_region = gt_fake_region_raw
                else:
                    gt_fake_region = eval(gt_fake_region_raw)

        if is_partial_only:
            if gt_real_or_fake == "real":
                continue
            else:
                if gt_fake_region_raw == "all":
                    continue

        gt_segments[file_id] = gt_fake_region

        if is_json:
            try:
                pred = json.loads(item['pred'])
                # Process prediction labels
                pred_fake_region = []
                if pred["real_or_fake"] == "fake":
                    pred_fake_region_raw = pred["fake_region"]
                    if pred_fake_region_raw == "all":
                        # If predicting the entire audio is fake, use the full duration
                        pred_fake_region = [[0, duration_dict[file_id]]]
                    elif isinstance(pred_fake_region_raw, list):
                        # Filter out invalid regions (start time >= end time)
                        pred_fake_region = [
                            region for region in pred_fake_region_raw if len(region) == 2 and region[0] < region[1]
                        ]
                    # If real_or_fake == "real", pred_fake_region remains an empty list

                # Store results (including both real and fake audio)
                pred_segments[file_id] = pred_fake_region

            except (JSONDecodeError, KeyError) as e:
                print(f"Error processing item {idx}: {e}")
                pred_segments[file_id] = []
        else:
            pred_real_or_fake = item['pred'].split(".")[0].lower()
            match = re.search(r"The fake region is (.+?)(?=\.\s|\.$)", item['pred'])
            if match:
                pred_fake_region_raw = match.group(1)
            else:
                pred_fake_region_raw = ""

            pred_fake_region = []
            if pred_real_or_fake == "fake":
                if pred_fake_region_raw == "the whole clip":
                    pred_fake_region = [[0, duration_dict[file_id]]]
                else:
                    try:
                        pred_fake_region_raw = eval(pred_fake_region_raw)
                        pred_fake_region = [
                            region for region in pred_fake_region_raw if len(region) == 2 and region[0] < region[1]
                        ]
                    except SyntaxError:
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
