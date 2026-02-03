#! /usr/bin/env python3

import argparse
import json
from pathlib import Path
import librosa
import pandas as pd
from tqdm import tqdm
from pypeln import process as pr


def extract_duration(row):
    if isinstance(row, tuple):
        idx, item = row
        aid = item["audio_id"]
        if "file_name" in item:
            fname = item["file_name"]
        elif "hdf5_path" in item:
            fname = item["hdf5_path"]
    elif isinstance(row, dict):
        aid = row["audio_id"]
        fname = row["audio"]

    if fname.startswith("s3://"):
        fname = fname.replace("s3://", "/mnt/brainllm_s3/", 1)
    try:
        duration = librosa.core.get_duration(path=fname)
    except Exception as e:
        print("Encounter exception: ", e)
        duration = -1.0

    return aid, duration


parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", type=str, required=True)
parser.add_argument("--output", "-o", type=str, required=True)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--sample_rate", type=int)

args = parser.parse_args()

file_items = []

all_items = json.load(open(args.input))
for item in tqdm(all_items):
    audio_path = item["audio"]
    audio_id = Path(audio_path).name
    file_items.append({"audio_id": audio_id, "audio": audio_path})
file_num = len(file_items)

sample_rate = args.sample_rate

output_data = []
with tqdm(total=file_num, ascii=True) as pbar:
    for aid, duration in pr.map(extract_duration, file_items, workers=args.num_workers, maxsize=4):
        output_data.append({"audio_id": aid, "duration": duration})
        pbar.update()

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(output_data).to_csv(args.output, sep="\t", index=False, float_format="%.3f")
