import json
import random

from tqdm import tqdm

from evaluation.parsing_utils import SpoofingParser, validate_cot_format

data_list = [
    "./data_cot/partial_edit/train.json",
    "./data_cot/partial_spoof/train.json",
    "./data_cot/asvspoof2019/train.json",
    "./data_cot/sine/train.json",
    "./data_cot/wavefake/train.json",
    "./data_cot/codecfake_ntu/train.json",
    "./data_cot/vctk/train.json",
    "./data_cot/recent_tts/train.json",
    "./data_cot/ljspeech/train.json",
]

data = []
for data_file in data_list:
    data.extend(json.load(open(data_file)))
random.shuffle(data)

for item in tqdm(data):
    text = item["conversations"][1]["value"]
    assert validate_cot_format(text), f"text: \n{text} is not valid"

for item in data[:10]:
    text = item["conversations"][1]["value"]

    parser = SpoofingParser(data_format="cot")
    result = parser(text)
    print(f"text: \n{text}\nresult: \n{result}")
    print("-" * 100)
