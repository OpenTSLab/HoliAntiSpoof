"""
{
    "annotation": [
        {
            "path": "/path/to/location",
            "text": "xxx",
            "task": "xxxxxx",
        }
    ]
}

transformed to

[
  {
    "audio": "xxx",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nYour Prompt Here."
      },
      {
        "from": "gpt",
        "value": "xxx"
      }
    ]
  },
  // ...
]
"""

import json
import random
from pathlib import Path

import fire

random.seed(42)


class Runner:
    def transform(self, input: str, output: str):

        salmonn_prompts = json.load(
            open("/mnt/shared-storage-user/xuxuenan/workspace/duplex_training/"
                 "salmonn_prompts/train_prompt.json")
        )
        data = json.load(open(input, "r"))
        transformed_data = []
        for item in data["annotation"]:
            audio_path = item["path"]
            text = item["text"]
            task = item["task"]
            instruction = random.choice(salmonn_prompts[task])
            instruction = instruction.replace("<Speech><SpeechHere></Speech> ", "")
            qwen_conv = [{"from": "human", "value": f"<image>\n{instruction}"}, {"from": "gpt", "value": text}]
            qwen_item = {"audio": audio_path, "conversations": qwen_conv}
            if "spoof_words" in item:
                qwen_item["keywords"] = item["spoof_words"]
            transformed_data.append(qwen_item)

        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(transformed_data, f, indent=2)


if __name__ == "__main__":
    fire.Fire(Runner)
