from dataclasses import dataclass, field

from trl.trainer.grpo_config import GRPOConfig

from qwenvl.train.argument import TrainingArguments


@dataclass
class GRPOArguments(TrainingArguments, GRPOConfig):
    use_liger_loss: bool = field(
        default=True,
        metadata={"help": "Whether to use the Liger GRPO loss."},
    )
