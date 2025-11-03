from trl.trainer.grpo_config import GRPOConfig
from dataclasses import dataclass, field


@dataclass
class GRPOArguments(GRPOConfig):
    unused_items_for_generation: list[str] = field(default_factory=list)
