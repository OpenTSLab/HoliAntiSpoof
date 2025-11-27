from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.dpo_config import DPOConfig
from dataclasses import dataclass, field


@dataclass
class UnusedItemsMixin:
    unused_items_for_generation: list[str] = field(default_factory=list)


@dataclass
class GRPOArguments(GRPOConfig, UnusedItemsMixin):
    pass


@dataclass
class DPOArguments(DPOConfig, UnusedItemsMixin):
    pass
