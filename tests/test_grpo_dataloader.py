# from trl.trainer.grpo_trainer import (
#     RepeatSampler,
# )

from collections.abc import Sized
from typing import Optional
import fire

import torch
from torch.utils.data import DataLoader, Sampler
from accelerate import Accelerator


class RepeatSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the dataset.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(
    ...     ["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4
    ... )
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """
    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            self.generator = torch.Generator()  # Create a local random generator
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        else:
            indexes = list(range(self.num_samples))

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i:i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class Dataset:
    def __len__(self, ):
        return 18

    def __getitem__(self, index):
        return {"id": index}


def collate_fn(instances):
    return {"id": [instance["id"] for instance in instances]}


def main(stage: str = "train"):

    accelerator = Accelerator()
    per_device_batch_size = 8
    grad_accum_steps = 2
    num_processes = accelerator.num_processes
    batch_size = per_device_batch_size

    num_generations = 8  # = G in the GRPO paper
    dataset = Dataset()

    steps_per_generation = grad_accum_steps
    generation_batch_size = per_device_batch_size * num_processes * steps_per_generation

    num_iterations = 1

    generate_every = steps_per_generation * num_iterations

    print(f"generate_every: {generate_every}")

    if stage == "train":

        sampler = RepeatSampler(
            data_source=dataset,
            mini_repeat_count=num_generations,
            batch_size=generation_batch_size // num_generations,  # 4
            repeat_count=num_iterations * steps_per_generation,  # 2
            # steps_per_generation: gradient accumulation steps
            shuffle=False,
        )

        dataloader_params = {
            "batch_size": batch_size * steps_per_generation,  # 32
            "num_workers": 8,
            "sampler": sampler,
            "collate_fn": collate_fn,
        }
        dataloader = DataLoader(dataset, **dataloader_params)
        """
        Output:

        generate_every: 2
        Step 0
        RANK 0 ids: [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        RANK 1 ids: [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3] 

        Step 1
        RANK 0 ids: [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        RANK 1 ids: [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]

        Step 2
        RANK 0 ids: [4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5]
        RANK 1 ids: [6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7]

        Step 3
        RANK 0 ids: [4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5]
        RANK 1 ids: [6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7]


        Explanation:
        Step 0 (rank 0): 
        - generate [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] (note the batch size for rollout/generation, not `per_device_batch_size`) 
          and store into buffer: [[0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]]
        - retrieve [0, 0, 0, 0, 0, 0, 0, 0] (this is `per_device_batch_size`), calculate reward, advantage (sync with other ranks), optimize
        Step 1 (rank 0):
        - retrieve [1, 1, 1, 1, 1, 1, 1, 1], calculate reward, advantage (sync with other ranks), optimize
        Step 2 (rank 0):
        - generate [4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5] 
          and store into buffer: [[4, 4, 4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5, 5, 5]]
        - retrieve [4, 4, 4, 4, 4, 4, 4, 4], calculate reward, advantage (sync with other ranks), optimize
        Step 3 (rank 0):
        - retrieve [5, 5, 5, 5, 5, 5, 5, 5], calculate reward, advantage (sync with other ranks), optimize
        ...
        Same for other ranks. No sync requirements between processes.
        """

    elif stage == "eval":
        sampler = RepeatSampler(
            data_source=dataset,
            mini_repeat_count=num_generations,
            shuffle=False,
        )
        dataloader_params = {
            "batch_size": batch_size,
            "num_workers": 8,
            "sampler": sampler,
            "collate_fn": collate_fn,
        }
        dataloader = DataLoader(dataset, **dataloader_params)

    dataloader = accelerator.prepare(dataloader)
    for step, batch in enumerate(dataloader):
        ids = batch["id"]
        accelerator.print(f"Step {step}")
        accelerator.wait_for_everyone()
        print(f"RANK {accelerator.process_index} ids: {list(ids)}")
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    fire.Fire(main)
