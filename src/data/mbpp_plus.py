from pathlib import Path

from datasets import load_dataset
from data.data import DEFAULT_CACHE, Dataset


class MBPPPlus(Dataset):
    HF_REPO = "evalplus/mbppplus"

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE, split: str = "test") -> None:
        super().__init__() 
        self.data = load_dataset(self.HF_REPO, split=split, cache_dir=str(cache_dir))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]

