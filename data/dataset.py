from typing import Dict
import json
from pathlib import Path

import torch
import transformers
from torch.utils.data import Dataset


class QueryDataset(Dataset):
    def __init__(self, examples):
        self.data = examples

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict:
        return self.data[idx]


class DataModule:
    def __init__(self, args, tokenizer: transformers.PreTrainedTokenizer):
        self.args = args
        self.tokenizer = tokenizer

        base = Path(args.dataset_path)
        with (base / "train.json").open("r", encoding="utf-8") as f:
            train_json = json.load(f)
        with (base / "valid.json").open("r", encoding="utf-8") as f:
            eval_json = json.load(f)
        with (base / "test.json").open("r", encoding="utf-8") as f:
            test_json = json.load(f)
        
        self.train_ds = QueryDataset(train_json)
        self.eval_ds = QueryDataset(eval_json)
        self.test_ds = QueryDataset(test_json)
 