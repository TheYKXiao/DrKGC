from dataclasses import dataclass
from typing import Sequence, Dict, Any

import torch
from torch.nn.utils.rnn import pad_sequence
import transformers
from .dataset import DataModule 


@dataclass
class QueryCollator:
    args: None
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        bos_id, eos_id = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id

        sources = [ex["input"] for ex in instances]
        targets = [ex["output"] for ex in instances]

        src_max = max(1, self.source_max_len - (1 if bos_id is not None else 0))
        tgt_max = max(1, self.target_max_len - (1 if eos_id is not None else 0))

        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=src_max,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=tgt_max,
            truncation=True,
            add_special_tokens=False,
        )

        source_input_ids = tokenized_sources_with_prompt['input_ids']
        target_input_ids = tokenized_targets['input_ids']

        input_ids = []
        labels = []
        for src_ids, tgt_ids in zip(source_input_ids, target_input_ids):
            seq = [bos_id] + src_ids + tgt_ids + [eos_id]
            input_ids.append(torch.tensor(seq, dtype=torch.long))
            lab = torch.full((len(seq),), -100, dtype=torch.long)
            start = len(src_ids) + 1
            lab[start:] = torch.tensor(tgt_ids + [eos_id], dtype=torch.long)
            labels.append(lab)

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        query_ids = torch.tensor([ex['query_entity_id'] for ex in instances], dtype=torch.long)
        entity_ids = torch.tensor([ex['rank_entities_id'] for ex in instances], dtype=torch.long)
        subgraph = [ex['subgraph'] for ex in instances]
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': (input_ids != self.tokenizer.pad_token_id).long(),
            'labels': labels,
            "query_ids": query_ids,
            "entity_ids": entity_ids,
            "subgraph": subgraph,
        }

        return data_dict

def make_data_module(args, tokenizer: transformers.PreTrainedTokenizer):
    data_module = DataModule(args, tokenizer)
    data_collator = QueryCollator(
        args=args, tokenizer=tokenizer, 
        source_max_len=args.source_max_len, target_max_len=args.target_max_len
    )

    return {
        'train_dataset': data_module.train_ds,
        'eval_dataset': data_module.eval_ds,
        'data_collator': data_collator,
    }


