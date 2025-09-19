import os
import json
import numpy as np
from time import time
from tqdm import trange, tqdm
import argparse
from pathlib import Path

import bitsandbytes as bnb
import torch

import transformers
from transformers import AutoConfig,  GenerationConfig
from transformers import AutoTokenizer, LlamaTokenizer, PreTrainedTokenizer
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, HfArgumentParser
from transformers import set_seed, Seq2SeqTrainer, BitsAndBytesConfig

from peft.tuners.lora import LoraLayer
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM, prepare_model_for_kbit_training, PeftModel

from arguments import Arguments, FinetuningArguments, GenerationArguments
from data import DataModule, QueryCollator
from model import GraphEnhancer, DrKGC

from torch.cuda.amp import autocast

import torch
torch.cuda.empty_cache()


class Evaluator:
    def __init__(self, args, tokenizer, model, data_module, generation_config):
        self.args = args
        self.generation_config = generation_config

        self.tokenizer = tokenizer
        self.model = model
        self.data_module = data_module

        self.output_dir = os.path.dirname(args.checkpoint_dir)
        self.log_file_path = os.path.join(self.output_dir, 'metrics.txt')


    @torch.no_grad()
    def ranking_metrics(self, dataset):
        self.model.eval()

        preds = []
        ranks = np.array([])

        generated = []
        for ex_idx, ex in enumerate(tqdm(dataset)):
            prompt = ex['input']

            inputs = self.tokenizer(prompt, return_tensors='pt')
            input_ids = inputs.input_ids.cuda() 
            self.generation_config.eos_token_id = self.tokenizer.eos_token_id 

            subgraph = [ex['subgraph']] if 'subgraph' in ex else None
            
            output = self.model.generate(
                input_ids=input_ids, 
                query_ids=torch.LongTensor([ex['query_entity_id']]).to(input_ids.device), 
                entity_ids=torch.LongTensor([ex['rank_entities_id']]).to(input_ids.device), 
                subgraph=subgraph, 
                generation_config=self.generation_config,
            )
            generated.append(output.sequences[0].cpu().numpy().tolist())
            ex.pop('input')
        
        batch_preds = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        for ex_idx, ex in enumerate(dataset):
            target = ex.pop('output')
            rank = ex['rank']
            pred = str(batch_preds[ex_idx]).strip()

            topk_names = ex['rank_entities']
            if target == pred:
                rank = 1
            else:    
                if pred not in set(topk_names) or topk_names.index(pred) >= rank:
                    rank += 1
            
            ex['target'] = target
            ex['pred_rank'] = rank
            ex['pred'] = pred
            preds.append(ex)
            ranks = np.append(ranks, rank)
        
        metrics = {
        'mrr': np.mean(1. / ranks),
        'hits1': np.mean(ranks <= 1),
        'hits3': np.mean(ranks <= 3),
        'hits10': np.mean(ranks <= 10),
        }
        metrics = {k: round(v, 8) for k, v in metrics.items()}
        
        print("ranking metrics:")
        print(metrics)
        
        with open(self.log_file_path, 'w', encoding='utf-8') as log_file:
            log_line = f'ranking metrics: {metrics}\n'
            log_file.write(log_line)

        
        return preds


if __name__ == '__main__':
    set_seed(3407)
    
    hfparser = HfArgumentParser((Arguments, GenerationArguments))
    (data_args, generation_args, _) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    generation_config = GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(**vars(data_args))

    print(f"Load LLM: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(['[QUERY]', '[ENTITY]', '[RELATION]'])
    

    generation_config.bos_token_id = tokenizer.bos_token_id
    
    model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, low_cpu_mem_usage=True, device_map='auto')
    model = PeftModel.from_pretrained(model, args.checkpoint_dir)

    model = model.half()
    
    kge_embedding = torch.load(args.kge_embedding_path)
    kge_embedding_dim = kge_embedding.shape[1]
    llm_config = model.config
    embed_model = GraphEnhancer(kge_embedding, kge_embedding_dim, 4, 128, 1, 1024, llm_config.hidden_size, llm_config.hidden_act)
    ckpt_dir = Path(args.checkpoint_dir)  
    state = torch.load(ckpt_dir / "graph_model.bin", map_location="cpu")
    embed_model.load_state_dict(state)
        
    model = DrKGC(tokenizer, model, embed_model)

    model = model.half()
    
    model.cuda()
    model.eval()

    data_module = DataModule(args, tokenizer)

    evaluator = Evaluator(args, tokenizer, model, data_module, generation_config)

    with autocast():
        preds = evaluator.ranking_metrics(data_module.test_ds)
    output = {
        'args': vars(args),
        'generation_config': vars(generation_config),
        'prediction': preds,
    }
    output_path = os.path.join(os.path.dirname(args.checkpoint_dir), f'prediction.json')
    json.dump(output, open(output_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)