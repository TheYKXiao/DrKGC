import os
import argparse

import bitsandbytes as bnb
import torch

import transformers
from transformers import AutoConfig,  GenerationConfig
from transformers import AutoTokenizer, LlamaTokenizer, PreTrainedTokenizer
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, HfArgumentParser
from transformers import set_seed, Seq2SeqTrainer, BitsAndBytesConfig


from peft.tuners.lora import LoraLayer
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM, prepare_model_for_kbit_training

from arguments import Arguments, FinetuningArguments, GenerationArguments
from data import make_data_module
from model import GraphEnhancer, DrKGC

def get_accelerate_model(args, config, pretrained_model_class):
    device_map = 'auto' if os.environ.get('LOCAL_RANK') is None else {'': int(os.environ.get('LOCAL_RANK', '0'))}
    
   
    if args.use_quant:
        compute_dtype = torch.bfloat16 
        model = pretrained_model_class.from_pretrained(
            args.model_name_or_path,
            config=config,
            device_map='auto',
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=args.bits == 4,
                load_in_8bit=args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=args.double_quant,
                bnb_4bit_quant_type=args.quant_type,
            ),
            torch_dtype=torch.bfloat16,
        )
    else:
        model = pretrained_model_class.from_pretrained(
            args.model_name_or_path, 
            config=config,
            low_cpu_mem_usage=True, 
            device_map=device_map, 
        )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.use_quant)
    
    if args.model_type == "llama":
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif args.model_type == "mistral":
        config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
            ],
)
        
    model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model

        

class SavePeftModelCallback(transformers.TrainerCallback):
    KEEP_FILES = {
        "adapter_model.bin",
        "adapter_config.json",
        "graph_model.bin",
        "README.md",
    }

    def on_save(self, args, state, control, **kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = state.best_model_checkpoint
            print(f"Saving the best checkpoint to: {checkpoint_folder}")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            print(f"Saving checkpoint at step {state.global_step} to: {checkpoint_folder}")

        peft_model_path = checkpoint_folder
        kwargs["model"].save_pretrained(peft_model_path)

        # Comment out this code if need training status
        for file_name in os.listdir(checkpoint_folder):
            if file_name not in self.KEEP_FILES:
                os.remove(os.path.join(checkpoint_folder, file_name))

    def on_train_end(self, args, state, control, **kwargs):
        checkpoint_folder = os.path.join(args.output_dir, "checkpoint-final")
        print(f"Saving the final checkpoint to: {checkpoint_folder}")

        peft_model_path = checkpoint_folder
        kwargs["model"].save_pretrained(peft_model_path)




def train():
    set_seed(3407)

    hfparser = HfArgumentParser((Arguments, FinetuningArguments, GenerationArguments))
    (data_args, training_args, generation_args, _) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(**vars(data_args), **vars(training_args))
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Load LLM: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(data_args.model_name_or_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(['[QUERY]', '[ENTITY]', '[RELATION]'])

    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    
    if args.model_type == "llama":
        model = get_accelerate_model(args, model_config, LlamaForCausalLM)
    elif args.model_type == "mistral":
        model = get_accelerate_model(args, model_config, AutoModelForCausalLM)
    model.config.use_cache = False

    kge_embedding = torch.load(args.kge_embedding_path)
    kge_embedding_dim = kge_embedding.shape[1]
    llm_config = model.config
    embed_model = GraphEnhancer(kge_embedding, kge_embedding_dim, 4, 128, 1, 1024, llm_config.hidden_size, llm_config.hidden_act)
    model = DrKGC(tokenizer, model, embed_model)

    data_module = make_data_module(args, tokenizer)
    
    trainer = Seq2SeqTrainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        **data_module,
    )

    trainer.add_callback(SavePeftModelCallback)
    
    # Training
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state() 

if __name__ == '__main__':
    train()

