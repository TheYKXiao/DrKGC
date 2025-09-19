# DrKGC
This repository contains the code for the paper:  
**"	DrKGC: Dynamic Subgraph Retrieval-Augmented LLMs for Knowledge Graph Completion across General and Biomedical Domains"**  
<!-- [[Paper Link]](https://arxiv.org/abs/xxxx.xxxxx) | [[EMNLP Anthology]](https://aclanthology.org/xxxx) -->


## Requirements
**Note:** Please install the appropriate PyTorch and DGL version with CUDA support. 
```bash
# python 3.12
torch==2.3.1+cu118
dgl==2.4.0+cu118
ogb==1.3.6
networkx==3.4.2
transformers==4.38.2
peft==0.4.0
accelerate==0.27.2
bitsandbytes==0.40.2
safetensors==0.4.3
tokenizers==0.15.2
datasets==2.20.0
```

## 🚀 Quick Start
### 💾 Download the preprocessed data
For convenience, we provide preprocessed datasets that include coarse ranking results, retrieved subgraphs, and structured entity embeddings: 📥 [Download](https://drive.google.com/file/d/1GfJFHX5bT-mp182MooMKYkcJA43i7S1P/view?usp=sharing)

Please unzip the file and place the contents under the `./dataset` directory.

### ⚡ Fast Run on WN18RR
**Step 1:** Train DrKGC on WN18RR:
```bash
python main.py --dataset_path "dataset/wn18rr" --kge_embedding_path "dataset/wn18rr/entity_embeddings.pt" --model_name_or_path "meta-llama/Meta-Llama-3-8B" --model_type llama --use_quant True --bf16 --output_dir "results/wn18rr/llama3" --num_train_epochs 10 --per_device_train_batch_size 8 --learning_rate 2e-4 --lora_r 32 --lora_alpha 32 --lora_dropout 0.1 --save_strategy steps --save_steps 200 --save_total_limit 10
```

**Step 2:** Inference & Evaluation on WN18RR:
```bash
python infer.py --dataset_path "dataset/wn18rr" --kge_embedding_path "dataset/wn18rr/entity_embeddings.pt" --checkpoint_dir "results/wn18rr/llama3/checkpoint-final" --model_name_or_path "meta-llama/Meta-Llama-3-8B" --model_type llama --num_return_sequences 1
```

### ⚡ Fast Run on FB15k-237
**Step 1:** Train DrKGC on FB15k-237:
```bash
python main.py --dataset_path "dataset/fb15k237" --kge_embedding_path "dataset/fb15k237/entity_embeddings.pt" --model_name_or_path "meta-llama/Meta-Llama-3-8B" --model_type llama --use_quant True --bf16 --output_dir "results/fb15k237/llama3" --num_train_epochs 4 --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --learning_rate 2e-4 --lora_r 32 --lora_alpha 32 --lora_dropout 0.1 --dataloader_num_workers 32 --save_strategy steps --save_steps 20 --save_total_limit 10
```

**Step 2:** Inference & Evaluation on FB15k-237:
```bash
python infer.py --dataset_path "dataset/fb15k237" --kge_embedding_path "dataset/fb15k237/entity_embeddings.pt" --checkpoint_dir "results/fb15k237/llama3/checkpoint-final" --model_name_or_path "meta-llama/Meta-Llama-3-8B" --model_type llama --num_return_sequences 1
```


## Citation
If you use this code, please cite our paper:
```bibtex
@article{xiao2025drkgc,
  title={DrKGC: Dynamic Subgraph Retrieval-Augmented LLMs for Knowledge Graph Completion across General and Biomedical Domains},
  author={Xiao, Yongkang and Zhang, Sinian and Dai, Yi and Zhou, Huixue and Hou, Jue and Ding, Jie and Zhang, Rui},
  journal={arXiv preprint arXiv:2506.00708},
  year={2025}
}
```