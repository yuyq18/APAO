import json
import os
import random
import datetime

import numpy as np
import torch

def parse_global_args(parser):
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_config_file", type=str, default="./config/models/t5_config.json",help="basic model path")
    parser.add_argument("--tokenizer_save_dir", type=str, default="./config/tokenizer", help="tokenizer save dir")
    parser.add_argument("--output_dir", type=str, default="./ckpt", help="The output directory")
    return parser

def parse_dataset_args(parser):
    parser.add_argument("--data_path", type=str, default="./data", help="data directory")
    parser.add_argument("--dataset", type=str, default="Office", help="Dataset name")
    parser.add_argument("--index_file", type=str, default=".index.rq-kmeans.json", help="the item indices file")
    parser.add_argument("--max_his_len", type=int, default=20, help="the max number of items in history sequence, -1 means no limit")
    parser.add_argument("--decoder_only_data_mode", type=str, default="inter-level", choices=["user-level", "inter-level"], help="the data mode for decoder-only model")
    return parser

def parse_train_args(parser):
    parser.add_argument("--optim", type=str, default="adamw_torch", help='The name of the optimizer')
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--per_device_batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--logging_step", type=int, default=10)
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=False, help="whether resume from the last checkpoint of output_dir")
    parser.add_argument("--train_sample_num", type=int, default=-1, help="train sample number, -1 represents using all train data")
    parser.add_argument("--eval_sample_num", type=int, default=-1, help="eval sample number, -1 represents using all eval data")
    parser.add_argument("--warmup_ratio", type=float, default=0.01, help="warmup ratio for learning rate scheduler")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="learning rate scheduler type")
    parser.add_argument("--save_and_eval_strategy", type=str, default="epoch", help="save and eval strategy")
    parser.add_argument("--save_and_eval_steps", type=int, default=1000, help="save and eval steps")
    parser.add_argument("--fp16",  action="store_true", default=False, help="whether to use fp16")
    parser.add_argument("--bf16", action="store_true", default=False, help="whether to use bf16")
    parser.add_argument("--wandb_run_name", type=str, default="default")

    parser.add_argument("--from_pretrained", action="store_true", default=False, help="Whether to load model from pretrained.")
    parser.add_argument("--pretrained_model_path", type=str, default="./config/models/pretrained_model_path_config.json", help="Path to the pretrained model.")
    
    return parser

def parse_test_args(parser):

    parser.add_argument("--ckpt_path", type=str, default="./ckpt", help="The checkpoint path")
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--results_file", type=str, default="./results", help="result output path")

    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--is_constrained_beam_search", action="store_true", default=False, help="whether use constrained beam search")
    parser.add_argument("--sample_num", type=int, default=-1, help="test sample number, -1 represents using all test data")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID when testing with single GPU")
    parser.add_argument("--metrics", type=str, default="hit@10,hit@20,ndcg@10,ndcg@20", help="test metrics, separate by comma")
    return parser

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def prefix_allowed_tokens_fn(candidate_trie):
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        trie_out = candidate_trie.get(sentence)
        return trie_out
    return prefix_allowed_tokens

def prefix_allowed_tokens_fn_for_decoderonly(candidate_trie, eos_token_id=0):
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        x = len(sentence) % 4
        trie_out = None
        # If aligned, it means we want to start generating a new item from the BOS (i.e., eos_id) position
        if x == 0:
            trie_out = candidate_trie.get([eos_token_id])
        else:
            # If not aligned, we want to continue the current item generation
            trie_out = candidate_trie.get([eos_token_id]+sentence[-x:])
        return trie_out
    return prefix_allowed_tokens

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data